# hallucination_diag.py
# Shows exactly which candidates are being extracted from responses,
# their fuzzy scores against the allowlist, and their ChromaDB scores.
# Run: python hallucination_diag.py

import re
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path

from config import DB_PATH

# ── copy of the same constants from evaluate.py ──────────────────────────────
ALLOWLIST_FUZZY_THRESHOLD   = 0.42
HALLUCINATION_SIM_THRESHOLD = 0.48
MIN_CANDIDATE_TOKENS = 3
MIN_CANDIDATE_LEN    = 10

PROGRAM_SIGNAL_WORDS = {
    "program", "programme", "fund", "funding", "initiative", "strategy",
    "benefit", "grant", "support", "services", "service", "centre", "center",
    "office", "agency", "act", "plan", "project", "partnership", "network",
    "council", "commission", "institute", "foundation", "authority",
    "association", "society", "board", "bureau",
}
DISQUALIFYING_STARTS = {
    "this", "these", "there", "they", "their", "the", "a", "an",
    "as", "at", "by", "for", "from", "if", "in", "it", "its",
    "many", "most", "much", "no", "not", "of", "on", "or",
    "some", "such", "that", "to", "we", "with", "you", "your",
    "several", "various", "other", "also", "both", "each",
    "local", "federal", "provincial", "national", "canadian",
    "includes", "funding", "personalized", "skilled", "service",
    "indigenous", "bilateral", "canada",
}

SEED_PROGRAMS = [
    "Canada Summer Jobs", "Skills Link", "Canada Student Grant",
    "Canada Student Loan", "Canada Learning Bond",
    "Apprenticeship Incentive Grant", "Apprenticeship Completion Grant",
    "Sectoral Workforce Solutions Program", "Workforce Development Agreements",
    "Indigenous Skills and Employment Training", "ISET Program",
    "First Nations and Inuit Youth Employment Strategy",
    "First Nations and Inuit Summer Work Experience Program",
    "Aboriginal Skills and Employment Training Strategy",
    "Opportunities Fund for Persons with Disabilities",
    "Enabling Accessibility Fund", "Social Development Partnerships Program",
    "New Horizons for Seniors Program", "Settlement Program",
    "Language Instruction for Newcomers to Canada", "LINC Program",
    "Resettlement Assistance Program", "Employment Insurance",
    "Canada Child Benefit", "Guaranteed Income Supplement",
    "Old Age Security", "Registered Disability Savings Plan",
    "Disability Tax Credit", "Canada Pension Plan Disability Benefit",
    "Post-Secondary Student Support Program",
    "Youth Employment and Skills Strategy",
    "Student Work Placement Program",
    "Women Entrepreneurship Strategy", "Women Entrepreneurship Fund",
    "Black Entrepreneurship Program", "Indigenous Entrepreneurship Program",
    "Universal Broadband Fund", "Aboriginal Head Start",
    "Community Action Program for Children", "Wellness Together Canada",
    "Hope for Wellness Help Line", "Reaching Home", "Canada Housing Benefit",
]

SEP = "─" * 70

def _normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s)

def _fuzzy(a, b):
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

def _extract_candidates(text):
    candidates = set()
    title_run = re.compile(r"\b(?:[A-Z][a-z]{1,25}\s+){2,6}[A-Z][a-z]{1,25}\b")
    for m in title_run.finditer(text):
        phrase = m.group().strip()
        tokens = phrase.split()
        if len(tokens) >= MIN_CANDIDATE_TOKENS and len(phrase) >= MIN_CANDIDATE_LEN and tokens[0].lower() not in DISQUALIFYING_STARTS:
            candidates.add(phrase)
    signal_pat = re.compile(r"\b([A-Z][A-Za-z\-']{1,}(?:\s+[A-Za-z\-']{1,}){1,8})\b")
    for m in signal_pat.finditer(text):
        phrase = m.group().strip()
        tokens = phrase.split()
        lower_tokens = {t.lower() for t in tokens}
        if (lower_tokens & PROGRAM_SIGNAL_WORDS and len(tokens) >= MIN_CANDIDATE_TOKENS
                and len(phrase) >= MIN_CANDIDATE_LEN and tokens[0].lower() not in DISQUALIFYING_STARTS
                and tokens[0][0].isupper()):
            candidates.add(phrase)
    sorted_cands = sorted(candidates, key=len, reverse=True)
    pruned = []
    for cand in sorted_cands:
        norm = _normalize(cand)
        if not any(norm in _normalize(k) and norm != _normalize(k) for k in pruned):
            pruned.append(cand)
    return pruned

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Build allowlist
    rows = conn.execute("SELECT name_en FROM programs WHERE is_active=1 AND name_en IS NOT NULL").fetchall()
    db_names = [r["name_en"] for r in rows if r["name_en"].strip()]
    allowlist = list({n for n in db_names + SEED_PROGRAMS if n.strip()})
    print(f"Allowlist size: {len(allowlist)} names ({len(db_names)} from DB + {len(SEED_PROGRAMS)} seeds)\n")

    # Load embedder
    try:
        from embed import ProgramEmbedder
        embedder = ProgramEmbedder()
        use_embed = True
    except Exception as e:
        print(f"Warning: embedder not available ({e}) — skipping ChromaDB scores\n")
        embedder = None
        use_embed = False

    # Sample 2 responses per platform
    platforms = [r["platform"] for r in conn.execute("SELECT DISTINCT platform FROM ai_responses").fetchall()]

    for platform in platforms:
        responses = conn.execute(
            "SELECT r.response_id, r.question_id, r.response_text FROM ai_responses r WHERE r.platform=? LIMIT 2",
            (platform,)
        ).fetchall()

        print(SEP)
        print(f"PLATFORM: {platform.upper()}")
        print(SEP)

        for resp in responses:
            print(f"\nQ={resp['question_id']}  response_id={resp['response_id']}")
            text = resp["response_text"] or ""
            print(f"Response preview: {text[:120]}...")

            candidates = _extract_candidates(text)
            print(f"\nExtracted {len(candidates)} candidates:")

            for cand in candidates:
                # Fuzzy vs allowlist
                best_name, best_ratio = "", 0.0
                for name in allowlist:
                    r = _fuzzy(cand, name)
                    nc, nn = _normalize(cand), _normalize(name)
                    if nc in nn or nn in nc:
                        r = max(r, 0.70)
                    if r > best_ratio:
                        best_ratio, best_name = r, name

                fuzzy_pass = best_ratio >= ALLOWLIST_FUZZY_THRESHOLD

                # ChromaDB
                embed_sim = 0.0
                if use_embed:
                    hits = embedder.query(cand, n_results=1)
                    embed_sim = hits[0]["similarity"] if hits else 0.0
                embed_pass = embed_sim >= HALLUCINATION_SIM_THRESHOLD

                verdict = "✓ REAL" if (fuzzy_pass or embed_pass) else "⚠ HALLUCINATED"
                print(
                    f"  {verdict}  '{cand[:55]}'\n"
                    f"           fuzzy={best_ratio:.2f} vs '{best_name[:40]}'  "
                    f"embed={embed_sim:.2f}  "
                    f"[fuzzy_pass={fuzzy_pass}, embed_pass={embed_pass}]"
                )

        print()

    # Summary stats from evaluations table
    print(SEP)
    print("EVALUATION SUMMARY FROM DB")
    rows = conn.execute("""
        SELECT platform,
               COUNT(*) as n,
               SUM(has_hallucination) as flagged,
               AVG(has_hallucination) as rate
        FROM evaluations GROUP BY platform
    """).fetchall()
    for r in rows:
        print(f"  {r['platform']:<12} flagged={r['flagged']}/{r['n']}  rate={r['rate']:.2f}")

    # Show sample hallucination notes
    print(f"\nSAMPLE hallucination_note VALUES (first 5 flagged):")
    notes = conn.execute("""
        SELECT platform, question_id, hallucination_note
        FROM evaluations
        WHERE has_hallucination=1 AND hallucination_note != ''
        LIMIT 5
    """).fetchall()
    for n in notes:
        print(f"  [{n['platform']}] {n['question_id']}: {n['hallucination_note'][:120]}")

    conn.close()

if __name__ == "__main__":
    main()