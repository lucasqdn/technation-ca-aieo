"""
diagnose.py  —  run this in your project to pinpoint why visibility=0 and
hallucination=0.  It checks each failure mode independently.
Drop it next to your other scripts and run: python diagnose.py
"""
import sqlite3, sys, re
from pathlib import Path
from difflib import SequenceMatcher

# ── adjust if your paths differ ──────────────────────────────────────────────
try:
    from config import DB_PATH, CHROMA_DIR
except ImportError:
    print("ERROR: run from your project root so config.py is importable.")
    sys.exit(1)

SEP = "─" * 60

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic DB sanity
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("1. DATABASE SANITY")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

for table in ("programs", "classifications", "questions", "ai_responses", "evaluations"):
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"   {table:<20} {n:>6} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sample one response + its relevant programs
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("2. SAMPLE RESPONSE + RELEVANT PROGRAMS")

row = conn.execute("""
    SELECT r.response_id, r.question_id, r.platform,
           substr(r.response_text,1,300) AS snippet,
           q.target_population, q.target_sector, q.target_region
    FROM ai_responses r
    JOIN questions q ON r.question_id = q.question_id
    LIMIT 1
""").fetchone()

if not row:
    print("   No responses found — run build_db.py first.")
    sys.exit(1)

print(f"   response_id      : {row['response_id']}")
print(f"   question_id      : {row['question_id']}")
print(f"   platform         : {row['platform']}")
print(f"   target_population: {row['target_population']}")
print(f"   target_sector    : {row['target_sector']}")
print(f"   target_region    : {row['target_region']}")
print(f"   response snippet : {row['snippet']!r}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. What does get_relevant_programs return for this question?
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("3. RELEVANT PROGRAMS FOR THIS QUESTION")

pop = row["target_population"] or ""
sec = row["target_sector"] or ""
reg = row["target_region"] or ""

clauses, params = [], []
if pop and pop != "general_public":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='population' AND c.value=?)")
    params.append(pop)
if sec and sec != "public_services":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='sector' AND c.value=?)")
    params.append(sec)
if reg and reg != "national":
    clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='region' AND c.value IN (?, 'national'))")
    params.append(reg)

base = "SELECT DISTINCT p.entity_id, p.name_en FROM programs p WHERE p.is_active = 1"
q = (base + " AND " + " AND ".join(clauses)) if clauses else base + " LIMIT 150"
relevant = conn.execute(q, params).fetchall()
print(f"   relevant programs found: {len(relevant)}")
for r2 in relevant[:5]:
    print(f"   • [{r2['entity_id']}] {r2['name_en']}")
if len(relevant) > 5:
    print(f"   … and {len(relevant)-5} more")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ChromaDB entity_id type check — the most common silent bug
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("4. CHROMADB ENTITY_ID TYPE CHECK")
try:
    from embed import ProgramEmbedder
    embedder = ProgramEmbedder()
    chroma_count = embedder.collection.count()
    print(f"   ChromaDB documents: {chroma_count}")

    # Peek at a few stored IDs
    sample = embedder.collection.get(limit=3, include=["metadatas"])
    stored_ids = sample["ids"]
    stored_meta_eids = [m.get("entity_id") for m in sample["metadatas"]]
    print(f"   Sample stored IDs (ChromaDB string keys) : {stored_ids}")
    print(f"   Sample entity_id in metadata (type check): {[(v, type(v).__name__) for v in stored_meta_eids]}")

    if relevant:
        db_eid = relevant[0]["entity_id"]
        print(f"   SQLite entity_id example (type): {db_eid!r} ({type(db_eid).__name__})")
        # Check if the int version matches a stored key
        chroma_hit = embedder.collection.get(ids=[str(db_eid)], include=["metadatas"])
        found = bool(chroma_hit["ids"])
        print(f"   Does SQLite entity_id {db_eid} exist in ChromaDB as '{db_eid}'? {found}")
except Exception as e:
    print(f"   ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Raw similarity scores — what does a real query return?
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("5. RAW SIMILARITY SCORES  (response text → ChromaDB top-10)")
try:
    response_text = conn.execute(
        "SELECT response_text FROM ai_responses WHERE response_id=?",
        (row["response_id"],)
    ).fetchone()["response_text"] or ""

    hits = embedder.query(response_text, n_results=10)
    relevant_id_set = {r2["entity_id"] for r2 in relevant}
    print(f"   Top-10 hits (entity_id → similarity):")
    for h in hits:
        in_relevant = "✓ RELEVANT" if h["entity_id"] in relevant_id_set else ""
        print(f"   [{h['entity_id']:>6}] sim={h['similarity']:.4f}  dist={h['distance']:.4f}  {in_relevant}")

    print(f"\n   Any of top-10 in relevant set? "
          f"{any(h['entity_id'] in relevant_id_set for h in hits)}")

    # Wider fetch
    wide_hits = embedder.query(response_text, n_results=min(500, chroma_count))
    wide_relevant = [h for h in wide_hits if h["entity_id"] in relevant_id_set]
    print(f"   Top-500 hits that are in relevant set: {len(wide_relevant)}")
    if wide_relevant:
        print(f"   Their similarities: {[h['similarity'] for h in wide_relevant[:10]]}")
        print(f"   Max similarity among relevant: {max(h['similarity'] for h in wide_relevant):.4f}")
    else:
        print("   ⚠ NONE of the relevant programs appear in top-500 hits!")
        print("   This means the response text and program documents are in very")
        print("   different parts of embedding space — likely an embedding mismatch.")

except Exception as e:
    print(f"   ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Hallucination: does name extraction work?
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("6. HALLUCINATION CANDIDATE EXTRACTION")

PROGRAM_SIGNAL_WORDS = {
    "program", "programme", "fund", "funding", "initiative", "strategy",
    "benefit", "grant", "support", "services", "service", "centre", "center",
    "office", "agency", "act", "plan", "project", "partnership", "network",
    "council", "commission", "institute", "foundation", "authority",
}
MIN_LEN = 8

def extract_candidates(text):
    candidates = set()
    for m in re.finditer(r"\b(?:[A-Z][a-z]{1,}\s+){1,7}[A-Z][a-z]{1,}\b", text):
        if len(m.group()) >= MIN_LEN:
            candidates.add(m.group().strip())
    pat = re.compile(
        r"\b(?:[A-Z][A-Za-z\-']{1,}\s+){0,6}"
        r"(?:" + "|".join(re.escape(w) for w in PROGRAM_SIGNAL_WORDS) + r")"
        r"(?:\s+[A-Z][A-Za-z\-']{1,}){0,4}\b", re.IGNORECASE)
    for m in pat.finditer(text):
        if len(m.group()) >= MIN_LEN:
            candidates.add(m.group().strip())
    return list(candidates)

sample_response = conn.execute(
    "SELECT response_text FROM ai_responses LIMIT 1"
).fetchone()["response_text"] or ""

candidates = extract_candidates(sample_response)
print(f"   Candidates extracted from sample response: {len(candidates)}")
for c in candidates[:10]:
    print(f"   • {c!r}")
if not candidates:
    print("   ⚠ Zero candidates extracted — response may be all-lowercase or")
    print("   lack title-cased phrases. Check response_text encoding/format.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Quick fuzzy match test
# ─────────────────────────────────────────────────────────────────────────────
if candidates and relevant:
    print(SEP)
    print("7. FUZZY MATCH: first candidate vs first 5 ground-truth names")
    cand = candidates[0]
    gt_names = [r2["name_en"] for r2 in relevant[:5] if r2["name_en"]]
    for gt in gt_names:
        ratio = SequenceMatcher(None, cand.lower(), gt.lower()).ratio()
        print(f"   {cand!r} vs {gt!r}  →  {ratio:.3f}")

print(SEP)
print("DONE. Share this output to pinpoint the fix needed.")
conn.close()