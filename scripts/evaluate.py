# evaluate.py
# Scores AI responses against the ground truth program database.
#
# ── HOW THE program_text TABLE CHANGES THINGS ────────────────────────────────
#
# Previously, ChromaDB documents were built from name + description[:500].
# Descriptions came from CKAN metadata — publication abstracts and report
# summaries, not program content. The embedder had nothing useful to match
# against when an LLM response mentioned "Skills Link" or "Canada Summer Jobs."
#
# Now that fetch.py also retrieves the actual HTML/PDF content of each
# program's URL, embed.py builds richer ChromaDB documents from that content
# (up to 2000 chars). This means:
#
#   VISIBILITY uses a two-signal approach:
#     Signal A — name-mention: did the response mention this program by name?
#                (fuzzy sentence matching — works regardless of embedding quality)
#     Signal B — semantic: does the program's ChromaDB document embed close to
#                any chunk of the response?
#                (only applied to programs with has_full_text=1, because
#                 metadata-only documents still don't embed reliably)
#     A program is "covered" if EITHER signal fires.
#
#   HALLUCINATION: unchanged logic — allowlist fuzzy match first, ChromaDB
#     fallback second. Full_text improves the ChromaDB fallback quality.
#
# ── VISIBILITY SCORING ───────────────────────────────────────────────────────
#
#   For each relevant program:
#     Signal A: split response into sentences, fuzzy-match each against name_en.
#               Short names (< 4 words) use substring containment only.
#     Signal B: split response into chunks (≤ MAX_CHUNKS sentences), query
#               ChromaDB top-CHUNK_K per chunk, check if this entity_id appears
#               with similarity >= CHUNK_SIM_THRESHOLD.
#               Skipped for programs without full_text (not reliable).
#
#   visibility_score = (A OR B covered) / total_relevant

import re
import sqlite3
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from config import DB_PATH

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Signal A — name mention
NAME_MATCH_THRESHOLD = 0.60
MIN_WORDS_FOR_FUZZY  = 2

# Signal B — semantic chunk similarity (full-text programs only)
CHUNK_SIM_THRESHOLD  = 0.50
MAX_CHUNKS           = 12
CHUNK_K              = 40

# Hallucination
ALLOWLIST_FUZZY_THRESHOLD   = 0.525
HALLUCINATION_SIM_THRESHOLD = 0.525

# General
MAX_RELEVANT_PROGRAMS = 30

# Candidate name extraction
MIN_CANDIDATE_TOKENS = 2
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

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s)

def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\n])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]

def _program_covered_by_name(
    name: str, sentences: list[str], full_text: str
) -> bool:
    """Signal A: did the response explicitly mention this program by name?"""
    words     = name.split()
    norm_name = _normalize(name)
    norm_full = _normalize(full_text)

    if len(words) < MIN_WORDS_FOR_FUZZY:
        return norm_name in norm_full

    for sentence in sentences:
        if _fuzzy_ratio(name, sentence) >= NAME_MATCH_THRESHOLD:
            return True
        if norm_name in _normalize(sentence):
            return True

    return norm_name in norm_full

def _extract_candidate_names(text: str) -> list[str]:
    """Extract likely program/org name candidates from LLM response."""
    candidates: set[str] = set()

    title_run = re.compile(r"\b(?:[A-Z][a-z]{1,25}\s+){2,6}[A-Z][a-z]{1,25}\b")
    for m in title_run.finditer(text):
        phrase = m.group().strip()
        tokens = phrase.split()
        if (
            len(tokens) >= MIN_CANDIDATE_TOKENS
            and len(phrase) >= MIN_CANDIDATE_LEN
            and tokens[0].lower() not in DISQUALIFYING_STARTS
        ):
            candidates.add(phrase)

    signal_pat = re.compile(
        r"\b([A-Z][A-Za-z\-']{1,}(?:\s+[A-Za-z\-']{1,}){1,8})\b"
    )
    for m in signal_pat.finditer(text):
        phrase = m.group().strip()
        tokens = phrase.split()
        lower_tokens = {t.lower() for t in tokens}
        if (
            lower_tokens & PROGRAM_SIGNAL_WORDS
            and len(tokens) >= MIN_CANDIDATE_TOKENS
            and len(phrase) >= MIN_CANDIDATE_LEN
            and tokens[0].lower() not in DISQUALIFYING_STARTS
            and tokens[0][0].isupper()
        ):
            candidates.add(phrase)

    sorted_cands = sorted(candidates, key=len, reverse=True)
    pruned: list[str] = []
    for cand in sorted_cands:
        norm = _normalize(cand)
        if not any(
            norm in _normalize(k) and norm != _normalize(k) for k in pruned
        ):
            pruned.append(cand)
    return pruned

def _get_embedder():
    try:
        from embed import ProgramEmbedder
        return ProgramEmbedder()
    except ImportError as e:
        print(f"ERROR loading embedder: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ResponseEvaluator:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.embedder = _get_embedder()
        self._full_text_ids: set[int] = self._load_full_text_ids()
        self._name_allowlist: list[str] = self._build_allowlist()
        print(
            f"ResponseEvaluator ready. "
            f"Full-text programs: {len(self._full_text_ids)}, "
            f"Allowlist: {len(self._name_allowlist)} names."
        )

    def _load_full_text_ids(self) -> set[int]:
        """
        Entity IDs with successfully fetched full_text.
        Signal B (semantic) is only applied to these — metadata-only embeddings
        are not reliable enough to use as a coverage signal.
        """
        try:
            rows = self.conn.execute(
                "SELECT entity_id FROM program_text WHERE fetch_status = 'ok'"
            ).fetchall()
            return {r["entity_id"] for r in rows}
        except sqlite3.OperationalError:
            return set()  # table doesn't exist yet

    def _build_allowlist(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name_en FROM programs WHERE is_active = 1 AND name_en IS NOT NULL"
        ).fetchall()
        return [r["name_en"] for r in rows if r["name_en"].strip()]

    # ------------------------------------------------------------------
    # Ground-truth retrieval
    # ------------------------------------------------------------------

    def get_relevant_programs(
        self,
        target_population: str,
        target_sector: str,
        target_region: str,
    ) -> list[dict]:
        base   = "SELECT DISTINCT p.entity_id, p.name_en, p.description FROM programs p WHERE p.is_active = 1"
        clauses: list[str] = []
        params:  list[str] = []

        if target_population and target_population not in ("general_public",):
            clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='population' AND c.value=?)")
            params.append(target_population)

        if target_sector and target_sector not in ("public_services",):
            clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='sector' AND c.value=?)")
            params.append(target_sector)

        if target_region and target_region not in ("national",):
            clauses.append("EXISTS (SELECT 1 FROM classifications c WHERE c.entity_id=p.entity_id AND c.dimension='region' AND c.value IN (?, 'national'))")
            params.append(target_region)

        query = base + (" AND " + " AND ".join(clauses) if clauses else f" LIMIT {MAX_RELEVANT_PROGRAMS}")
        rows  = self.conn.execute(query, params).fetchall()
        return [
            {"entity_id": r["entity_id"], "name_en": r["name_en"] or "", "description": r["description"] or ""}
            for r in rows[:MAX_RELEVANT_PROGRAMS]
        ]

    # ------------------------------------------------------------------
    # Visibility — two-signal
    # ------------------------------------------------------------------

    def compute_visibility(
        self,
        response_text: str,
        relevant_programs: list[dict],
    ) -> tuple[float, int, int, float]:
        """
        Returns (visibility_score, covered, gap_count, top_match_score).
        Signal A (name-mention) always runs.
        Signal B (chunk-semantic) runs only for full-text programs.
        """
        total = len(relevant_programs)
        if total == 0:
            return 0.0, 0, 0, 0.0

        sentences = _sentence_split(response_text)

        # Pre-compute chunk results for Signal B
        relevant_ft_ids = {
            p["entity_id"] for p in relevant_programs
            if p["entity_id"] in self._full_text_ids
        }
        chunk_hits: dict[int, float] = {}
        if relevant_ft_ids:
            for chunk in sentences[:MAX_CHUNKS]:
                if len(chunk.strip()) < 15:
                    continue
                for hit in self.embedder.query(chunk, n_results=CHUNK_K):
                    eid = hit["entity_id"]
                    if eid in relevant_ft_ids:
                        chunk_hits[eid] = max(chunk_hits.get(eid, 0.0), hit["similarity"])

        covered = 0
        best_scores: list[float] = []

        for program in relevant_programs:
            eid  = program["entity_id"]
            name = program["name_en"]

            signal_a = _program_covered_by_name(name, sentences, response_text)
            signal_b = (
                eid in relevant_ft_ids
                and chunk_hits.get(eid, 0.0) >= CHUNK_SIM_THRESHOLD
            )

            if signal_a or signal_b:
                covered += 1
                best_scores.append(1.0)
            else:
                name_best = max((_fuzzy_ratio(name, s) for s in sentences), default=0.0)
                sem_best  = chunk_hits.get(eid, 0.0)
                best_scores.append(max(name_best, sem_best))

        top_match_score  = max(best_scores, default=0.0)
        visibility_score = round(covered / total, 4) if total > 0 else 0.0
        return visibility_score, covered, total - covered, round(top_match_score, 4)

    # ------------------------------------------------------------------
    # Hallucination — allowlist-first, ChromaDB fallback
    # ------------------------------------------------------------------

    def detect_hallucinations(
        self,
        response_text: str,
        relevant_programs: list[dict],
    ) -> tuple[int, str]:
        """
        A candidate is hallucinated only if it fails BOTH:
          (a) fuzzy match >= ALLOWLIST_FUZZY_THRESHOLD against all name_en
          (b) ChromaDB similarity >= HALLUCINATION_SIM_THRESHOLD

        This prevents real programs from being flagged just because ChromaDB
        didn't index them well (the root cause of the 80% hallucination rate).
        """
        if not response_text.strip():
            return 0, ""

        candidates = _extract_candidate_names(response_text)
        if not candidates:
            return 0, ""

        hallucinated: list[str] = []

        for candidate in candidates:
            # Primary: allowlist fuzzy match
            best_name, best_ratio = "", 0.0
            for name in self._name_allowlist:
                r = _fuzzy_ratio(candidate, name)
                norm_c, norm_n = _normalize(candidate), _normalize(name)
                if norm_c in norm_n or norm_n in norm_c:
                    r = max(r, 0.70)
                if r > best_ratio:
                    best_ratio, best_name = r, name

            if best_ratio >= ALLOWLIST_FUZZY_THRESHOLD:
                continue

            # Fallback: ChromaDB embedding
            hits     = self.embedder.query(candidate, n_results=1)
            best_sim = hits[0]["similarity"] if hits else 0.0
            if best_sim >= HALLUCINATION_SIM_THRESHOLD:
                continue

            hallucinated.append(
                f"'{candidate}' (allowlist='{best_name}' @ {best_ratio:.2f}, embed={best_sim:.2f})"
            )

        if hallucinated:
            return 1, "; ".join(hallucinated)
        return 0, ""

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def evaluate_response(self, response_id: int) -> dict | None:
        if self.conn.execute(
            "SELECT eval_id FROM evaluations WHERE response_id = ?", (response_id,)
        ).fetchone():
            return None

        row = self.conn.execute(
            """
            SELECT r.response_id, r.question_id, r.platform, r.response_text,
                   q.target_population, q.target_sector, q.target_region
            FROM ai_responses r
            JOIN questions q ON r.question_id = q.question_id
            WHERE r.response_id = ?
            """,
            (response_id,),
        ).fetchone()

        if not row:
            return None

        response_text = (row["response_text"] or "").strip()
        if not response_text:
            return None

        relevant = self.get_relevant_programs(
            row["target_population"] or "",
            row["target_sector"] or "",
            row["target_region"] or "",
        )
        if not relevant:
            print(f"  No relevant programs for {row['question_id']}, skipping.")
            return None

        vis, covered, gaps, top = self.compute_visibility(response_text, relevant)
        halluc, note = self.detect_hallucinations(response_text, relevant)

        eval_row = {
            "response_id": response_id, "question_id": row["question_id"],
            "platform": row["platform"], "visibility_score": vis,
            "top_match_score": top, "coverage_gap_count": gaps,
            "has_hallucination": halluc, "hallucination_note": note,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.conn.execute(
            """
            INSERT INTO evaluations (
                response_id, question_id, platform,
                visibility_score, top_match_score, coverage_gap_count,
                has_hallucination, hallucination_note, evaluated_at
            ) VALUES (
                :response_id, :question_id, :platform,
                :visibility_score, :top_match_score, :coverage_gap_count,
                :has_hallucination, :hallucination_note, :evaluated_at
            )
            """,
            eval_row,
        )
        self.conn.commit()
        return eval_row

    def run(self) -> None:
        response_ids = [
            r["response_id"]
            for r in self.conn.execute(
                "SELECT response_id FROM ai_responses ORDER BY response_id"
            ).fetchall()
        ]
        if not response_ids:
            print("No AI responses found. Run build_db.py first.")
            return

        print(f"Evaluating {len(response_ids)} responses...")
        scored = skipped = 0
        for rid in response_ids:
            result = self.evaluate_response(rid)
            if result:
                scored += 1
                flag = "YES ⚠" if result["has_hallucination"] else "no"
                print(
                    f"  [{rid:>4}] {result['platform']:<12} Q={result['question_id']}  "
                    f"vis={result['visibility_score']:.2f}  "
                    f"gaps={result['coverage_gap_count']}  halluc={flag}"
                )
                if result["hallucination_note"]:
                    note = result["hallucination_note"]
                    print(f"         └─ {note[:140]}{'…' if len(note)>140 else ''}")
            else:
                skipped += 1

        print(f"\nDone. Scored {scored}, skipped {skipped}.")
        self.conn.close()


if __name__ == "__main__":
    ResponseEvaluator().run()