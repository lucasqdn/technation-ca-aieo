# evaluate.py
# Scores AI responses against the ground truth ChromaDB collection.
# For each recorded response:
#   1. Embeds the response text.
#   2. Fetches all ground-truth programs relevant to the question
#      (matched by target_population + target_sector + target_region).
#   3. Computes visibility_score = fraction of relevant programs with
#      cosine similarity >= SIMILARITY_THRESHOLD to the response.
#   4. Counts coverage gaps and flags potential hallucinations.
#   5. Writes results to the evaluations table in SQLite.
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import CHROMA_DIR, DB_PATH

SIMILARITY_THRESHOLD = 0.55   # cosine similarity >= this → "covered"
HALLUCINATION_THRESHOLD = 0.5  # program name in response but sim < this → flag
TOP_K_RESPONSE = 30            # how many ChromaDB results to fetch per response


def get_embedder():
    try:
        from embed import ProgramEmbedder
        return ProgramEmbedder()
    except ImportError as e:
        print(f"ERROR loading embedder: {e}")
        print("Make sure chromadb and sentence-transformers are installed.")
        sys.exit(1)


class ResponseEvaluator:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.embedder = get_embedder()
        print("ResponseEvaluator ready.")

    def get_relevant_program_ids(
        self, target_population: str, target_sector: str, target_region: str
    ) -> list[int]:
        """
        Return active entity_ids whose classifications match the question's
        target population, sector, and region.

        Matching logic:
        - population must match if target_population is specific
        - sector must match if target_sector is specific
        - region must match target_region OR 'national' if target_region is specific
        - broad defaults are ignored:
            * general_public
            * public_services
            * national

        This uses AND logic across dimensions, which matches the intended
        target_population + target_sector + target_region behavior.
        """
        base_query = """
            SELECT DISTINCT p.entity_id
            FROM programs p
            WHERE p.is_active = 1
        """

        clauses: list[str] = []
        params: list[str] = []

        if target_population and target_population != "general_public":
            clauses.append("""
                EXISTS (
                    SELECT 1
                    FROM classifications c_pop
                    WHERE c_pop.entity_id = p.entity_id
                    AND c_pop.dimension = 'population'
                    AND c_pop.value = ?
                )
            """)
            params.append(target_population)

        if target_sector and target_sector != "public_services":
            clauses.append("""
                EXISTS (
                    SELECT 1
                    FROM classifications c_sec
                    WHERE c_sec.entity_id = p.entity_id
                    AND c_sec.dimension = 'sector'
                    AND c_sec.value = ?
                )
            """)
            params.append(target_sector)

        if target_region and target_region != "national":
            clauses.append("""
                EXISTS (
                    SELECT 1
                    FROM classifications c_reg
                    WHERE c_reg.entity_id = p.entity_id
                    AND c_reg.dimension = 'region'
                    AND c_reg.value IN (?, 'national')
                )
            """)
            params.append(target_region)

        if not clauses:
            rows = self.conn.execute(
                "SELECT DISTINCT entity_id FROM programs WHERE is_active = 1 LIMIT 200"
            ).fetchall()
            return [r["entity_id"] for r in rows]

        query = base_query + "\n AND " + "\n AND ".join(clauses)
        rows = self.conn.execute(query, params).fetchall()
        return [r["entity_id"] for r in rows]

    def evaluate_response(self, response_id: int) -> dict | None:
        """
        Evaluate one AI response. Returns the evaluation dict or None if skipped.
        """
        # Check if already evaluated
        exists = self.conn.execute(
            "SELECT eval_id FROM evaluations WHERE response_id = ?", (response_id,)
        ).fetchone()
        if exists:
            return None

        # Load response + question metadata
        row = self.conn.execute(
            """
            SELECT r.response_id, r.question_id, r.platform, r.response_text,
                   r.programs_mentioned,
                   q.target_population, q.target_sector, q.target_region
            FROM ai_responses r
            JOIN questions q ON r.question_id = q.question_id
            WHERE r.response_id = ?
            """,
            (response_id,),
        ).fetchone()

        if not row:
            print(f"  response_id {response_id} not found or missing question join.")
            return None

        response_text = row["response_text"] or ""
        if not response_text.strip():
            print(f"  response_id {response_id} has empty text, skipping.")
            return None

        # Get relevant ground-truth programs for this question
        relevant_ids = self.get_relevant_program_ids(
            row["target_population"] or "",
            row["target_sector"] or "",
            row["target_region"] or "",
        )
        total_relevant = len(relevant_ids)

        if total_relevant == 0:
            print(f"  No relevant programs found for {row['question_id']}, skipping.")
            return None

        # Embed the response and get top-k similar programs from ChromaDB
        top_hits = self.embedder.query(response_text, n_results=min(TOP_K_RESPONSE, total_relevant + 5))
        hit_ids_above_threshold = {
            h["entity_id"] for h in top_hits if h["similarity"] >= SIMILARITY_THRESHOLD
        }
        top_match_score = top_hits[0]["similarity"] if top_hits else 0.0

        # Visibility: fraction of relevant programs covered by the response
        covered = len(hit_ids_above_threshold & set(relevant_ids))
        visibility_score = round(covered / total_relevant, 4) if total_relevant > 0 else 0.0
        coverage_gap_count = total_relevant - covered

        # Hallucination check: any program_name mentioned manually but very low similarity?
        has_hallucination = 0
        hallucination_note = ""
        mentioned_raw = row["programs_mentioned"] or ""
        mentioned_names = [n.strip() for n in mentioned_raw.split("|") if n.strip()]
        hallucinated = []
        for name in mentioned_names:
            hits = self.embedder.query(name, n_results=1)
            if hits and hits[0]["similarity"] < HALLUCINATION_THRESHOLD:
                hallucinated.append(f"'{name}' (best match: {hits[0]['similarity']:.2f})")

        if hallucinated:
            has_hallucination = 1
            hallucination_note = "; ".join(hallucinated)

        eval_row = {
            "response_id": response_id,
            "question_id": row["question_id"],
            "platform": row["platform"],
            "visibility_score": visibility_score,
            "top_match_score": round(top_match_score, 4),
            "coverage_gap_count": coverage_gap_count,
            "has_hallucination": has_hallucination,
            "hallucination_note": hallucination_note,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.conn.execute(
            """
            INSERT INTO evaluations
              (response_id, question_id, platform, visibility_score,
               top_match_score, coverage_gap_count, has_hallucination,
               hallucination_note, evaluated_at)
            VALUES
              (:response_id, :question_id, :platform, :visibility_score,
               :top_match_score, :coverage_gap_count, :has_hallucination,
               :hallucination_note, :evaluated_at)
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
            print("No AI responses found in database. Run build_db.py after adding responses.")
            return

        print(f"Evaluating {len(response_ids)} responses...")
        scored = 0
        for rid in response_ids:
            result = self.evaluate_response(rid)
            if result:
                scored += 1
                print(
                    f"  [{rid}] {result['platform']:12s} Q={result['question_id']} "
                    f"visibility={result['visibility_score']:.2f} "
                    f"gaps={result['coverage_gap_count']} "
                    f"hallucination={'YES' if result['has_hallucination'] else 'no'}"
                )

        print(f"\nEvaluation complete. Scored {scored} responses.")
        self.conn.close()


if __name__ == "__main__":
    ResponseEvaluator().run()
