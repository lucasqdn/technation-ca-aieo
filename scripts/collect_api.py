# collect_api.py
# Queries all four LLM platforms concurrently and writes responses directly
# to the ai_responses table in ground_truth.sqlite.
#
# USAGE
#   python collect_api.py                   # all platforms, all questions
#   python collect_api.py --platform claude # one platform only
#   python collect_api.py --questions Q001,Q005,Q010
#   python collect_api.py --workers 2       # questions-per-platform concurrency
#
# FIXES VS PREVIOUS VERSION
#   - Writes to SQLite ai_responses (not responses.csv)
#   - All 4 platforms run concurrently via ThreadPoolExecutor
#   - Fixed model names: gemini-2.0-flash, gpt-4o, claude-sonnet-4-5
#   - Fixed OpenAI API: chat.completions.create (not client.responses.create)
#   - Existing-response check reads from SQLite (not CSV)
#   - Per-platform errors don't abort other platforms

import argparse
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH, _ROOT

QUESTIONS_CSV = _ROOT / "data" / "questions.csv"

# ── Model names ───────────────────────────────────────────────────────────────
CLAUDE_MODEL     = "claude-sonnet-4-6"
GEMINI_MODEL     = "gemini-3-flash-preview"
OPENAI_MODEL     = "gpt-5.4"
PERPLEXITY_MODEL = "sonar"

SYSTEM_PROMPT = (
    "You are a helpful assistant with expertise in Canadian government programs "
    "and social services. Answer questions factually and specifically. "
    "Name real programs, organizations, and services where possible. "
    "If you are uncertain about a specific detail, say so clearly."
)

RATE_DELAY = 1.0  # seconds between requests per platform

# Serializes all SQLite writes across threads
_db_lock = threading.Lock()


# ── Environment ───────────────────────────────────────────────────────────────

def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass


# ── Database helpers ──────────────────────────────────────────────────────────

def load_questions(
    db_path: Path, ids: list[str] | None = None
) -> list[dict]:
    """Load questions from SQLite. Falls back to CSV if table is empty."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM questions ORDER BY question_id").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    questions = [dict(r) for r in rows]

    # Fallback: read from CSV if DB has no questions yet
    if not questions and QUESTIONS_CSV.exists():
        import csv
        with QUESTIONS_CSV.open(encoding="utf-8") as f:
            questions = list(csv.DictReader(f))

    if ids:
        questions = [q for q in questions if q["question_id"] in ids]

    return questions


def load_existing(db_path: Path) -> set[tuple[str, str]]:
    """Return (question_id, platform) pairs already in ai_responses."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT question_id, platform FROM ai_responses"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {(r[0], r[1]) for r in rows}


def insert_response(db_path: Path, row: dict) -> None:
    """Thread-safe insert into ai_responses."""
    with _db_lock:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            INSERT OR IGNORE INTO ai_responses
              (question_id, platform, model_version, response_text,
               programs_mentioned, collected_at, collection_method)
            VALUES
              (:question_id, :platform, :model_version, :response_text,
               :programs_mentioned, :collected_at, :collection_method)
            """,
            row,
        )
        conn.commit()
        conn.close()


# ── Per-platform query functions ──────────────────────────────────────────────

def query_claude(question_text: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question_text}],
    )
    return message.content[0].text


def query_gemini(question_text: str) -> str:
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{question_text}",
    )
    return response.text


def query_chatgpt(question_text: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    # Use chat.completions — NOT client.responses (that's the Assistants API)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_text},
        ],
    )
    return completion.choices[0].message.content or ""


def query_perplexity(question_text: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    completion = client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_text},
        ],
    )
    return completion.choices[0].message.content or ""


PLATFORM_FUNCS = {
    "claude":     (query_claude,     CLAUDE_MODEL),
    "gemini":     (query_gemini,     GEMINI_MODEL),
    "chatgpt":    (query_chatgpt,    OPENAI_MODEL),
    "perplexity": (query_perplexity, PERPLEXITY_MODEL),
}


# ── Collection logic ──────────────────────────────────────────────────────────

def collect_platform(
    platform: str,
    questions: list[dict],
    existing: set[tuple[str, str]],
    db_path: Path,
    workers: int = 1,
) -> tuple[int, int, int]:
    """
    Collect responses for one platform. Returns (ok, skipped, errors).
    Uses a ThreadPoolExecutor so multiple questions are in-flight at once.
    """
    query_fn, model_version = PLATFORM_FUNCS[platform]

    # Validate API key and import before launching threads
    try:
        query_fn("ping test")
    except RuntimeError as e:
        print(f"  [{platform}] SETUP ERROR: {e} — skipping platform.")
        return 0, 0, len(questions)
    except Exception:
        pass  # actual API errors are fine here — key is present

    to_collect = [
        q for q in questions
        if (q["question_id"], platform) not in existing
    ]
    skipped = len(questions) - len(to_collect)

    if skipped:
        print(f"  [{platform}] Skipping {skipped} already-collected questions.")
    if not to_collect:
        print(f"  [{platform}] Nothing to collect.")
        return 0, skipped, 0

    ok_count = 0
    err_count = 0
    counters_lock = threading.Lock()

    def fetch_one(q: dict) -> None:
        nonlocal ok_count, err_count
        qid = q["question_id"]
        print(f"  [{platform}] {qid} querying...", flush=True)
        try:
            text = query_fn(q["question_text"])
            insert_response(db_path, {
                "question_id":      qid,
                "platform":         platform,
                "model_version":    model_version,
                "response_text":    text,
                "programs_mentioned": "",
                "collected_at":     datetime.now(timezone.utc).isoformat(),
                "collection_method": "api",
            })
            print(f"  [{platform}] {qid} ✓ ({len(text)} chars)")
            with counters_lock:
                ok_count += 1
        except Exception as e:
            print(f"  [{platform}] {qid} ERROR: {type(e).__name__}: {e}")
            with counters_lock:
                err_count += 1

        time.sleep(RATE_DELAY)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, q) for q in to_collect]
        for f in as_completed(futures):
            f.result()  # surfaces unhandled exceptions

    return ok_count, skipped, err_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(
        description="Collect LLM responses for all questions and write to SQLite."
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Delete existing responses for the selected platform(s) and re-collect from scratch.",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORM_FUNCS),
        default=None,
        help="Single platform to query. Omit to run all four concurrently.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question IDs, e.g. Q001,Q005. Omit for all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel questions per platform (default 1 — increase carefully to avoid rate limits).",
    )
    args = parser.parse_args()

    q_ids = [q.strip() for q in args.questions.split(",")] if args.questions else None
    questions = load_questions(DB_PATH, q_ids)
    if not questions:
        print("No questions found. Run build_db.py first (or check questions.csv).")
        sys.exit(1)

    platforms = [args.platform] if args.platform else list(PLATFORM_FUNCS)

    # --refetch: delete existing responses for the target platforms before collecting
    if args.refetch:
        conn = sqlite3.connect(DB_PATH)
        for p in platforms:
            deleted = conn.execute(
                "DELETE FROM ai_responses WHERE platform = ?", (p,)
            ).rowcount
            conn.execute(
                "DELETE FROM evaluations WHERE platform = ?", (p,)
            )
            print(f"  --refetch: deleted {deleted} responses (+ evaluations) for {p}")
        conn.commit()
        conn.close()

    existing = load_existing(DB_PATH)

    print(f"Questions   : {len(questions)}")
    print(f"Platforms   : {', '.join(platforms)}")
    print(f"Workers     : {args.workers} per platform")
    print(f"Already done: {len(existing)} (question, platform) pairs")
    print()

    if args.platform:
        # Single platform — run directly
        ok, skipped, errors = collect_platform(
            args.platform, questions, existing, DB_PATH, args.workers
        )
        print(f"\n{args.platform}: ok={ok} skipped={skipped} errors={errors}")
    else:
        # All platforms concurrently — one thread per platform
        # Each platform thread runs its own questions sequentially (or with
        # --workers > 1, in parallel). This gives true 4-way concurrency
        # without overloading any single API.
        results: dict[str, tuple[int, int, int]] = {}
        results_lock = threading.Lock()

        def run_platform(p: str) -> None:
            ok, skipped, errors = collect_platform(
                p, questions, existing, DB_PATH, args.workers
            )
            with results_lock:
                results[p] = (ok, skipped, errors)

        with ThreadPoolExecutor(max_workers=len(platforms)) as executor:
            futures = {executor.submit(run_platform, p): p for p in platforms}
            for future in as_completed(futures):
                future.result()

        print("\n" + "=" * 50)
        print("COLLECTION SUMMARY")
        print(f"  {'platform':<12} {'ok':>5} {'skipped':>8} {'errors':>7}")
        total_ok = total_skip = total_err = 0
        for p in platforms:
            ok, skip, err = results.get(p, (0, 0, 0))
            total_ok += ok; total_skip += skip; total_err += err
            print(f"  {p:<12} {ok:>5} {skip:>8} {err:>7}")
        print(f"  {'TOTAL':<12} {total_ok:>5} {total_skip:>8} {total_err:>7}")

        if total_err:
            print(f"\n⚠  {total_err} errors — check API keys and model availability.")
        if total_ok:
            print(f"\nNext: run evaluate.py  (DELETE FROM evaluations; first if re-collecting)")


if __name__ == "__main__":
    main()