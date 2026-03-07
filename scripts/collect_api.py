# collect_api.py
# Optional: Automates AI response collection via Claude and Gemini APIs.
# Requires API keys set in a .env file in the repo root:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=...
#
# Usage:
#   pip install anthropic google-generativeai python-dotenv
#   python scripts/collect_api.py --platform claude --questions Q001,Q005,Q010
#   python scripts/collect_api.py --platform gemini --all
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import _ROOT

QUESTIONS_CSV = _ROOT / "data" / "questions.csv"
RESPONSES_CSV = _ROOT / "data" / "responses" / "responses.csv"

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-1.5-pro"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass  # dotenv optional; keys can be set in environment directly


def load_questions(ids: list[str] | None = None) -> list[dict]:
    questions = []
    with QUESTIONS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ids is None or row["question_id"] in ids:
                questions.append(row)
    return questions


def load_existing_responses() -> set[tuple[str, str]]:
    """Return set of (question_id, platform) already recorded."""
    existing: set[tuple[str, str]] = set()
    if not RESPONSES_CSV.exists():
        return existing
    with RESPONSES_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add((row["question_id"], row["platform"]))
    return existing


def append_response(row: dict) -> None:
    file_exists = RESPONSES_CSV.exists() and RESPONSES_CSV.stat().st_size > 0
    fieldnames = [
        "question_id", "platform", "model_version", "response_text",
        "programs_mentioned", "collected_at", "collection_method",
    ]
    with RESPONSES_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def collect_claude(questions: list[dict], existing: set) -> None:
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    for q in questions:
        if (q["question_id"], "claude") in existing:
            print(f"  Skipping {q['question_id']} (claude) — already collected.")
            continue

        print(f"  Querying Claude: {q['question_id']} ...")
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": q["question_text"]}],
            )
            response_text = message.content[0].text
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        append_response(
            {
                "question_id": q["question_id"],
                "platform": "claude",
                "model_version": CLAUDE_MODEL,
                "response_text": response_text.replace("\n", " "),
                "programs_mentioned": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "collection_method": "api",
            }
        )
        time.sleep(1)

    print("Claude collection complete.")


def collect_gemini(questions: list[dict], existing: set) -> None:
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    for q in questions:
        if (q["question_id"], "gemini") in existing:
            print(f"  Skipping {q['question_id']} (gemini) — already collected.")
            continue

        print(f"  Querying Gemini: {q['question_id']} ...")
        try:
            response = model.generate_content(q["question_text"])
            response_text = response.text
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        append_response(
            {
                "question_id": q["question_id"],
                "platform": "gemini",
                "model_version": GEMINI_MODEL,
                "response_text": response_text.replace("\n", " "),
                "programs_mentioned": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "collection_method": "api",
            }
        )
        time.sleep(1)

    print("Gemini collection complete.")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Collect AI responses via API.")
    parser.add_argument(
        "--platform",
        choices=["claude", "gemini"],
        required=True,
        help="Which platform to query.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question IDs (e.g. Q001,Q005). Omit to run all.",
    )
    args = parser.parse_args()

    q_ids = [q.strip() for q in args.questions.split(",")] if args.questions else None
    questions = load_questions(q_ids)
    if not questions:
        print("No questions found for the given IDs.")
        sys.exit(1)

    existing = load_existing_responses()
    print(f"Found {len(questions)} questions. {len(existing)} responses already recorded.")

    if args.platform == "claude":
        collect_claude(questions, existing)
    elif args.platform == "gemini":
        collect_gemini(questions, existing)


if __name__ == "__main__":
    main()
