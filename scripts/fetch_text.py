# fetch_text.py
# Fetches full text from each program's official_url (HTML or PDF) and stores
# results in the `program_text` table in ground_truth.sqlite.
#
# Run from the scripts/ directory:
#   python fetch_text.py
#
# Safe to re-run — already-fetched rows are skipped unless you pass --refetch.
# Broken URLs are recorded with fetch_status='error' so you can audit them.

import argparse
import io
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import DB_PATH, USER_AGENT

# ── constants ────────────────────────────────────────────────────────────────

# (connect_timeout, read_timeout) — separate values matter: SSL hangs at the
# connect phase, not the read phase, so a short connect timeout kills stalled
# handshakes fast while still allowing slow-but-responding servers to send data.
REQUEST_TIMEOUT = (5, 15)
DELAY_BETWEEN_REQUESTS = 0.5  # seconds per thread — reduced since threads run concurrently
MAX_TEXT_LENGTH = 100_000     # characters — truncate giant PDFs

# Serializes all SQLite writes — WAL mode handles concurrent reads fine but
# concurrent writes still raise "database is locked" without a mutex.
_db_lock = threading.Lock()

# Set by the main thread on Ctrl+C so worker threads can exit cleanly
# instead of hitting a closed DB connection.
_shutdown = threading.Event()

DDL = """
CREATE TABLE IF NOT EXISTS program_text (
    entity_id    INTEGER PRIMARY KEY REFERENCES programs(entity_id),
    url          TEXT,
    source_type  TEXT,        -- 'html' or 'pdf'
    full_text    TEXT,
    fetched_at   TEXT,
    fetch_status TEXT         -- 'ok', 'error', 'skipped'
);
"""

HEADERS = {"User-Agent": USER_AGENT}


# ── helpers ──────────────────────────────────────────────────────────────────

def classify_url(url: str) -> str | None:
    """
    Returns 'pdf', 'html', or None (skip).
    Relative paths like /data/... have no real base and can't be fetched.
    """
    if not url or not url.startswith("http"):
        return None
    if url.lower().endswith(".pdf") or "/download/" in url.lower():
        return "pdf"
    return "html"


def fetch_html_text(url: str) -> str:
    """
    GETs a page and returns its visible text, stripped of navigation/boilerplate
    by targeting common content containers, falling back to <body>.
    """
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Remove noise — scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Prefer the main content area if present (common on canada.gc.ca)
    content = (
        soup.find("main")
        or soup.find(id="main-content")
        or soup.find(class_="content")
        or soup.find("article")
        or soup.body
    )

    if content is None:
        return ""

    # get_text with separator gives readable whitespace between tags
    text = content.get_text(separator=" ", strip=True)
    return text[:MAX_TEXT_LENGTH]


def fetch_pdf_text(url: str) -> str:
    """
    Downloads a PDF and extracts text from all pages using pypdf.
    """
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    # PdfReader needs a file-like object
    reader = PdfReader(io.BytesIO(response.content))

    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)

    full_text = "\n".join(pages_text)
    return full_text[:MAX_TEXT_LENGTH]


# ── core logic ────────────────────────────────────────────────────────────────

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def already_fetched_ids(conn: sqlite3.Connection) -> set[int]:
    """Returns entity_ids that already have a row in program_text."""
    rows = conn.execute("SELECT entity_id FROM program_text").fetchall()
    return {row[0] for row in rows}


def load_programs(conn: sqlite3.Connection) -> list[dict]:
    """Loads entity_id + official_url for all programs."""
    rows = conn.execute(
        "SELECT entity_id, official_url FROM programs ORDER BY entity_id"
    ).fetchall()
    return [{"entity_id": row[0], "url": row[1]} for row in rows]


def upsert_result(
    conn: sqlite3.Connection,
    entity_id: int,
    url: str,
    source_type: str | None,
    full_text: str | None,
    status: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        conn.execute(
            """
            INSERT OR REPLACE INTO program_text
                (entity_id, url, source_type, full_text, fetched_at, fetch_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entity_id, url, source_type, full_text, now, status),
        )
        conn.commit()


def fetch_all(
    conn: sqlite3.Connection,
    refetch: bool = False,
    workers: int = 5,
    limit: int | None = None,
) -> None:
    ensure_schema(conn)

    programs = load_programs(conn)
    skip_ids = set() if refetch else already_fetched_ids(conn)

    # Apply limit before filtering so --limit N means "attempt N programs"
    # (rows in entity_id order, so re-runs with a higher limit are reproducible)
    if limit is not None:
        programs = programs[:limit]

    total = len(programs)

    # Shared counters — use a lock since threads write to them concurrently
    counters_lock = threading.Lock()
    counters = {"ok": 0, "errors": 0, "skipped": 0}

    def fetch_one(index_program: tuple[int, dict]) -> None:
        # Exit early if Ctrl+C was pressed — avoids hitting the closed DB
        if _shutdown.is_set():
            return

        i, program = index_program
        entity_id = program["entity_id"]
        url = program["url"] or ""
        prefix = f"[{i}/{total}] entity={entity_id}"

        # Skip already-done
        if entity_id in skip_ids:
            with counters_lock:
                counters["skipped"] += 1
            return

        # Classify the URL
        source_type = classify_url(url)
        if source_type is None:
            print(f"{prefix} SKIP  — not a fetchable URL: {url[:60]}")
            upsert_result(conn, entity_id, url, None, None, "skipped")
            with counters_lock:
                counters["skipped"] += 1
            return

        # Attempt fetch
        try:
            if source_type == "pdf":
                text = fetch_pdf_text(url)
            else:
                text = fetch_html_text(url)

            print(f"{prefix} OK    ({source_type}, {len(text)} chars) {url[:60]}")
            upsert_result(conn, entity_id, url, source_type, text, "ok")
            with counters_lock:
                counters["ok"] += 1

        except Exception as exc:
            print(f"{prefix} ERROR ({source_type}) {url[:60]}")
            print(f"         {type(exc).__name__}: {exc}")
            # Store the error message so we can audit later
            upsert_result(conn, entity_id, url, source_type, str(exc), "error")
            with counters_lock:
                counters["errors"] += 1

        # Each thread sleeps independently — total throughput ≈ workers / delay
        time.sleep(DELAY_BETWEEN_REQUESTS)

    indexed = list(enumerate(programs, start=1))

    print(f"Fetching {total} programs with {workers} workers (delay={DELAY_BETWEEN_REQUESTS}s per thread)")

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(fetch_one, item): item for item in indexed}
        for future in as_completed(futures):
            future.result()  # propagates any unhandled exception
    except KeyboardInterrupt:
        print("\nInterrupted — cancelling pending tasks (already-written rows are safe)...")
        _shutdown.set()  # signals in-flight threads to skip any remaining work
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(1)
    else:
        executor.shutdown(wait=True)

    ok, errors, skipped = counters["ok"], counters["errors"], counters["skipped"]
    print(f"\nDone. ok={ok}  errors={errors}  skipped={skipped}  total={total}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full text from program URLs.")
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch even rows that already have a result (overwrites existing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel fetch threads (default: 5, max recommended: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of programs to fetch in this run (e.g. 500). Useful for testing.",
    )
    args = parser.parse_args()

    # check_same_thread=False allows worker threads to use this connection.
    # _db_lock in upsert_result serializes all writes, making this safe.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        fetch_all(conn, refetch=args.refetch, workers=args.workers, limit=args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
