# fetch.py
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from config import (
    CKAN_PACKAGE_SEARCH_URL,
    RAW_DIR,
    ROWS_PER_QUERY,
    SEARCH_QUERIES,
    USER_AGENT,
)


class CKANFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_query(self, query: str, rows: int = ROWS_PER_QUERY) -> list[dict]:
        results: list[dict] = []
        start = 0

        while True:
            params = {
                "q": query,
                "rows": rows,
                "start": start,
            }

            try:
                resp = self.session.get(CKAN_PACKAGE_SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                print(f"  Request failed for '{query}': {e}")
                break
            except ValueError as e:
                print(f"  Invalid JSON for '{query}': {e}")
                break

            if not isinstance(data, dict) or not data.get("success"):
                print(f"  CKAN returned an unsuccessful response for '{query}'")
                break

            result_obj = data.get("result", {})
            batch = result_obj.get("results", [])
            total = result_obj.get("count", 0)

            if not batch:
                break

            results.extend(batch)
            start += len(batch)

            print(f"  Fetched {start}/{total} for '{query}'")

            if start >= total:
                break

            time.sleep(0.5)

        return results

    def save_raw(self, query: str, records: list[dict]) -> None:
        slug = query.lower().replace(" ", "_")[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RAW_DIR / f"{slug}_{timestamp}.json"

        payload = {
            "query": query,
            "fetched_at": datetime.now().isoformat(),
            "count": len(records),
            "records": records,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(records)} records -> {path}")

    def run(self) -> None:
        for query in SEARCH_QUERIES:
            print(f"\nFetching: '{query}'")
            records = self.fetch_query(query)
            if records:
                self.save_raw(query, records)
            else:
                print("  No records returned.")
        print("\nFetch complete.")


if __name__ == "__main__":
    CKANFetcher().run()