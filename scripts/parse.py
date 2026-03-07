# parse.py
import json
import re
from pathlib import Path

import pandas as pd

from config import (
    POPULATION_KEYWORDS,
    PROCESSED_DIR,
    RAW_DIR,
    SECTOR_KEYWORDS,
)


class CKANParser:
    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.seen_ids: set[str] = set()

    def load_all_raw(self) -> list[dict]:
        all_records: list[dict] = []

        for path in RAW_DIR.glob("*.json"):
            with path.open(encoding="utf-8") as f:
                data = json.load(f)

            for rec in data.get("records", []):
                rec_id = rec.get("id")
                if rec_id and rec_id not in self.seen_ids:
                    self.seen_ids.add(rec_id)
                    all_records.append(rec)

        print(f"Loaded {len(all_records)} unique records from raw files.")
        return all_records

    def clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def safe_join_tags(self, rec: dict) -> str:
        tags = rec.get("tags", [])
        return "|".join(
            t.get("name", "").strip()
            for t in tags
            if isinstance(t, dict) and t.get("name")
        )

    def extract_name_fr(self, rec: dict) -> str:
        # Safer than assuming a specific translated title field exists.
        # Keeps schema stable while avoiding brittle field assumptions.
        title_translated = rec.get("title_translated")
        if isinstance(title_translated, dict):
            return self.clean_text(title_translated.get("fr", ""))

        translated = rec.get("translated_title")
        if isinstance(translated, dict):
            return self.clean_text(translated.get("fr", ""))

        return ""

    def choose_official_url(self, rec: dict) -> str:
        package_url = rec.get("url")
        if package_url:
            return package_url

        resources = rec.get("resources", [])
        for resource in resources:
            if isinstance(resource, dict) and resource.get("url"):
                return resource["url"]

        return ""

    def infer_jurisdiction(self, rec: dict) -> str:
        org_title = ((rec.get("organization") or {}).get("title") or "").lower()
        text = " ".join([
            rec.get("title", "") or "",
            rec.get("notes", "") or "",
            org_title,
        ]).lower()

        if "canada" in text or "government of canada" in text:
            return "federal"
        return "unknown"

    def infer_province(self, rec: dict) -> str:
        text = " ".join([
            rec.get("title", "") or "",
            rec.get("notes", "") or "",
            self.safe_join_tags(rec),
            ((rec.get("organization") or {}).get("title") or ""),
        ]).lower()

        province_map = {
            "BC": ["british columbia", " vancouver", " victoria"],
            "ON": ["ontario", " toronto", " ottawa"],
            "AB": ["alberta", " calgary", " edmonton"],
            "QC": ["quebec", "québec", " montréal", " montreal"],
            "MB": ["manitoba", " winnipeg"],
            "SK": ["saskatchewan", " regina", " saskatoon"],
            "NB": ["new brunswick"],
            "NS": ["nova scotia", " halifax"],
            "NL": ["newfoundland", " labrador"],
            "PEI": ["prince edward island", " charlottetown"],
            "YT": ["yukon", " whitehorse"],
            "NT": ["northwest territories", " yellowknife"],
            "NU": ["nunavut", " iqaluit"],
        }

        for province, keywords in province_map.items():
            if any(k in text for k in keywords):
                return province

        return "national"

    def extract_fields(self, rec: dict) -> dict:
        return {
            "raw_record_id": rec.get("id", ""),
            "name_en": self.clean_text(rec.get("title", "")),
            "name_fr": self.extract_name_fr(rec),
            "description": self.clean_text(rec.get("notes", "")),
            "parent_org": (rec.get("organization") or {}).get("title", ""),
            "official_url": self.choose_official_url(rec),
            "jurisdiction": self.infer_jurisdiction(rec),
            "province_territory": self.infer_province(rec),
            "is_active": 1 if rec.get("state", "active") == "active" else 0,
            "tags": self.safe_join_tags(rec),
            "metadata_created": rec.get("metadata_created", ""),
            "metadata_modified": rec.get("metadata_modified", ""),
        }

    def classify_populations(self, row: dict) -> list[str]:
        text = f"{row['name_en']} {row['description']} {row['tags']}".lower()
        matched = [
            pop for pop, keywords in POPULATION_KEYWORDS.items()
            if any(k in text for k in keywords)
        ]
        return matched if matched else ["general_public"]

    def classify_sectors(self, row: dict) -> list[str]:
        text = f"{row['name_en']} {row['description']} {row['tags']}".lower()
        matched = [
            sector for sector, keywords in SECTOR_KEYWORDS.items()
            if any(k in text for k in keywords)
        ]
        return matched if matched else ["public_services"]

    def run(self) -> None:
        records = self.load_all_raw()
        rows: list[dict] = []
        classifications: list[dict] = []

        for rec in records:
            entity = self.extract_fields(rec)
            entity_id = len(rows) + 1
            entity["entity_id"] = entity_id
            entity["entity_type"] = "program"

            populations = self.classify_populations(entity)
            sectors = self.classify_sectors(entity)
            region = entity["province_territory"]

            for i, pop in enumerate(populations):
                classifications.append({
                    "entity_id": entity_id,
                    "dimension": "population",
                    "value": pop,
                    "is_primary": 1 if i == 0 else 0,
                })

            for i, sec in enumerate(sectors):
                classifications.append({
                    "entity_id": entity_id,
                    "dimension": "sector",
                    "value": sec,
                    "is_primary": 1 if i == 0 else 0,
                })

            classifications.append({
                "entity_id": entity_id,
                "dimension": "region",
                "value": region,
                "is_primary": 1,
            })

            rows.append(entity)

        entities_df = pd.DataFrame(rows)
        classif_df = pd.DataFrame(classifications)

        entities_df.to_csv(PROCESSED_DIR / "entities.csv", index=False)
        classif_df.to_csv(PROCESSED_DIR / "classifications.csv", index=False)

        print(f"Parsed {len(rows)} entities and {len(classifications)} classification tags.")
        print(f"Saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    CKANParser().run()