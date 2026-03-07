# build_db.py
# Populates ground_truth.sqlite with the 5-table schema from processed CSVs,
# the question bank, and any recorded AI responses.
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DB_PATH, PROCESSED_DIR, _ROOT

QUESTIONS_CSV = _ROOT / "data" / "questions.csv"
RESPONSES_CSV = _ROOT / "data" / "responses" / "responses.csv"

DDL = """
CREATE TABLE IF NOT EXISTS programs (
    entity_id         INTEGER PRIMARY KEY,
    raw_record_id     TEXT UNIQUE,
    name_en           TEXT,
    name_fr           TEXT,
    description       TEXT,
    parent_org        TEXT,
    official_url      TEXT,
    jurisdiction      TEXT,
    province_territory TEXT,
    is_active         INTEGER DEFAULT 1,
    tags              TEXT,
    metadata_created  TEXT,
    metadata_modified TEXT,
    entity_type       TEXT DEFAULT 'program'
);

CREATE TABLE IF NOT EXISTS classifications (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER REFERENCES programs(entity_id),
    dimension TEXT,
    value     TEXT,
    is_primary INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS questions (
    question_id       TEXT PRIMARY KEY,
    question_text     TEXT NOT NULL,
    target_population TEXT,
    target_sector     TEXT,
    target_region     TEXT,
    difficulty        TEXT DEFAULT 'medium',
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS ai_responses (
    response_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id       TEXT REFERENCES questions(question_id),
    platform          TEXT NOT NULL,
    model_version     TEXT,
    response_text     TEXT NOT NULL,
    programs_mentioned TEXT,
    collected_at      TEXT,
    collection_method TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id        INTEGER REFERENCES ai_responses(response_id),
    question_id        TEXT,
    platform           TEXT,
    visibility_score   REAL,
    top_match_score    REAL,
    coverage_gap_count INTEGER,
    has_hallucination  INTEGER DEFAULT 0,
    hallucination_note TEXT,
    evaluated_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_clf_entity    ON classifications(entity_id);
CREATE INDEX IF NOT EXISTS idx_clf_dim_val   ON classifications(dimension, value);
CREATE INDEX IF NOT EXISTS idx_resp_qid      ON ai_responses(question_id);
CREATE INDEX IF NOT EXISTS idx_resp_platform ON ai_responses(platform);
"""


class DBBuilder:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        print(f"Connected to {db_path}")

    def create_schema(self) -> None:
        self.conn.executescript(DDL)
        self.conn.commit()
        print("Schema created.")

    def load_entities(self, entities_csv: Path) -> int:
        df = pd.read_csv(entities_csv)
        rows = df.to_dict("records")
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO programs
              (entity_id, raw_record_id, name_en, name_fr, description,
               parent_org, official_url, jurisdiction, province_territory,
               is_active, tags, metadata_created, metadata_modified, entity_type)
            VALUES
              (:entity_id, :raw_record_id, :name_en, :name_fr, :description,
               :parent_org, :official_url, :jurisdiction, :province_territory,
               :is_active, :tags, :metadata_created, :metadata_modified, :entity_type)
            """,
            rows,
        )
        self.conn.commit()
        print(f"Loaded {len(rows)} entities.")
        return len(rows)

    def load_classifications(self, classif_csv: Path) -> int:
        df = pd.read_csv(classif_csv)
        rows = df[["entity_id", "dimension", "value", "is_primary"]].to_dict("records")
        self.conn.executemany(
            """
            INSERT INTO classifications (entity_id, dimension, value, is_primary)
            VALUES (:entity_id, :dimension, :value, :is_primary)
            """,
            rows,
        )
        self.conn.commit()
        print(f"Loaded {len(rows)} classifications.")
        return len(rows)

    def load_questions(self, questions_csv: Path) -> int:
        if not questions_csv.exists():
            print(f"  questions.csv not found at {questions_csv}, skipping.")
            return 0
        df = pd.read_csv(questions_csv)
        rows = df.to_dict("records")
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO questions
              (question_id, question_text, target_population, target_sector,
               target_region, difficulty, notes)
            VALUES
              (:question_id, :question_text, :target_population, :target_sector,
               :target_region, :difficulty, :notes)
            """,
            rows,
        )
        self.conn.commit()
        print(f"Loaded {len(rows)} questions.")
        return len(rows)

    def load_responses(self, responses_csv: Path) -> int:
        if not responses_csv.exists():
            print(f"  responses.csv not found at {responses_csv}, skipping.")
            return 0
        df = pd.read_csv(responses_csv)
        if df.empty:
            print("  responses.csv is empty, skipping.")
            return 0
        rows = df.to_dict("records")
        inserted = 0
        for row in rows:
            try:
                self.conn.execute(
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
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        print(f"Loaded {inserted} AI responses.")
        return inserted

    def run(self) -> None:
        self.create_schema()
        self.load_entities(PROCESSED_DIR / "entities.csv")
        self.load_classifications(PROCESSED_DIR / "classifications.csv")
        self.load_questions(QUESTIONS_CSV)
        self.load_responses(RESPONSES_CSV)
        self.conn.close()
        print("\nDatabase build complete.")


if __name__ == "__main__":
    DBBuilder().run()
