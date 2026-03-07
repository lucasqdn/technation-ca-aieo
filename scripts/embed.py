# embed.py
# Embeds all 5,618 entities into a persistent ChromaDB vector store using
# sentence-transformers/all-MiniLM-L6-v2 (free, local, no API key needed).
# Run once after parse.py; rerunning uses upsert so it's idempotent.
import sys
from pathlib import Path

import pandas as pd

from config import CHROMA_DIR, PROCESSED_DIR

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "canadian_programs"
BATCH_SIZE = 500
SIMILARITY_THRESHOLD = 0.55  # cosine similarity cutoff for "covered"


class ProgramEmbedder:
    def __init__(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            print("ERROR: Run 'pip install chromadb sentence-transformers' first.")
            sys.exit(1)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def build_document_text(self, row: "pd.Series") -> str:
        name = str(row.get("name_en", "") or "")
        desc = str(row.get("description", "") or "")[:500]
        org = str(row.get("parent_org", "") or "")
        tags = str(row.get("tags", "") or "")
        region = str(row.get("province_territory", "") or "")
        return f"{name}. {desc}. Organization: {org}. Tags: {tags}. Region: {region}."

    def _build_metadata_map(
        self, entities_df: pd.DataFrame, classif_df: pd.DataFrame
    ) -> dict:
        """Pivot classifications into pipe-joined strings per entity_id."""
        meta: dict[int, dict] = {}

        for entity_id, group in classif_df.groupby("entity_id"):
            pops = "|".join(
                group[group["dimension"] == "population"]["value"].tolist()
            )
            secs = "|".join(
                group[group["dimension"] == "sector"]["value"].tolist()
            )
            meta[int(entity_id)] = {"populations": pops, "sectors": secs}

        # Fill any entity with no classifications
        for _, row in entities_df.iterrows():
            eid = int(row["entity_id"])
            if eid not in meta:
                meta[eid] = {"populations": "general_public", "sectors": "public_services"}

        return meta

    def embed_all(
        self,
        entities_csv: Path = PROCESSED_DIR / "entities.csv",
        classif_csv: Path = PROCESSED_DIR / "classifications.csv",
    ) -> None:
        print("Loading CSVs...")
        entities_df = pd.read_csv(entities_csv)
        classif_df = pd.read_csv(classif_csv)

        print("Building metadata map...")
        meta_map = self._build_metadata_map(entities_df, classif_df)

        total = len(entities_df)
        print(f"Embedding {total} entities in batches of {BATCH_SIZE}...")

        for batch_start in range(0, total, BATCH_SIZE):
            batch = entities_df.iloc[batch_start : batch_start + BATCH_SIZE]

            ids = [str(int(row["entity_id"])) for _, row in batch.iterrows()]
            documents = [self.build_document_text(row) for _, row in batch.iterrows()]
            metadatas = []
            for _, row in batch.iterrows():
                eid = int(row["entity_id"])
                m = meta_map.get(eid, {})
                metadatas.append(
                    {
                        "entity_id": eid,
                        "province_territory": str(row.get("province_territory", "") or ""),
                        "populations": m.get("populations", "general_public"),
                        "sectors": m.get("sectors", "public_services"),
                        "is_active": int(row.get("is_active", 1)),
                    }
                )

            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            end = min(batch_start + BATCH_SIZE, total)
            print(f"  Upserted {end}/{total}")

        print(f"\nDone. ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
        print(f"Total documents in collection: {self.collection.count()}")

    def query(
        self,
        text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Embed `text` and return the top-n most similar programs.
        Returns list of dicts: {entity_id, document, distance, metadata}.
        Lower distance = more similar (cosine distance = 1 - cosine_similarity).
        """
        kwargs: dict = {"query_texts": [text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "entity_id": int(doc_id),
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": round(1 - results["distances"][0][i], 4),
                    "metadata": results["metadatas"][0][i],
                }
            )
        return output


if __name__ == "__main__":
    embedder = ProgramEmbedder()
    embedder.embed_all()

    # Quick sanity check
    print("\n--- Sanity check: 'Indigenous youth employment training BC' ---")
    hits = embedder.query("Indigenous youth employment training BC", n_results=5)
    for h in hits:
        print(f"  [{h['similarity']:.3f}] {h['document'][:120]}")
