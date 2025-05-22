import json
from pathlib import Path
import chromadb
from chromadb.api.models import Collection
from backend.RagCore.Utils.pathProvider import get_vector_base_path


class ChromaManager:
    def __init__(self, collection_name: str = "collection_name"):
        self.db_path = get_vector_base_path()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection: Collection = self.client.get_or_create_collection(
            collection_name
        )

    def export_to_json(self, output_file: Path = Path("chroma_export.json")) -> None:
        """Export the entire Chroma collection to a JSON file."""
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings", "ids"]
        )

        export_data = [
            {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
                "embedding": results["embeddings"][i],
            }
            for i in range(len(results["ids"]))
        ]

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Export completed: {output_file.resolve()}")

    def import_from_json(self, input_file: Path = Path("chroma_export.json")) -> None:
        """Import documents from a JSON file into the current Chroma collection."""
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file.resolve()}")

        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ids = [item["id"] for item in data]
        documents = [item["document"] for item in data]
        metadatas = [item["metadata"] for item in data]
        embeddings = [item["embedding"] for item in data]

        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

        print(
            f"✅ Import completed: {len(ids)} items added to collection '{self.collection.name}'"
        )
