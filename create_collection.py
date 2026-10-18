
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

name = "rag_documents"

if client.collection_exists(name):
    info = client.get_collection(name)
    if info.points_count:
        raise RuntimeError(
            "Collection contains data. Do not delete it automatically."
        )
    client.delete_collection(name)

client.create_collection(
    collection_name=name,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)

print("Created rag_documents with 384 dimensions.")
