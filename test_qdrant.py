
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError(
        "QDRANT_URL or QDRANT_API_KEY is missing from .env"
    )

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=15,
)

try:
    collections = client.get_collections()
    print("Qdrant connection successful!")
    print("Available collections:", [
        collection.name for collection in collections.collections
    ])
except Exception as exc:
    print("Qdrant connection failed:", type(exc).__name__)
    raise
