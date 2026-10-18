
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "rag_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Connect to Qdrant Cloud
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# Load the same model used when storing document embeddings
model = SentenceTransformer(MODEL_NAME)

question = input("Ask a question about the PDF: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# Convert the question into a 384-dimensional vector
query_vector = model.encode(
    question,
    normalize_embeddings=True,
).tolist()

# Find the most relevant document chunks
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True,
).points

print("\nTop matching document chunks:\n")

for rank, point in enumerate(results, start=1):
    payload = point.payload or {}

    print(f"Result {rank} | Score: {point.score:.4f}")
    print(f"Source: {payload.get('source', 'unknown')}")
    print(f"Page: {payload.get('page', 'unknown')}")
    print(payload.get("text", "No text found"))
    print("-" * 70)
