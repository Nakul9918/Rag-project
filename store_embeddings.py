
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv()

PDF_DIR = "data/pdfs"
COLLECTION_NAME = "rag_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

print("Loading local embedding model...")
model = SentenceTransformer(MODEL_NAME)

documents = SimpleDirectoryReader(
    input_dir=PDF_DIR,
    required_exts=[".pdf"],
).load_data()

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
nodes = splitter.get_nodes_from_documents(documents)

print(f"Loaded {len(documents)} documents")
print(f"Generated {len(nodes)} chunks")

for start in range(0, len(nodes), 32):
    batch = nodes[start:start + 32]
    texts = [node.get_content() for node in batch]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    points = [
        models.PointStruct(
            id=node.node_id,
            vector=embedding.tolist(),
            payload={
                "text": node.get_content(),
                "source": node.metadata.get("file_name", "unknown"),
                "page": node.metadata.get("page_label", ""),
            },
        )
        for node, embedding in zip(batch, embeddings)
    ]

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Stored {min(start + len(batch), len(nodes))}/{len(nodes)}")

print("Local embedding and Qdrant storage completed!")
print("Total points:", qdrant.count(COLLECTION_NAME).count)
