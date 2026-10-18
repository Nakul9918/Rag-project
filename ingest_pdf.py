
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

PDF_DIR = Path("data/pdfs")

if not PDF_DIR.exists():
    raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

documents = SimpleDirectoryReader(
    input_dir=str(PDF_DIR),
    required_exts=[".pdf"],
).load_data()

if not documents:
    raise ValueError("No PDF documents were found.")

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

nodes = splitter.get_nodes_from_documents(documents)

print(f"Loaded document pages: {len(documents)}")
print(f"Generated text chunks: {len(nodes)}")

if nodes:
    print("\nFirst chunk preview:\n")
    print(nodes[0].get_content()[:500])
