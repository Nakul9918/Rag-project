
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    return {"status": "success", "message": "Ingestion function executed"}

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "RAG API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf],
)
