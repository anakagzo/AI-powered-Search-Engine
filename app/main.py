from fastapi import FastAPI
from app.api.search import router as search_router
from app.api.upload import router as upload_router

app = FastAPI(
    title="RAG Document Ingestion API",
    description="DOCX ingestion and chunking service for RAG pipelines",
    version="1.0.0"
)

app.include_router(upload_router)
app.include_router(search_router)












 

    

