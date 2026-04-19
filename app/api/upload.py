from fastapi import APIRouter, UploadFile, File, HTTPException

from app.chunking.hierarchical_chunker import HierarchicalChunker
from app.core.config import get_settings
from app.models.schemas import UploadResponse
from app.parsing.docx_converter import convert_docx_to_markdown
from app.services.embedding_service import OpenAIEmbeddingService
from app.services.opensearch_service import OpenSearchService

router = APIRouter()
settings = get_settings()

embedding_service = OpenAIEmbeddingService(
    api_key=settings.openai_api_key,
    model=settings.openai_embedding_model,
    dimensions=settings.openai_embedding_dimensions,
)

opensearch_service = OpenSearchService(
    host=settings.opensearch_host,
    port=settings.opensearch_port,
    username=settings.opensearch_username,
    password=settings.opensearch_password,
    use_ssl=settings.opensearch_use_ssl,
    verify_certs=settings.opensearch_verify_certs,
    index_name=settings.opensearch_index,
)

@router.post(
    "/upload",
    response_model=UploadResponse, 
    summary="Upload DOCX, embed chunks, and index to OpenSearch",
    description=(
        "Parses a DOCX file, chunks it, creates OpenAI embeddings, "
        "and stores vectors + metadata in OpenSearch."
    )
)
async def upload_docx(file: UploadFile = File(...)):
    """
    Upload and preprocess a DOCX document.

    args:
        file (UploadFile): The DOCX file to be uploaded.

    Returns:
        UploadResponse: Status, filename, created chunk count, and indexed chunk count.
    """

    # Validate file type - only .docx files are supported
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    filename = file.filename

    # handle empty docx files
    try:  
        # convert docx to markdown
        markdown = convert_docx_to_markdown(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # handle corrupted docx files
    if not markdown or not markdown.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded document contains no extractable text."
        )
    
    try:
        # Chunk the markdown content using hierarchical structure first.
        hierarchical_chunker = HierarchicalChunker()
        chunks = hierarchical_chunker.chunk_markdown_by_headers(
            markdown=markdown,
            source_file=filename
        )

        # Post-process chunks to ensure retrieval-friendly metadata.
        post_processed_chunks = hierarchical_chunker.finalize_chunks(
            initial_chunks=chunks,
            source_file=filename
        )

        # Generate vector embeddings for each chunk content.
        texts = [chunk["content"] for chunk in post_processed_chunks]
        embeddings = embedding_service.embed_texts(texts)

        # Index content + metadata + embeddings into OpenSearch.
        indexed_count = opensearch_service.index_chunks(post_processed_chunks, embeddings)

        return UploadResponse(
            status="success",
            filename=filename,
            chunks_created=len(post_processed_chunks),
            chunks_indexed=indexed_count,
            index_name=settings.opensearch_index,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc




   