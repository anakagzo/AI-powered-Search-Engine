from fastapi import APIRouter, UploadFile, File, HTTPException
from app.parsing.docx_converter import convert_docx_to_markdown
from app.chunking.hierarchical_chunker import HierarchicalChunker
from app.models.schemas import UploadResponse

router = APIRouter()

@router.post(
    "/upload",
    response_model=UploadResponse, 
    summary="Upload and preprocess a DOCX document",
    description="Parses a DOCX file, chunks it, and returns RAG-ready JSON."
)
async def upload_docx(file: UploadFile = File(...)):
    """
    Upload and preprocess a DOCX document.

    args:
        file (UploadFile): The DOCX file to be uploaded.

    returns:
        UploadResponse: A response model containing the status, filename,
                        number of chunks created, and the chunks themselves.            
    """

    # Validate file type - only .docx files are supported
    if not file.filename.endswith(".docx"):
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
    
    # Chunk the markdown content using Hierarchical Chunking approach
    hierarchical_chunker = HierarchicalChunker()
    chunks = hierarchical_chunker.chunk_markdown_by_headers(
        markdown=markdown,
        source_file=filename
    )
    # Post-process chunks to ensure RAG readiness
    post_processed_chunks = hierarchical_chunker.finalize_chunks(
        initial_chunks=chunks,
        source_file=filename
    ) 
   
    return UploadResponse(
        status="success",
        filename=filename,
        chunks_created=len(post_processed_chunks),
        chunks=post_processed_chunks
    )




   