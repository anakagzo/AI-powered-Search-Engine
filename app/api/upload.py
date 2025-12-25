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
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    filename = file.filename

    try:
    # handle empty docx files        
        markdown = convert_docx_to_markdown(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # handle corrupted docx files
    if not markdown or not markdown.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded document contains no extractable text."
        )
    #print(markdown)

    hierarchical_chunker = HierarchicalChunker()
    chunks = hierarchical_chunker.chunk_markdown_by_headers(
        markdown=markdown,
        source_file=filename
    )
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


    for a_chunk in post_processed_chunks:
        print(f'\n{a_chunk}\n\n')
    print(len(post_processed_chunks))


   