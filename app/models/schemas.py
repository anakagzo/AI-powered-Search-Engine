from pydantic import BaseModel
from typing import Optional

class ChunkMetadata(BaseModel):
    '''Metadata for a chunk of text'''
    source: str
    chunk_type: str
    heading: str
    sub_heading: Optional[str]
    section_title: str
    image_paths: Optional[str] = None
    date: str
    


class Chunk(BaseModel):
    '''A chunk of text with metadata'''
    chunk_id: str
    content: str
    metadata: ChunkMetadata


class UploadResponse(BaseModel):
    '''Response model for upload endpoint'''
    status: str
    filename: str
    chunks_created: int
    chunks: list[Chunk]
