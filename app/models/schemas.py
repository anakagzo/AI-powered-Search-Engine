from pydantic import BaseModel
from typing import Optional

class ChunkMetadata(BaseModel):
    source: str
    chunk_type: str
    heading: str
    sub_heading: Optional[str]
    section_title: str
    image_paths: Optional[str] = None
    date: str
    


class Chunk(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    chunks: list[Chunk]
