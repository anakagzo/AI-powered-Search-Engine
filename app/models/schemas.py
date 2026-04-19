from typing import Literal, Optional

from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    '''Metadata for a chunk of text'''
    source: str
    chunk_type: str
    heading: Optional[str] = None
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
    chunks_indexed: int
    index_name: str


class MetadataFilter(BaseModel):
    '''Optional metadata filters for retrieval.'''
    source: Optional[str] = None
    chunk_type: Optional[str] = None
    heading: Optional[str] = None
    sub_heading: Optional[str] = None
    section_title: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SearchRequest(BaseModel):
    '''Request model for retrieval queries.'''
    query: str = Field(..., min_length=1)
    search_type: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[MetadataFilter] = None
    keyword_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    '''A single retrieval hit returned by OpenSearch.'''
    chunk_id: str
    score: float
    content: str
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    '''Response model for search endpoint.'''
    status: str
    search_type: str
    total_hits: int
    results: list[SearchResult]
