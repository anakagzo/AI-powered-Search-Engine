from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import SearchRequest, SearchResponse
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
    "/search",
    response_model=SearchResponse,
    summary="Search indexed chunks",
    description="Supports keyword, semantic, and hybrid retrieval with metadata filtering.",
)
async def search_chunks(payload: SearchRequest) -> SearchResponse:
    """Search over OpenSearch indexed chunks using selected retrieval strategy."""
    try:
        filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

        if payload.search_type == "keyword":
            hits = opensearch_service.keyword_search(
                query=payload.query,
                top_k=payload.top_k,
                filters=filters,
            )
        elif payload.search_type == "semantic":
            query_vector = embedding_service.embed_query(payload.query)
            hits = opensearch_service.semantic_search(
                query_vector=query_vector,
                top_k=payload.top_k,
                filters=filters,
            )
        else:
            if payload.keyword_weight + payload.semantic_weight == 0:
                raise HTTPException(
                    status_code=400,
                    detail="keyword_weight and semantic_weight cannot both be zero.",
                )

            query_vector = embedding_service.embed_query(payload.query)
            hits = opensearch_service.hybrid_search(
                query=payload.query,
                query_vector=query_vector,
                top_k=payload.top_k,
                keyword_weight=payload.keyword_weight,
                semantic_weight=payload.semantic_weight,
                filters=filters,
            )

        return SearchResponse(
            status="success",
            search_type=payload.search_type,
            total_hits=len(hits),
            results=hits,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
