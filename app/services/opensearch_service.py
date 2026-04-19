from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers


class OpenSearchService:
    """Handles OpenSearch index lifecycle, document ingestion, and retrieval."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        use_ssl: bool,
        verify_certs: bool,
        index_name: str,
    ) -> None:
        """Initialize an authenticated OpenSearch client."""
        self.index_name = index_name
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )

    def ensure_index(self, vector_dim: int) -> None:
        """Create index with metadata and vector mapping if it does not exist."""
        if self.client.indices.exists(index=self.index_name):
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": vector_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                    "source": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "heading": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "sub_heading": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "section_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "image_paths": {"type": "keyword"},
                    "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=index_body)

    def index_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        """Bulk index chunk documents and their vectors into OpenSearch."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match")

        if not chunks:
            return 0

        self.ensure_index(vector_dim=len(embeddings[0]))

        actions = []
        for chunk, vector in zip(chunks, embeddings):
            metadata = chunk["metadata"]
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": chunk["chunk_id"],
                    "_op_type": "index",
                    "_source": {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "embedding": vector,
                        "source": metadata.get("source"),
                        "chunk_type": metadata.get("chunk_type"),
                        "heading": metadata.get("heading"),
                        "sub_heading": metadata.get("sub_heading"),
                        "section_title": metadata.get("section_title"),
                        "image_paths": metadata.get("image_paths"),
                        "date": metadata.get("date"),
                    },
                }
            )

        success, _ = helpers.bulk(self.client, actions, raise_on_error=True)
        self.client.indices.refresh(index=self.index_name)
        return success

    def _build_filter_clauses(self, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Translate metadata filters into OpenSearch bool filter clauses."""
        if not filters:
            return []

        clauses: list[dict[str, Any]] = []

        exact_keyword_fields = ["source", "chunk_type", "heading", "sub_heading", "section_title"]
        for field in exact_keyword_fields:
            value = filters.get(field)
            if value:
                keyword_field = field if field in {"source", "chunk_type"} else f"{field}.keyword"
                clauses.append({"term": {keyword_field: value}})

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from or date_to:
            range_filter: dict[str, Any] = {}
            if date_from:
                range_filter["gte"] = date_from
            if date_to:
                range_filter["lte"] = date_to
            clauses.append({"range": {"date": range_filter}})

        return clauses

    @staticmethod
    def _normalize_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenSearch hits to API response shape."""
        normalized: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {})
            normalized.append(
                {
                    "chunk_id": source.get("chunk_id"),
                    "score": float(hit.get("_score", 0.0)),
                    "content": source.get("content", ""),
                    "metadata": {
                        "source": source.get("source"),
                        "chunk_type": source.get("chunk_type"),
                        "heading": source.get("heading"),
                        "sub_heading": source.get("sub_heading"),
                        "section_title": source.get("section_title"),
                        "image_paths": source.get("image_paths"),
                        "date": source.get("date"),
                    },
                }
            )
        return normalized

    def keyword_search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run lexical retrieval using BM25 and metadata filters."""
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^3",
                                    "section_title^2",
                                    "heading^2",
                                    "sub_heading",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": self._build_filter_clauses(filters),
                }
            },
        }
        response = self.client.search(index=self.index_name, body=search_body)
        return self._normalize_hits(response["hits"]["hits"])

    def semantic_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run vector similarity retrieval with metadata filters."""
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": self._build_filter_clauses(filters),
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                }
                            }
                        }
                    ],
                }
            },
        }
        response = self.client.search(index=self.index_name, body=search_body)
        return self._normalize_hits(response["hits"]["hits"])

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        keyword_weight: float,
        semantic_weight: float,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Combine keyword and vector rankings using weighted reciprocal rank fusion."""
        expanded_k = min(100, top_k * 3)
        keyword_hits = self.keyword_search(query=query, top_k=expanded_k, filters=filters)
        semantic_hits = self.semantic_search(query_vector=query_vector, top_k=expanded_k, filters=filters)

        # Reciprocal rank fusion is robust because lexical and vector scores are not directly comparable.
        rrf_constant = 60
        merged: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(keyword_hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**hit, "score": 0.0}
            merged[chunk_id]["score"] += keyword_weight * (1.0 / (rrf_constant + rank))

        for rank, hit in enumerate(semantic_hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**hit, "score": 0.0}
            merged[chunk_id]["score"] += semantic_weight * (1.0 / (rrf_constant + rank))

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
