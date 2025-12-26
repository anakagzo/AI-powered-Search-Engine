# Architecture Legend / Design Rationale

## Ingestion and Indexing Stage:
1. Data Sources:
Multiple internal and external systems provide heterogeneous data (documents, structured exports, APIs).
The architecture abstracts source-specific complexity early in the pipeline.

2. Document Normalisation (Markdown):
All inputs are converted into a single canonical Markdown format.
This simplifies downstream processing, enables consistent chunking, and preserves document structure (headers, tables, images).

3. Hierarchical + LLM-Assisted Chunking:

* Hierarchical chunking uses document headers to create coherent, context-preserving chunks in a fast, deterministic, and cost-effective way.

* LLM-assisted chunking is applied selectively for oversized, poorly structured, or media-heavy content (tables/images), ensuring robustness without unnecessary cost.

4. RAG-Optimised Chunks:
Each chunk is designed to be meaningful in isolation and enriched with metadata (headings, source, chunk type, image references), improving retrieval precision and explainability.

5. Embedding Generation:
Chunk content is transformed into vector embeddings to enable semantic similarity search alongside traditional keyword search.

6. OpenSearch (Hybrid Index):
OpenSearch stores embeddings, keyword indexes, and metadata, enabling scalable hybrid retrieval (vector + lexical + filters) across large datasets. OpenSearch is the preferred search engine (over any vector database) because of the hybrid retrieval capability and scalabilty

## Retrieval Stage:
1. Query Understanding (LLM):
User queries are converted (by LLM) into optimized search representations, combining semantic intent, keywords, and metadata filters.

2. Hybrid Retrieval & Ranking:
Relevant chunks—not full documents—are retrieved from OpenSearch storage and ranked based on semantic relevance, keyword matching, and metadata constraints.

3. Answer Generation (LLM):
The final LLM uses retrieved chunks as grounded context to produce accurate, explainable responses while minimizing hallucination risk.

## Design Principles

- Context-aware chunking over naive splitting


- Selective LLM usage for cost and latency control


- Hybrid search for accuracy and recall


- Metadata-first design for filtering, ranking, and explainability