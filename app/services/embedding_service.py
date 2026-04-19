from typing import Optional

from openai import OpenAI


class OpenAIEmbeddingService:
    """Creates vector embeddings for chunk text using OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ) -> None:
        """Initialize embedding client and model configuration.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            dimensions: Optional output vector dimensions for supported models.
            batch_size: Number of texts to embed per request.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return vectors in the same order."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            request_payload = {
                "model": self.model,
                "input": batch,
            }

            if self.dimensions is not None:
                request_payload["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**request_payload)
            all_vectors.extend(item.embedding for item in response.data)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query string."""
        vectors = self.embed_texts([query])
        return vectors[0]
