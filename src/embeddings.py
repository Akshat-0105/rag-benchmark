from abc import ABC, abstractmethod
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingModel(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self):
        try:
            self.model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"trust_remote_code": True}
            )
            self._dimension = 384  # bge-small-en-v1.5 dimension
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer embeddings: {e}")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
    
    def get_dimension(self) -> int:
        return self._dimension


def get_embedding_model(model_name: str) -> EmbeddingModel:
    models = {
        "sentence-transformers": SentenceTransformerEmbeddingModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    print(f"Initializing embedding model: {model_name}")
    return models[model_name]()
