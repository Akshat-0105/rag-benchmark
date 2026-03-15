import pickle
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from embeddings import EmbeddingModel


class FAISSVectorStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.faiss_index = None
        self.documents = []
    
    def build_index(self, documents: list[Document]) -> "FAISSVectorStore":
        if not documents:
            raise ValueError("Cannot build index with empty document list")
        
        print(f"Building FAISS index with {len(documents)} documents...")
        
        # Create FAISS index from documents
        # This uses langchain's FAISS wrapper which handles embedding internally
        self.faiss_index = FAISS.from_documents(
            documents,
            self.embedding_model.model
        )
        self.documents = documents
        
        print(f"Index built successfully")
        return self
    
    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Use similarity_search_with_score for relevance scores
        results = self.faiss_index.similarity_search_with_score(query, k=k)
        return results
    
    def save_index(self, save_path: str) -> None:
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss_index.save_local(str(save_dir))
        
        # Save documents metadata separately
        metadata_path = save_dir / "documents_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"Index saved to {save_path}")
    
    def load_index(self, load_path: str) -> "FAISSVectorStore":
        load_dir = Path(load_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Index path not found: {load_path}")
        
        # Load FAISS index
        self.faiss_index = FAISS.load_local(
            str(load_dir),
            self.embedding_model.model
        )
        
        # Load documents metadata
        metadata_path = load_dir / "documents_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)
        
        print(f"Index loaded from {load_path}")
        return self
    
    def get_docs_count(self) -> int:
        return len(self.documents) if self.documents else 0
