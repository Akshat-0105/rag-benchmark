from langchain.schema import Document
from vector_store import FAISSVectorStore


class Retriever:
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def rewrite_query(self, query: str) -> str:
        # Simple normalization + expansion
        query = query.strip().lower()

        if "supply chain" not in query:
            query = f"supply chain {query}"

        return query
    
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        print(f"Retrieving top-{k} documents for query: '{query}'")
        
        # Search vector store
        rewritten_query = self.rewrite_query(query)
        results = self.vector_store.search(rewritten_query, k=max(k*2, 10))
        
        # Rerank by similarity score (lower score = better in FAISS)
        results = self.vector_store.search(rewritten_query, k=max(k*2, 10))

        # Rerank using embedding similarity
        query_embedding = self.vector_store.embedding_model.embed_query(rewritten_query)
        
        reranked = []
        for doc, score in results:
            doc_embedding = self.vector_store.embedding_model.embed_query(doc.page_content)
            similarity = sum(q*d for q, d in zip(query_embedding, doc_embedding))
            reranked.append((doc, similarity))
        
        # Sort by similarity (higher is better)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        top_results = reranked[:k]
        
        documents = [doc for doc, score in top_results]
        
        print(f"Retrieved {len(documents)} documents")
        return documents
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        results = self.vector_store.search(query, k=k)
        return results
    
    def rewrite_query(self, query: str) -> str:
        # Simple normalization + expansion
        query = query.strip().lower()

        if "supply chain" not in query:
            query = f"supply chain {query}"

        return query
