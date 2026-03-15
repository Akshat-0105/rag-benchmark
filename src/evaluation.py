import json
from pathlib import Path
from typing import Optional

class RetrievalEvaluator:    
    def __init__(self, queries_path: str):
        self.queries = self._load_queries(queries_path)
    
    def _load_queries(self, queries_path: str) -> list[dict]:
        path = Path(queries_path)
        
        if not path.exists():
            print(f"Warning: Queries file not found: {queries_path}")
            return []
        
        try:
            with open(path, "r") as f:
                queries = json.load(f)
            print(f"Loaded {len(queries)} test queries from {queries_path}")
            return queries
        except Exception as e:
            print(f"Error loading queries: {e}")
            return []
    
    def precision_at_k(self, retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:

        if k <= 0:
            return 0.0

        top_k = retrieved_sources[:k]

        retrieved_unique = set(top_k)
        relevant_unique = set(relevant_sources)

        relevant_count = len(retrieved_unique.intersection(relevant_unique))

        return relevant_count / min(k, len(retrieved_unique))
    
    def recall_at_k(self, retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:

        if not relevant_sources:
            return 0.0

        top_k = retrieved_sources[:k]

        # use unique sources
        retrieved_unique = set(top_k)
        relevant_unique = set(relevant_sources)

        relevant_count = len(retrieved_unique.intersection(relevant_unique))

        return relevant_count / len(relevant_unique)
    
    def evaluate_retrieval(self, query: str, retrieved_docs: list, k: int = 5) -> Optional[dict]:
        # Find query in test set
        query_entry = None
        for q in self.queries:
            if q.get("query").lower() == query.lower():
                query_entry = q
                break
        
        if not query_entry:
            print(f"Query not found in test set: {query}")
            return None
        
        # Extract source names from retrieved documents
        retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
        relevant_sources = query_entry.get("relevant_docs", [])
        
        # Calculate metrics
        precision = self.precision_at_k(retrieved_sources, relevant_sources, k)
        recall = self.recall_at_k(retrieved_sources, relevant_sources, k)
        
        return {
            "query": query,
            "k": k,
            "precision@k": precision,
            "recall@k": recall,
            "retrieved_count": len(retrieved_docs),
            "relevant_count": len(relevant_sources),
        }
    
    def batch_evaluate(self, retrieval_results: list[dict], k: int = 5) -> dict:
        metrics = []
        
        for result in retrieval_results:
            query = result.get("query")
            retrieved_docs = result.get("retrieved_docs", [])
            
            metric = self.evaluate_retrieval(query, retrieved_docs, k)
            if metric:
                metrics.append(metric)
        
        # Calculate aggregate statistics
        if metrics:
            avg_precision = sum(m["precision@k"] for m in metrics) / len(metrics)
            avg_recall = sum(m["recall@k"] for m in metrics) / len(metrics)
            
            return {
                "total_queries": len(metrics),
                "avg_precision@k": avg_precision,
                "avg_recall@k": avg_recall,
                "query_metrics": metrics
            }
        
        return {"total_queries": 0, "avg_precision@k": 0, "avg_recall@k": 0}
