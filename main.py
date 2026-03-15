import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import load_pdfs_from_folder, create_sample_documents
from chunking import get_chunking_strategy
from embeddings import get_embedding_model
from vector_store import FAISSVectorStore
from retrieval import Retriever
from rag_pipeline import RAGPipeline
from evaluation import RetrievalEvaluator


def run_experiment(
    chunking_strategy: str,
    embedding_model: str,
    k_retrieve: int = 5,
    data_folder: str = "data/raw_docs",
    queries_file: str = "queries/test_queries.json",
    results_file: str = "experiments/results.csv"
) -> dict:
    print(f"Starting experiment: Strategy={chunking_strategy}, Model={embedding_model}")
    print(f"{'='*70}")
    
    try:
        # Step 1: Load documents
        print("\n[1/5] Loading documents...")
        data_path = Path(data_folder)
        if data_path.exists() and list(data_path.glob("*.pdf")):
            documents = load_pdfs_from_folder(str(data_path))
        else:
            print("No PDF files found. Using sample documents for demo.")
            documents = create_sample_documents()
        
        if not documents:
            raise ValueError("No documents loaded")
        
        # Step 2: Chunk documents
        print("\n[2/5] Chunking documents...")
        chunking = get_chunking_strategy(chunking_strategy)
        chunked_docs = chunking.chunk(documents)
        
        # Step 3: Initialize embedding model and build index
        print("\n[3/5] Building vector index...")
        embedding = get_embedding_model(embedding_model)
        vector_store = FAISSVectorStore(embedding)
        vector_store.build_index(chunked_docs)
        
        # Step 4: Initialize RAG pipeline
        print("\n[4/5] Setting up RAG pipeline...")
        retriever = Retriever(vector_store)
        rag = RAGPipeline(retriever)
        
        # Step 5: Evaluate retrieval quality
        print("\n[5/5] Evaluating retrieval metrics...")
        evaluator = RetrievalEvaluator(queries_file)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "k_retrieve": k_retrieve,
            "total_chunks": len(chunked_docs),
            "total_original_docs": len(documents),
            "metrics": {}
        }
        
        # Run test queries
        if evaluator.queries:
            test_results = []
            latencies = []

            for query_entry in evaluator.queries:
                query = query_entry.get("query")

                start_time = time.time()
                retrieved = retriever.retrieve(query, k=k_retrieve)
                latency = time.time() - start_time

                latencies.append(latency)

                test_results.append({
                    "query": query,
                    "retrieved_docs": retrieved
                })
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            results["metrics"]["avg_latency_sec"] = avg_latency

            # Get metrics
            eval_result = evaluator.batch_evaluate(test_results, k=k_retrieve)
            results["metrics"] = {
                "avg_precision@k": eval_result.get("avg_precision@k", 0),
                "avg_recall@k": eval_result.get("avg_recall@k", 0),
                "total_queries_tested": eval_result.get("total_queries", 0)
            }
        
        print(f"\n✓ Experiment completed successfully")
        print(f"  - Chunks created: {results['total_chunks']}")
        if results["metrics"]:
            print(f"  - Avg Precision@{k_retrieve}: {results['metrics']['avg_precision@k']:.3f}")
            print(f"  - Avg Recall@{k_retrieve}: {results['metrics']['avg_recall@k']:.3f}")
            print(f"  - Avg Retrieval Latency: {avg_latency:.3f} seconds")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during experiment: {e}")
        return {
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "error": str(e)
        }


def save_results(results: dict, results_file: str = "experiments/results.csv") -> None:
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we write header
    file_exists = results_path.exists()
    
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "timestamp",
                "chunking_strategy",
                "embedding_model",
                "k_retrieve",
                "total_chunks",
                "avg_precision@k",
                "avg_recall@k",
                "error"
            ])
        
        # Write result row
        writer.writerow([
            results.get("timestamp", ""),
            results.get("chunking_strategy", ""),
            results.get("embedding_model", ""),
            results.get("k_retrieve", ""),
            results.get("total_chunks", ""),
            results.get("metrics", {}).get("avg_precision@k", ""),
            results.get("metrics", {}).get("avg_recall@k", ""),
            results.get("error", "")
        ])
    
    print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --strategy A --model openai
  python main.py --strategy B --model sentence-transformers
  python main.py --all  # Run all combinations
        """
    )
    
    parser.add_argument(
        "--strategy",
        choices=["A", "B"],
        help="Chunking strategy (A or B)"
    )
    parser.add_argument(
        "--model",
        choices=["openai", "sentence-transformers"],
        help="Embedding model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all strategy and model combinations"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--data",
        default="data/raw_docs",
        help="Path to PDF folder (default: data/raw_docs)"
    )
    parser.add_argument(
        "--queries",
        default="queries/test_queries.json",
        help="Path to test queries JSON (default: queries/test_queries.json)"
    )
    parser.add_argument(
        "--output",
        default="experiments/results.csv",
        help="Path to results CSV (default: experiments/results.csv)"
    )
    
    args = parser.parse_args()
    
    # Determine experiments to run
    if args.all:
        experiments = [
            ("A", "sentence-transformers"),
            ("B", "sentence-transformers"),
            ]
    elif args.strategy and args.model:
        experiments = [(args.strategy, args.model)]
    else:
        parser.print_help()
        print("\nError: Specify --strategy and --model, or use --all")
        sys.exit(1)
    
    print("=" * 70)
    print("RAG BENCHMARKING SYSTEM")
    print("=" * 70)
    print(f"Running {len(experiments)} experiment(s)...")
    
    # Run experiments
    all_results = []
    for strategy, model in experiments:
        result = run_experiment(
            chunking_strategy=strategy,
            embedding_model=model,
            k_retrieve=args.k,
            data_folder=args.data,
            queries_file=args.queries,
            results_file=args.output
        )
        all_results.append(result)
        save_results(result, args.output)
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for result in all_results:
        strategy = result.get("chunking_strategy", "?")
        model = result.get("embedding_model", "?")
        
        if "error" in result:
            print(f"✗ Strategy {strategy} + {model}: {result['error']}")
        else:
            chunks = result.get("total_chunks", 0)
            precision = result.get("metrics", {}).get("avg_precision@k", 0)
            recall = result.get("metrics", {}).get("avg_recall@k", 0)
            print(f"✓ Strategy {strategy} + {model}: {chunks} chunks | P@k={precision:.3f} | R@k={recall:.3f}")
    
    print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()
