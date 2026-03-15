# RAG Benchmarking System

A modular **Retrieval-Augmented Generation (RAG) benchmarking system** designed to evaluate how different document chunking strategies affect retrieval performance in document-based question answering systems.

This project was developed as part of the **GCPL AI Intern Hackathon**.

---

# Overview

Retrieval-Augmented Generation (RAG) systems rely heavily on how documents are segmented before indexing.  
This project benchmarks two chunking strategies and evaluates their impact on retrieval quality using vector search.

The system ingests PDF documents, generates embeddings, stores them in a FAISS vector database, retrieves relevant chunks for queries, and evaluates retrieval performance using standard IR metrics.

---

# Features

- PDF document ingestion
- Two chunking strategies for comparison
- Vector embeddings using SentenceTransformers
- FAISS vector database for similarity search
- Retrieval benchmarking
- Precision@K and Recall@K evaluation
- Query rewriting
- Reranking of retrieved documents
- Retrieval latency analysis
- Modular architecture

---

# System Architecture

```
PDF Documents
      ↓
Document Loader (PyPDF)
      ↓
Chunking
  Strategy A: 500 tokens
  Strategy B: 1000 tokens
      ↓
Embedding Model
  BAAI/bge-small-en-v1.5
      ↓
Vector Database
  FAISS
      ↓
Retriever
  Query Rewriting
  Top-K Retrieval
  Reranking
      ↓
Evaluation
  Precision@5
  Recall@5
  Latency
      ↓
LLM
  Gemini 1.5 Flash
```

---

# Project Structure

```
rag-benchmark/
│
├── data/
│   └── raw_docs/
│       └── (PDF dataset placed here)
│
├── queries/
│   └── test_queries.json
│
├── experiments/
│   └── results.csv
│
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── rag_pipeline.py
│   ├── retrieval.py
│   └── vector_store.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Dataset

The system was evaluated using **10 research papers on AI in supply chains**.

Dataset characteristics:

- Total documents: **10**
- Total pages: **~200**
- Domain: AI applications in supply chain management

Due to copyright restrictions, the PDF dataset is **not included in this repository**.

To reproduce the experiments, place your PDFs inside:

```
data/raw_docs/
```

---

# Embedding Model

This system uses:

```
BAAI/bge-small-en-v1.5
```

Reasons for choosing this model:

- High-quality semantic embeddings
- Lightweight and efficient
- Open-source
- Works well for retrieval tasks

---

# Vector Database

The project uses:

```
FAISS (Facebook AI Similarity Search)
```

Advantages:

- Extremely fast vector similarity search
- Efficient for large embedding collections
- Widely used in modern RAG pipelines

---

# Evaluation Metrics

The system evaluates retrieval performance using:

### Precision@5

```
Precision@5 = Relevant retrieved documents / 5
```

Measures how many of the top retrieved documents are relevant.

### Recall@5

```
Recall@5 = Relevant retrieved documents / Total relevant documents
```

Measures how many relevant documents were successfully retrieved.

### Retrieval Latency

Measures the time required to retrieve documents for a query.

---

# Benchmark Experiments

Two chunking strategies were evaluated.

| Strategy | Chunk Size | Overlap |
|--------|--------|--------|
| A | 500 tokens | 50 |
| B | 1000 tokens | 200 |

---

# Results

| Strategy | Precision@5 | Recall@5 | Avg Latency |
|--------|--------|--------|--------|
| A | 0.55 | 0.47 | 0.37 s |
| B | 0.48 | 0.35 | 0.68 s |

### Observation

Smaller chunks improved retrieval precision and recall because they preserved more fine-grained semantic context within document segments.

---

# Bonus Features

The system includes several additional improvements:

### Query Rewriting
Queries are normalized and expanded before retrieval to improve semantic matching.

### Reranking
Retrieved candidate documents are reranked using embedding similarity before selecting the final results.

### Latency Analysis
Retrieval latency is measured to evaluate system efficiency.

---

# Running the System

Install dependencies:

```
pip install -r requirements.txt
```

Run Strategy A experiment:

```
python main.py --strategy A --model sentence-transformers
```

Run Strategy B experiment:

```
python main.py --strategy B --model sentence-transformers
```

Results will be stored in:

```
experiments/results.csv
```

---

# Example Query Evaluation

Example query:

```
How can AI improve demand forecasting in supply chains?
```

The system retrieves the most semantically similar document chunks and evaluates whether the retrieved documents match the ground truth relevance labels.

---

# Limitations

- Evaluation uses a relatively small dataset
- Reranking uses simple embedding similarity
- Relevance labels are defined at the document level
- More sophisticated hybrid retrieval methods could further improve results

---

# Future Improvements

Possible extensions include:

- Hybrid retrieval (BM25 + vector search)
- Cross-encoder reranking models
- Larger evaluation datasets
- LLM-based query expansion
- Multi-hop retrieval experiments

---

# Acknowledgements

This project was developed for the **GCPL AI Intern Hackathon** and demonstrates the design and evaluation of a modular RAG benchmarking pipeline.
