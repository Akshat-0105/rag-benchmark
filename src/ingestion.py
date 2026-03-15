from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdfs_from_folder(folder_path: str) -> list[Document]:
    documents = []
    folder = Path(folder_path)
    
    # Find all PDF files in the folder
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {folder_path}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Load each PDF
    for pdf_file in pdf_files:
        try:
            print(f"Loading: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            # Add source metadata
            for page in pages:
                page.metadata["source"] = pdf_file.name
                
            documents.extend(pages)
            print(f"  - Loaded {len(pages)} pages")
            
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def create_sample_documents() -> list[Document]:
    samples = [
        Document(
            page_content="Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. ML algorithms build models based on training data.",
            metadata={"source": "ml_101.txt", "page": 1}
        ),
        Document(
            page_content="Deep Learning uses neural networks with multiple layers to process complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition.",
            metadata={"source": "dl_intro.txt", "page": 1}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is the branch of AI concerned with the interaction between computers and human language. It enables machines to understand, interpret, and generate human language.",
            metadata={"source": "nlp_guide.txt", "page": 1}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and contextual responses.",
            metadata={"source": "rag_overview.txt", "page": 1}
        ),
        Document(
            page_content="Vector embeddings convert text into high-dimensional numerical representations. These embeddings capture semantic meaning and enable similarity-based retrieval in vector databases.",
            metadata={"source": "embeddings_guide.txt", "page": 1}
        ),
    ]
    return samples
