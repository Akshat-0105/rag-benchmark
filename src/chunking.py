from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class ChunkingStrategy:
    def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class StrategyA(ChunkingStrategy):
    def __init__(self):
        self.chunk_size = 500
        self.overlap = 50
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        chunked = []
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "strategy": "A"
                    }
                )
                chunked.append(chunked_doc)
        
        print(f"Strategy A: {len(documents)} docs -> {len(chunked)} chunks (size=500, overlap=50)")
        return chunked


class StrategyB(ChunkingStrategy):
    def __init__(self):
        self.chunk_size = 1000
        self.overlap = 200
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        chunked = []
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "strategy": "B"
                    }
                )
                chunked.append(chunked_doc)
        
        print(f"Strategy B: {len(documents)} docs -> {len(chunked)} chunks (size=1000, overlap=200)")
        return chunked


def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    strategies = {
        "A": StrategyA,
        "B": StrategyB,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name]()
