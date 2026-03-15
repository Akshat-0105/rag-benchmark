import google.generativeai as genai
import os
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from retrieval import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_model: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}")
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant. Answer the following question based on the provided context.

                Context:
                {context}

                Question:
                {question}

                Answer:"""
        )
    
    def generate(self, query: str, k: int = 5) -> dict:
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")
        
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, k=k)
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # Generate answer using LLM
        prompt = self.prompt_template.format(context=context, question=query)
        
        print("Generating answer with LLM...")
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        print(f"\nGenerated Answer:\n{answer}\n")
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "num_retrieved": len(retrieved_docs)
        }
    
    def _format_context(self, documents: list[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
