from typing import Dict, List
from vector_store import FinancialVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import logging

class FinancialRAGChain:
    """
    A RAG (Retrieval Augmented Generation) chain for financial analysis.
    
    Important Implementation Requirements:
    1. Query Validation:
       - Empty queries MUST be rejected with the error message: "Query cannot be empty" raising a ValueError
       - Queries shorter than 10 characters MUST be rejected with the error message: 
         "Query too short. Please provide more details." raising a ValueError
       
    2. The implementation should:
       - Validate queries before processing
       - Handle both empty strings and whitespace-only strings
       - Apply validation to both query() and get_relevant_documents() methods
    """

    def __init__(self, vector_store: FinancialVectorStore):
        """Initialize RAG chain with vector store.
        Call the _create_chain function to initialize the rag_chain
        """
        self.vector_store = vector_store
        self.rag_chain = self._create_chain()

    def _create_chain(self):
        """Create the RAG chain for financial analysis.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are a financial analyst assistant. Use the following documents to answer the question.\n\n"
            "Documents: {documents}\n\n"
            "Question: {question}\n\n"
            "Answer concisely and accurately."
        )

        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",  # you can swap with flan-t5, llama-2, etc.
            torch_dtype="auto",
            device_map="auto"  # will use GPU if available
        )
        llm = HuggingFacePipeline(pipeline=generator)

        chain = prompt | llm | StrOutputParser()

        return {"prompt": prompt, "llm": chain}
        
    def _validate_query(self, query: str):
        """Validate the query before processing."""
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

    async def query(self, question: str) -> str:
        """Query the RAG chain with a financial question."""
        self._validate_query(question)

        # Retrieve relevant documents first
        docs = self.get_relevant_documents(question, k=5)
        documents_text = "\n".join([doc.page_content for doc in docs])  # Access as attribute

        # Prepare inputs for the ChatPromptTemplate
        inputs = {"documents": documents_text, "question": question}

        # Run the LLM chain
        result = await self.rag_chain['llm'].ainvoke(inputs)
        return result

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Get relevant financial documents with metadata.    
        """
        self._validate_query(query)
        # Use the vector store to retrieve top-k relevant docs
        results = self.vector_store.similarity_search(query, k=k)
        return results
