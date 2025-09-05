from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import pickle

logger = logging.getLogger("financial_vector_store")
class FinancialVectorStore:
    """This class handles the creation, storage, and retrieval of vector embeddings for Financial 
    
    IMPORTANT:
    - Empty queries (null or whitespace-only) must be rejected with an empty result list
    - Queries shorter than 10 characters must be rejected with an empty result list
    """
    
    def __init__(self):
        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_name = "BAAI/bge-base-en-v1.5"
        """Initialize the vector store with Hugging Face embeddings."""
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"FinancialVectorStore initialized with Hugging Face model: {model_name}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store from documents.
        TODO: Implement this method
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vector store created with {len(documents)} documents")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Query the vector store for similar financial documents.
        Raise a ValueError with the error message "Vector store not initialized" if the vector store is not initialized.
        
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning using logger but DO NOT raise an exception
        
        Args:
            query (str): Query text to find similar documents
            k (int): Number of similar documents to return per collection
            
        Returns:
            List[Document]: List of similar Langchain Documents retrieved
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        if not query or query.strip() == "":
            logger.warning("Empty or whitespace-only query provided")
            return []

        if len(query.strip()) < 10:
            logger.warning("Query too short (<10 characters)")
            return []

        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def save_local(self, path: str) -> None:
        """Save vector store locally for loading it later on.
        TODO: Implement this method
        """
        if not self.vector_store:
            logger.warning("Cannot save uninitialized vector store")
            return

        os.makedirs(path, exist_ok=True)
        vector_path = os.path.join(path, "faiss_index")
        meta_path = os.path.join(path, "vector_store_meta.pkl")

        self.vector_store.save_local(vector_path)
        # Save any additional metadata if needed
        with open(meta_path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings}, f)

        logger.info(f"Vector store saved locally at {path}")

    @classmethod
    def load_local(cls, directory: str) -> 'FinancialVectorStore':
        """
        Load a vector store from local storage.
        
        Args:
            directory (str): Directory path containing the vector store
            
        Returns:
            SupportVectorStore: Loaded vector store instance
        """
        vector_path = os.path.join(directory, "faiss_index")
        meta_path = os.path.join(directory, "vector_store_meta.pkl")

        if not os.path.exists(vector_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Vector store not found in {directory}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls()
        instance.embeddings = meta["embeddings"]
        instance.vector_store = FAISS.load_local(vector_path, instance.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vector store loaded from {directory}")
        return instance