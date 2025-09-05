import asyncio
import logging
import streamlit as st

from document_loader import FinancialDocumentLoader
from rag_chain import FinancialRAGChain
from vector_store import FinancialVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("financial_rag")
VECTOR_STORE_DIR = "vector_store"
DATA_PATH = "financial_reports.json"
status_placeholder = st.empty()
progress_bar = st.progress(0)

def log_error(e: Exception) -> str:
    logger.error(e, exc_info=True)
    return f"❌ Error: {str(e)}"

def get_documents():
    """Load financial documents"""
    try:
        status_placeholder.info("📚 Loading financial reports...")
        loader = FinancialDocumentLoader(DATA_PATH)
        documents = loader.create_documents()
        status_placeholder.success("✅ Financial reports loaded successfully!")
        return documents
    except NotImplementedError:
        status_placeholder.error("⚠️ FinancialDocumentLoader methods need to be implemented")

def create_new_vector_store():
    """Create a new vector store from scratch"""
    try:
        vector_store = FinancialVectorStore()
        status_placeholder.info("⚙️ Creating new vector store...")
        logging.info("⚙️ Creating new vector store...")
        progress_bar.progress(40)

        documents = get_documents()
        progress_bar.progress(60)

        status_placeholder.info("🔨 Generating embeddings...")
        logging.info("🔨 Generating embeddings...")
        vector_store.create_vector_store(documents)
        progress_bar.progress(80)

        status_placeholder.info("💾 Saving vector store...")
        logging.info("💾 Saving vector store...")
        vector_store.save_local(VECTOR_STORE_DIR)
        progress_bar.progress(100)

        status_placeholder.success("✅ Vector store created and saved successfully!")
        return vector_store
    except NotImplementedError:
        status_placeholder.error("⚠️ FinancialVectorStore methods need to be implemented")

def load_existing_vector_store():
    """Load an existing vector store"""
    try:
        status_placeholder.info("🔄 Loading existing vector store...")
        progress_bar.progress(30)
        vector_store = FinancialVectorStore.load_local(VECTOR_STORE_DIR)
        progress_bar.progress(100)
        status_placeholder.success("✅ Vector store loaded successfully!")
        return vector_store
    except NotImplementedError:
        status_placeholder.error("⚠️ FinancialVectorStore methods need to be implemented")

def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        print('Initializing RAG system')
        try:
            vector_store = load_existing_vector_store()
        except Exception as e:
            log_error(e)
            logging.info(f'Creating new vector store')
            vector_store = create_new_vector_store()

        status_placeholder.info("🤖 Initializing Financial RAG chain...")
        rag_chain = FinancialRAGChain(vector_store)
        status_placeholder.empty()
        return rag_chain
    except NotImplementedError:
        status_placeholder.error("⚠️ FinancialRAGChain methods need to be implemented")
    except Exception as e:
        error_msg = log_error(e)
        status_placeholder.error(error_msg)

def display_implementation_guide():
    """Display guide for implementing required methods"""
    st.error("⚠️ System initialization failed: Some components need implementation")
    st.info(
        """
    Please implement the following methods:
    
    1. In `src/document_loader.py`:
       - `load_reports()`: Load and parse financial reports
       - `create_documents()`: Convert reports to LangChain documents
    
    2. In `src/vector_store.py`:
       - `create_vector_store()`: Create FAISS index
       - `save_local()`: Save vector store to disk
       - `load_local()`: Load vector store from disk
    
    3. In `src/rag_chain.py`:
       - `get_relevant_documents()`: Retrieve similar documents
       - `query()`: Generate responses using RAG
    
    Check the logs for detailed error information.
    """
    )

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = initialize_rag_system()

def main():
    progress_bar.empty()
    st.title("Financial Insights and Analysis System")
    st.write("Ask questions about market trends, economic indicators, and financial reports!")

    query = st.text_input(
        "Enter your financial query:",
        placeholder="e.g., 'What were the key market trends in Q4 2023?'"
    )

    if st.button("Search") and query:
        if not st.session_state.rag_chain:
            display_implementation_guide()
            return
        try:
            with st.spinner("🔍 Searching relevant financial reports..."):
                relevant_docs = st.session_state.rag_chain.get_relevant_documents(query)

            with st.spinner("🤖 Generating analysis..."):
                response = asyncio.run(st.session_state.rag_chain.query(query))

            st.subheader("Analysis")
            st.write(response)

            st.subheader("Relevant Reports")
            for i, doc in enumerate(relevant_docs, 1):
                with st.expander(f"{i}. {doc.metadata['title']} ({doc.metadata.get('publication_date', 'N/A')})"):
                    st.write(f"**Category:** {doc.metadata.get('category', 'N/A')}")
                    st.write(f"**Author:** {doc.metadata.get('author', 'Unknown')}")
                    st.write(f"**Key Statistics:** {doc.metadata.get('key_statistics', 'N/A')}")
                    st.write(f"**Content:** {doc.page_content}")

        except Exception as e:
            log_error(e)
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 
