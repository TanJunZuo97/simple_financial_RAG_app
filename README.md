# Financial Insights and Analysis System Using RAG

## Problem Description

Create a Financial Insights and Analysis system using Retrieval-Augmented Generation (RAG) that helps financial analysts access and understand insights from a vast repository of financial reports, market trends, and economic research.

### Requirements

1. **Data Processing**
   - Process financial reports and market analysis documents
   - Handle various financial metrics and indicators
   - Create embeddings for efficient financial information retrieval

2. **Core Features**
   - Search through financial reports and market analyses
   - Answer questions about market trends and economic indicators
   - Provide insights from recent financial data
   - Handle natural language queries about financial topics

3. **Technical Requirements**
   - Use LangChain for RAG implementation
   - Use FAISS for vector storage
   - Use a local Hugging Face `text2text-generation` pipeline (`google/flan-t5-base`) for text generation
   - Use Hugging Face sentence-transformers (`sentence-transformers/all-mpnet-base-v2`) for embeddings
   - Streamlit for the user interface

4. **Implementation Details**
   - Implement proper error handling
   - Ensure efficient vector search
   - Create a user-friendly interface
   - Handle empty and incomplete queries before retrieval

### Tasks to Complete

1. Implement the `FinancialDocumentLoader` class in `document_loader.py`
2. Implement the `FinancialVectorStore` class in `vector_store.py`
3. Implement the `FinancialRAGChain` class in `rag_chain.py`
4. Make sure all tests pass.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

On first run, the app builds a FAISS vector store from `financial_reports.json` and saves it to `vector_store/`. Subsequent runs load the saved vector store instead of rebuilding it. The embedding model and text-generation pipeline are downloaded from Hugging Face on first use, so an internet connection is required the first time the app runs.

### Sample Data Format
```json
[
    {
        "reportId": "FR001",
        "title": "Quarterly Global Forex Report - January 2024",
        "author": "Maria Trade",
        "publication_date": "2024-01-27",
        "category": "Global Forex",
        "summary": "Detailed analysis of global forex trends and market indicators for January 2024, focusing on key developments and future outlook.",
        "key_statistics": [
            "Forex Volatility: 7.3%",
            "USD/JPY: 147.1",
            "GBP/USD: 1.210",
            "EUR/USD: 1.140"
        ],
        "sentiment_score": 0.76
    },
    {
        "reportId": "FR002",
        "title": "Special Global Forex Overview - December 2023",
        "author": "David Exchange",
        "publication_date": "2023-12-13",
        "category": "Global Forex",
        "summary": "Detailed analysis of global forex trends and market indicators for December 2023, focusing on key developments and future outlook.",
        "key_statistics": [
            "GBP/USD: 1.281",
            "EUR/USD: 1.188",
            "USD/JPY: 126.0",
            "Forex Volatility: 9.6%"
        ],
        "sentiment_score": 0.89
    }
]
```