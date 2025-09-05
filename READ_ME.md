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
   - Use OpenAI's gpt-4o for text generation
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