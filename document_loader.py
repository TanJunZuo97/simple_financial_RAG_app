import os
import json
from typing import Dict, List
from langchain_core.documents import Document

class FinancialDocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.reports = []

    def load_reports(self) -> List[Dict]:
        """Load financial reports from data file.
        Raise FileNotFoundError if data_path does not exists.
        Raise ValueError if the file is not a valid JSON file.
        TODO: Implement this method
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.reports = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {self.data_path}")

        # Ensure the reports is a list of dictionaries
        if not isinstance(self.reports, list) or not all(isinstance(r, dict) for r in self.reports):
            raise ValueError("JSON file must contain a list of dictionaries")

        return self.reports

    def create_documents(self) -> List[Document]:
        """Convert financial reports to LangChain documents.
        Document content and metadata should contain:
        1. title
        2. publication_date 
        3. author
        4. category
        5. summary
        6. key_statistics
        7. sentiment_score
        TODO: Implement this method
        """
        if not self.reports:
            self.load_reports()
        documents = []
        for report in self.reports:
            content = report.get("summary", "")
            metadata = {
                "title": report.get("title"),
                "publication_date": report.get("publication_date"),
                "author": report.get("author"),
                "category": report.get("category"),
                "summary": report.get("summary"),
                "key_statistics": report.get("key_statistics"),
                "sentiment_score": report.get("sentiment_score")
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents
