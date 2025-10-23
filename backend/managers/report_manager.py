import os
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai



class ReportManager:
    """Generates narrative summaries from a collection of retrieved documents."""

    def __init__(self, logger=None):
        """Initializes the ReportManager and configures the LLM."""
        self.logger = logger or logging.getLogger(__name__)
        self._configure_llm()

    def _configure_llm(self):
        """Configures the Google Generative AI model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.error("GOOGLE_API_KEY not found. Report generation is disabled.")
            self.llm = None
            return
        from core.gemini_utils import configure_llm_from_env
        self.llm = configure_llm_from_env()

    def generate_report(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generates a structured report based on a query and retrieved documents.

        Args:
            query: The original user query.
            retrieved_docs: A list of documents returned from the retriever.

        Returns:
            A dictionary containing the summary and citations, or None if generation fails.
        """
        if not self.llm:
            return {"summary": "Report generation is disabled. No API key found.", "citations": []}

        if not retrieved_docs:
            return {"summary": "No relevant documents were found to generate a report.", "citations": []}

        # Prepare the context and citations for the LLM
        context_str = ""
        citations = []
        for i, doc in enumerate(retrieved_docs):
            context_str += f"--- Document {i+1} (ID: {doc['id']}) ---\n"
            context_str += f"Username: {doc['metadata']['username']}\n"
            context_str += f"URL: {doc['metadata']['url']}\n"
            context_str += f"Content: {doc['snippet']}\n\n"
            citations.append({
                "document_id": doc['id'],
                "url": doc['metadata']['url'],
                "username": doc['metadata']['username']
            })

        prompt = f"""
        As a research analyst, generate a structured narrative summary based on the following documents, which were retrieved for the query: '{query}'.

        Your summary should synthesize the key themes, insights, and differing viewpoints from the documents provided. Structure the output with clear headings.

        **Crucially, you must cite the documents using their ID (e.g., [Doc 1], [Doc 2]) whenever you reference their content.**

        **Documents:**
        {context_str}

        **Report:**
        """

        try:
            self.logger.info(f"Generating report for query: '{query}'")
            response = self.llm.generate_content(prompt)
            summary = response.text.strip()

            return {
                "summary": summary,
                "citations": citations
            }
        except Exception as e:
            self.logger.error(f"LLM report generation failed: {e}")
            return {
                "summary": "Report generation temporarily unavailable (LLM error).",
                "citations": []
            }
