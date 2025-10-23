import logging
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Search:
    """Encapsulates the BM25 keyword search functionality."""

    def __init__(self, corpus_data: Dict[str, str], logger=None):
        """
        Initializes and fits the BM25 model.

        Args:
            corpus_data: A dictionary mapping document IDs to their text content.
        """
        self.logger = logger or logging.getLogger(__name__)
        if not corpus_data:
            raise ValueError("BM25Search requires a non-empty corpus_data.")

        self.doc_ids = list(corpus_data.keys())
        self.corpus = list(corpus_data.values())
        
        logger.info(f"Tokenizing {len(self.corpus)} documents for BM25...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 model initialized.")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a keyword search against the corpus.

        Args:
            query: The search query string.
            limit: The maximum number of results to return.

        Returns:
            A list of dictionaries, each containing the document ID and BM25 score.
        """
        if not hasattr(self, 'bm25'):
            logger.warning("BM25 model not initialized. Returning empty list.")
            return []

        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get the top N scores and their indices
        top_n_indices = doc_scores.argsort()[::-1][:limit]

        results = [
            {
                "id": self.doc_ids[i],
                "score": doc_scores[i]
            }
            for i in top_n_indices if doc_scores[i] > 0
        ]

        return results
