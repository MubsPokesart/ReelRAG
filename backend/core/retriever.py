import os
import logging
import numpy as np
from typing import List, Dict, Any
import ast

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from .bm25_search import BM25Search
from .rocchio import rocchio_refine

# Define a base path relative to this file's location
from pathlib import Path
BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"



class Retriever:
    """Orchestrates hybrid search, including semantic, keyword, and query refinement."""

    def __init__(self, db_path: str = None, collection_name: str = "reels", logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        db_path = Path(db_path) if db_path else BASE_DATA_DIR / "reels_db"
        db_path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_collection(collection_name)
        self._configure_llm()
        self._init_bm25()

    def _configure_llm(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("GOOGLE_API_KEY not found. Paraphrasing will be disabled.")
            self.llm = None
            return
        from .gemini_utils import configure_llm_from_env
        self.llm = configure_llm_from_env()

    def _init_bm25(self):
        """Initializes the BM25 search model with the corpus from ChromaDB."""
        try:
            corpus = self.collection.get() # Fetch all documents
            if not corpus or not corpus['ids']:
                self.logger.warning("ChromaDB collection is empty. BM25 search will be disabled.")
                self.bm25_search = None
                return
            
            corpus_data = {doc_id: doc for doc_id, doc in zip(corpus['ids'], corpus['documents'])}
            self.bm25_search = BM25Search(corpus_data)
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 model: {e}")
            self.bm25_search = None

    def _paraphrase_query(self, query: str, num_paraphrases: int = 3) -> List[str]:
        """Generates paraphrases of the user query using an LLM."""
        if not self.llm or num_paraphrases == 0:
            return [query]
        
        prompt = f"Generate {num_paraphrases} diverse paraphrases for the following search query, focusing on different intents and keywords. Return only a Python list of strings. Query: '{query}'"
        try:
            response = self.llm.generate_content(prompt)
            paraphrases = ast.literal_eval(response.text.strip())
            if isinstance(paraphrases, list):
                return [query] + paraphrases
        except Exception as e:
            self.logger.error(f"LLM paraphrasing failed: {e}")
        return [query] # Fallback to original query

    def search(self, query: str, k: int = 10, semantic_weight: float = 0.7, use_rocchio: bool = False, num_paraphrases: int = 3) -> List[Dict[str, Any]]:
        """Performs a hybrid search and returns a ranked list of reels."""
        # 1. Generate Query Vector (with optional paraphrasing)
        paraphrases = self._paraphrase_query(query, num_paraphrases=num_paraphrases)
        query_embeddings = self.model.encode(paraphrases)
        query_vector = np.mean(query_embeddings, axis=0).tolist()

        # 2. Semantic Search (ChromaDB)
        semantic_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k * 2 # Fetch more to allow for merging
        )

        # 3. Keyword Search (BM25)
        keyword_results = self.bm25_search.search(query, limit=k * 2) if self.bm25_search else []

        # 4. Combine and Rerank
        combined_scores = self._combine_and_normalize(semantic_results, keyword_results, semantic_weight)

        # 5. Rocchio Refinement (Optional)
        if use_rocchio and combined_scores:
            top_doc_ids = [doc['id'] for doc in combined_scores[:5]] # Use top 5 for refinement
            pos_doc_vectors = self.collection.get(ids=top_doc_ids, include=["embeddings"])['embeddings']
            
            refined_vector = rocchio_refine(np.array(query_vector), [np.array(v) for v in pos_doc_vectors])
            
            # Re-query with the refined vector
            self.logger.info("Performing second-pass search with Rocchio-refined vector.")
            refined_semantic_results = self.collection.query(query_embeddings=[refined_vector.tolist()], n_results=k)
            
            # Just use the refined semantic results for simplicity in this step
            final_results = self._format_results(refined_semantic_results)
            return final_results

        # 6. Format and Return
        final_doc_ids = [doc['id'] for doc in combined_scores[:k]]
        final_docs = self.collection.get(ids=final_doc_ids, include=["metadatas", "documents"])
        
        # Create a mapping from ID to score for easy lookup
        score_map = {doc['id']: doc['score'] for doc in combined_scores}
        
        # Build the final result list, ensuring order is preserved
        formatted_results = []
        for i, doc_id in enumerate(final_docs['ids']):
            snippet = final_docs['documents'][i][:250] + '...' # Create a snippet
            formatted_results.append({
                "id": doc_id,
                "score": score_map.get(doc_id, 0.0),
                "metadata": final_docs['metadatas'][i],
                "snippet": snippet
            })
        return formatted_results

    def _combine_and_normalize(self, semantic_results: Dict, keyword_results: List[Dict], weight: float) -> List[Dict[str, Any]]:
        """Normalizes scores from both search methods and combines them."""
        scores = {}

        # Process semantic results (distances -> scores)
        if semantic_results and semantic_results['ids'][0]:
            max_dist = max(semantic_results['distances'][0]) if semantic_results['distances'][0] else 1.0
            for doc_id, dist in zip(semantic_results['ids'][0], semantic_results['distances'][0]):
                scores[doc_id] = scores.get(doc_id, {'semantic': 0, 'keyword': 0})
                scores[doc_id]['semantic'] = 1 - (dist / max_dist) # Normalize distance to a score

        # Process keyword results
        if keyword_results:
            max_score = max(res['score'] for res in keyword_results) if keyword_results else 1.0
            for res in keyword_results:
                doc_id = res['id']
                scores[doc_id] = scores.get(doc_id, {'semantic': 0, 'keyword': 0})
                scores[doc_id]['keyword'] = res['score'] / max_score # Normalize score

        # Calculate final weighted score
        final_scores = [
            {'id': doc_id, 'score': (data['semantic'] * weight) + (data['keyword'] * (1 - weight))}
            for doc_id, data in scores.items()
        ]

        return sorted(final_scores, key=lambda x: x['score'], reverse=True)

    def _format_results(self, query_result: Dict) -> List[Dict[str, Any]]:
        """Formats ChromaDB query results into the final structure."""
        if not query_result or not query_result['ids'][0]:
            return []

        results = []
        for i, doc_id in enumerate(query_result['ids'][0]):
            distance = query_result['distances'][0][i]
            metadata = query_result['metadatas'][0][i]
            document = query_result['documents'][0][i]
            results.append({
                "id": doc_id,
                "score": 1 - distance, # Convert distance to similarity score
                "metadata": metadata,
                "snippet": document[:250] + '...'
            })
        return results
