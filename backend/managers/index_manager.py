import os
import logging
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import hdbscan
import google.generativeai as genai

# Define a base path relative to this file's location
BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexManager:
    """Manages embedding, clustering, and indexing of reel data into ChromaDB."""

    def __init__(self, chroma_path: str = None, collection_name: str = "reels"):
        """
        Initializes the IndexManager.

        Args:
            chroma_path: Path to the ChromaDB persistence directory.
            collection_name: Name of the collection within ChromaDB.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        db_path = Path(chroma_path) if chroma_path else BASE_DATA_DIR / "reels_db"
        db_path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(collection_name)
        self._configure_llm()

    def _configure_llm(self):
        """Configures the Google Generative AI model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Topic labeling will be disabled.")
            self.llm = None
            return
        from core.gemini_utils import configure_llm_from_env
        self.llm = configure_llm_from_env()

    def _generate_topic_tags(self, text: str) -> List[str]:
        """Generates simple keyword-based topic tags as a fallback."""
        # This is the simple placeholder from the original script
        topics = []
        text_lower = text.lower()
        topic_map = {
            'career': ['job', 'internship', 'career', 'networking', 'resume'],
            'fitness': ['fitness', 'diet', 'workout', 'protein', 'gym'],
            'relationships': ['relationship', 'dating', 'marriage', 'friendship'],
            'tech': ['ai', 'tech', 'software', 'coding', 'data', 'python'],
            'finance': ['money', 'finance', 'invest', 'budget', 'wealth'],
        }
        for topic, keywords in topic_map.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        return list(set(topics)) if topics else ['general']

    def index_reels(self, data: Dict[str, Dict[str, Any]], use_clustering=True):
        """
        Generates embeddings and indexes reels into ChromaDB.

        Args:
            data: A dictionary of processed reel data.
            use_clustering: Whether to perform advanced clustering-based topic labeling.
        """
        ids, documents, metadatas = [], [], []

        for reel_id, reel_data in data.items():
            if not reel_data or not isinstance(reel_data, dict) or not reel_data.get('transcription'):
                logger.warning(f"Skipping invalid or incomplete reel data for id: {reel_id}")
                continue

            text_blob = f"Caption: {reel_data.get('caption', '')}. Transcription: {reel_data.get('transcription', '')}"
            
            documents.append(text_blob)
            ids.append(str(reel_id))
            metadatas.append({
                "url": reel_data.get("url", ""),
                "username": reel_data.get("username", ""),
                "post_date": reel_data.get("post_date", ""),
                "topic_tags": ", ".join(self._generate_topic_tags(text_blob))
            })

        if not documents:
            logger.info("No new documents to index.")
            return

        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, show_progress_bar=True)

        if use_clustering and len(documents) > 10: # Clustering needs a minimum number of samples
            logger.info("Performing topic clustering...")
            cluster_labels = self._cluster_and_label_topics(documents, embeddings)
            for i, label in enumerate(cluster_labels):
                # Append the generated cluster label to the existing tags
                existing_tags = metadatas[i]["topic_tags"].split(", ")
                if label not in existing_tags:
                    existing_tags.append(label)
                metadatas[i]["topic_tags"] = ", ".join(filter(None, existing_tags))

        logger.info(f"Adding {len(documents)} documents to ChromaDB...")
        # Upsert is safer as it handles duplicates
        self.collection.upsert(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully indexed {len(documents)} reels.")

    def _cluster_and_label_topics(self, docs: List[str], embeddings: np.ndarray) -> List[str]:
        """Performs clustering and uses an LLM to generate human-readable labels."""
        # 1. Dimensionality Reduction with UMAP
        reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        # 2. Clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
        cluster_ids = clusterer.fit_predict(reduced_embeddings)

        df = pd.DataFrame({'doc': docs, 'cluster': cluster_ids})
        
        # 3. TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectorizer.fit(docs)
        
        topic_labels = {}
        for i in np.unique(cluster_ids):
            if i == -1: continue # Skip noise points
            cluster_docs = df[df['cluster'] == i]['doc'].tolist()
            tfidf_matrix = vectorizer.transform(cluster_docs)
            feature_names = vectorizer.get_feature_names_out()
            # Get top 5 keywords for the cluster
            top_keywords = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1][:5].tolist()[0]]
            topic_labels[i] = self._get_llm_label(top_keywords)

        # Map cluster labels back to each document
        df['topic_label'] = df['cluster'].map(topic_labels).fillna('miscellaneous')
        return df['topic_label'].tolist()


    def _generate_content_with_retry(self, prompt: str) -> str:
        """Generates content with a retry mechanism for API version errors."""
        try:
            response = self.llm.generate_content(prompt) if self.llm else None
            return (response.text if response else "unlabeled").strip().replace("\n","").replace("**","")
        except Exception as e:
            logger.error(f"LLM topic labeling failed: {e}")
            return "unlabeled"

    def _get_llm_label(self, keywords: List[str]) -> str:
        """Uses an LLM to generate a concise topic label from keywords."""
        if not self.llm:
            return ", ".join(keywords)
        
        prompt = f"Generate a single, concise, human-readable topic label (2-3 words max) for a cluster of documents characterized by these keywords: {keywords}. Example: 'Career Advice', 'Fitness Hacks', 'Tech Startups'. Label:"
        
        label = self._generate_content_with_retry(prompt)
        
        if label != "unlabeled":
            logger.info(f"Generated label '{label}' for keywords: {keywords}")
        
        return label

