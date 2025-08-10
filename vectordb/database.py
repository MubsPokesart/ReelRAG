"""
Vector database management using ChromaDB.
Handles storage, retrieval, and similarity search.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import hashlib

from config import config
from vectordb.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Manages vector database operations
    
    Why: ChromaDB provides persistent storage with built-in
    similarity search and metadata filtering
    """
    
    def __init__(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedding_generator = EmbeddingGenerator()
        self.collection_name = "instagram_reels"
        self._initialize_collection()
    
    def _initialize_collection(self):
        """
        Initialize or get existing collection
        
        Why: Ensures collection exists with proper configuration
        """
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Instagram Reels transcriptions and metadata"}
            )
            logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_reel(self, reel_data: Dict[str, Any]) -> str:
        """
        Add a reel to the vector database
        
        Args:
            reel_data: Dictionary containing reel metadata and transcription
            
        Returns:
            Document ID
            
        Why: Stores complete reel data with embedding for search
        """
        # Generate unique ID based on URL
        doc_id = hashlib.md5(reel_data['url'].encode()).hexdigest()
        
        # Check if already exists
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            logger.info(f"Reel already exists: {doc_id}")
            return doc_id
        
        # Prepare text for embedding
        embedding_text = self._prepare_embedding_text(reel_data)
        
        # Generate embedding
        embedding = self.embedding_generator.create_combined_embedding(
            transcription=reel_data['transcription']['refined_text'],
            caption=reel_data['metadata'].get('caption', '')
        )
        
        # Prepare metadata (ChromaDB requires simple types)
        metadata = {
            'url': reel_data['url'],
            'shortcode': reel_data['metadata']['shortcode'],
            'caption': reel_data['metadata'].get('caption', ''),
            'hashtags': json.dumps(reel_data['metadata'].get('hashtags', [])),
            'views': reel_data['metadata'].get('views', 0),
            'likes': reel_data['metadata'].get('likes', 0),
            'duration': reel_data['metadata'].get('duration', 0),
            'language': reel_data['transcription']['language'],
            'date_posted': reel_data['metadata']['date_posted'],
            'processed_at': datetime.now().isoformat()
        }
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[reel_data['transcription']['refined_text']],
            metadatas=[metadata]
        )
        
        logger.info(f"Added reel to database: {doc_id}")
        return doc_id
    
    def _prepare_embedding_text(self, reel_data: Dict[str, Any]) -> str:
        """
        Prepare text for embedding generation
        
        Why: Combines relevant fields for comprehensive semantic representation
        """
        parts = []
        
        # Add transcription (most important)
        parts.append(reel_data['transcription']['refined_text'])
        
        # Add caption if available
        if reel_data['metadata'].get('caption'):
            parts.append(reel_data['metadata']['caption'])
        
        # Add hashtags
        hashtags = reel_data['metadata'].get('hashtags', [])
        if hashtags:
            parts.append(' '.join(hashtags))
        
        return ' '.join(parts)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar reels
        
        Args:
            query: Search query text
            limit: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with metadata
            
        Why: Provides semantic search with optional filtering
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, config.MAX_SEARCH_RESULTS),
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'url': results['metadatas'][0][i]['url'],
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'transcription_excerpt': results['documents'][0][i][:200] + "...",
                'metadata': {
                    'caption': results['metadatas'][0][i].get('caption', ''),
                    'hashtags': json.loads(results['metadatas'][0][i].get('hashtags', '[]')),
                    'views': results['metadatas'][0][i].get('views', 0),
                    'likes': results['metadatas'][0][i].get('likes', 0),
                    'duration': results['metadatas'][0][i].get('duration', 0),
                    'date_posted': results['metadatas'][0][i].get('date_posted', '')
                }
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def generate_synopsis(self, transcription: str, max_length: int = 150) -> str:
        """
        Generate AI synopsis of transcription
        
        Why: Provides quick summary for search results
        Note: This is a simple implementation. Consider using
        GPT or other LLMs for better summaries
        """
        # Simple extractive summary (first N characters)
        # In production, use proper summarization model
        sentences = transcription.split('. ')
        synopsis = ""
        
        for sentence in sentences:
            if len(synopsis) + len(sentence) <= max_length:
                synopsis += sentence + ". "
            else:
                break
        
        return synopsis.strip() or transcription[:max_length] + "..."
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Why: Useful for monitoring and debugging
        """
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_generator.model_type
        }