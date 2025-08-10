"""
Embedding generation for vector similarity search.
Supports multiple embedding models for flexibility.
"""
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from config import config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text using various models
    
    Why: Abstracts embedding generation to easily switch between
    different models based on requirements
    """
    
    def __init__(self, model_type: str = None):
        """
        Initialize embedding model
        
        Args:
            model_type: Type of embedding model to use
        """
        self.model_type = model_type or config.EMBEDDING_MODEL
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate embedding model"""
        if self.model_type == "sentence-transformers":
            # Using all-MiniLM-L6-v2 for good balance of speed and quality
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized Sentence Transformer model")
            
        elif self.model_type == "openai":
            # OpenAI embeddings (requires API key)
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            import openai
            openai.api_key = config.OPENAI_API_KEY
            self.model = "text-embedding-3-small"  # or text-embedding-3-large
            logger.info("Initialized OpenAI embedding model")
            
        else:
            raise ValueError(f"Unknown embedding model type: {self.model_type}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
            
        Why: Single text embedding for simple use cases
        """
        return self.generate_embeddings([text])[0]
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
            
        Why: Batch processing is more efficient for multiple texts
        """
        if not texts:
            return []
        
        try:
            if self.model_type == "sentence-transformers":
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
                
            elif self.model_type == "openai":
                import openai
                response = openai.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def create_combined_embedding(
        self,
        transcription: str,
        caption: str,
        weight_transcription: float = 0.7
    ) -> List[float]:
        """
        Create weighted combination of transcription and caption embeddings
        
        Why: Combining both sources provides richer semantic representation
        Caption provides context, transcription provides detailed content
        """
        trans_embedding = np.array(self.generate_embedding(transcription))
        caption_embedding = np.array(self.generate_embedding(caption))
        
        # Weighted average
        combined = (trans_embedding * weight_transcription + 
                   caption_embedding * (1 - weight_transcription))
        
        # Normalize to unit length
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined.tolist()