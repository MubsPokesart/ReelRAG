"""
Pydantic models for request/response validation.
Defines the data structures used throughout the API.
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ReelProcessRequest(BaseModel):
    """Request model for processing a single Instagram Reel"""
    url: HttpUrl = Field(..., description="Instagram Reel URL")
    force_refresh: bool = Field(False, description="Force re-processing even if exists")
    
class ReelMetadata(BaseModel):
    """Metadata extracted from Instagram Reel"""
    shortcode: str
    url: str
    caption: Optional[str]
    hashtags: List[str]
    views: Optional[int]
    likes: Optional[int]
    comments: Optional[int]
    duration: Optional[float]
    date_posted: datetime
    
class TranscriptionResult(BaseModel):
    """Result of audio transcription"""
    raw_text: str
    refined_text: str
    language: str
    confidence: float
    duration_seconds: float
    
class ReelDocument(BaseModel):
    """Complete document stored in vector database"""
    id: str
    url: str
    metadata: ReelMetadata
    transcription: TranscriptionResult
    embedding_id: str
    processed_at: datetime
    
class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return")
    include_metadata: bool = Field(True, description="Include full metadata in results")
    
class SearchResult(BaseModel):
    """Individual search result"""
    reel_url: str
    similarity_score: float
    synopsis: str
    metadata: Optional[ReelMetadata]
    relevant_excerpt: str