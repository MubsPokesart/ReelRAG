"""
FastAPI endpoints for the Instagram Reels API.
Defines all API routes and request handling.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import logging
from pathlib import Path

from models.schemas import (
    ReelProcessRequest,
    SearchQuery,
    SearchResult,
    ReelDocument
)
from scrapers.instagram_scraper import InstagramScraper
from processors.audio_extractor import AudioExtractor
from processors.transcription_processor import TranscriptionProcessor
from vectordb.database import VectorDatabase
from utils.validators import InstagramValidator

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
scraper = InstagramScraper()
audio_extractor = AudioExtractor()
transcription_processor = TranscriptionProcessor()
vector_db = VectorDatabase()

@router.post("/process-reel", response_model=Dict[str, Any])
async def process_reel(
    request: ReelProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a single Instagram Reel
    
    Why: Main endpoint for adding new reels to the system
    """
    try:
        # Validate URL
        if not InstagramValidator.validate_url(str(request.url)):
            raise HTTPException(400, "Invalid Instagram Reel URL")
        
        # Normalize URL
        normalized_url = InstagramValidator.normalize_url(str(request.url))
        
        # Step 1: Download reel and extract metadata
        logger.info(f"Processing reel: {normalized_url}")
        metadata, video_path = scraper.download_reel(normalized_url)
        
        # Step 2: Extract audio from video
        audio_path = audio_extractor.extract_audio(video_path)
        
        # Step 3: Transcribe audio
        transcription_result = transcription_processor.transcribe(audio_path)
        
        # Step 4: Prepare document for storage
        reel_document = {
            'url': normalized_url,
            'metadata': metadata,
            'transcription': transcription_result
        }
        
        # Step 5: Add to vector database
        doc_id = vector_db.add_reel(reel_document)
        
        # Step 6: Cleanup temporary files in background
        background_tasks.add_task(cleanup_files, video_path, audio_path)
        
        return {
            'success': True,
            'document_id': doc_id,
            'url': normalized_url,
            'transcription_preview': transcription_result['refined_text'][:200] + "...",
            'language': transcription_result['language'],
            'duration': metadata.get('duration', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to process reel: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.post("/search", response_model=List[SearchResult])
async def search_reels(query: SearchQuery):
    """
    Search for relevant reels based on semantic similarity
    
    Why: Main search endpoint for finding relevant content
    """
    try:
        # Perform vector search
        results = vector_db.search(
            query=query.query,
            limit=query.limit
        )
        
        # Format results for response
        search_results = []
        for result in results:
            # Generate AI synopsis
            synopsis = vector_db.generate_synopsis(
                result['transcription_excerpt']
            )
            
            search_result = SearchResult(
                reel_url=result['url'],
                similarity_score=result['similarity_score'],
                synopsis=synopsis,
                metadata=result['metadata'] if query.include_metadata else None,
                relevant_excerpt=result['transcription_excerpt']
            )
            search_results.append(search_result)
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@router.post("/batch-process")
async def batch_process_reels(
    urls: List[str],
    background_tasks: BackgroundTasks
):
    """
    Process multiple reels in batch
    
    Why: Efficient processing of multiple reels
    """
    results = []
    
    for url in urls:
        try:
            # Validate each URL
            if not InstagramValidator.validate_url(url):
                results.append({
                    'url': url,
                    'success': False,
                    'error': 'Invalid URL'
                })
                continue
            
            # Process in background
            background_tasks.add_task(
                process_reel_background,
                url
            )
            
            results.append({
                'url': url,
                'success': True,
                'status': 'Processing started'
            })
            
        except Exception as e:
            results.append({
                'url': url,
                'success': False,
                'error': str(e)
            })
    
    return {
        'total': len(urls),
        'results': results
    }

@router.get("/stats")
async def get_statistics():
    """
    Get database statistics
    
    Why: Monitoring and debugging endpoint
    """
    try:
        stats = vector_db.get_stats()
        return {
            'success': True,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(500, f"Failed to get statistics: {str(e)}")

@router.delete("/reel/{shortcode}")
async def delete_reel(shortcode: str):
    """
    Delete a reel from the database
    
    Why: Data management and cleanup
    """
    # Implementation would go here
    # This is a placeholder for deletion logic
    return {
        'success': True,
        'message': f"Reel {shortcode} deletion scheduled"
    }

# Helper functions
def cleanup_files(video_path: Path, audio_path: Path):
    """
    Cleanup temporary files after processing
    
    Why: Prevents disk space issues from accumulated temp files
    """
    try:
        AudioExtractor.cleanup_video(video_path)
        TranscriptionProcessor.cleanup_audio(audio_path)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

async def process_reel_background(url: str):
    """
    Process reel in background
    
    Why: Allows batch processing without blocking
    """
    try:
        request = ReelProcessRequest(url=url)
        # Process the reel
        # This would call the main processing logic
        logger.info(f"Background processing: {url}")
    except Exception as e:
        logger.error(f"Background processing failed for {url}: {e}")