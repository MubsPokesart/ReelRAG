import instaloader
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import logging
from datetime import datetime

from config import config
from utils.validators import InstagramValidator

logger = logging.getLogger(__name__)

class InstagramScraper:
    """
    Handles Instagram Reel downloading and metadata extraction
    
    Why: Separates scraping logic from other concerns, making it
    easier to switch scraping libraries if needed
    """
    
    def __init__(self):
        """Initialize Instaloader with optimized settings"""
        self.loader = instaloader.Instaloader(
            download_pictures=False,  # We only need video
            download_videos=True,
            download_comments=False,  # Not needed for transcription
            compress_json=False,
            save_metadata=True,
            quiet=True,
            user_agent='Mozilla/5.0'
        )
        self._setup_session()
    
    def _setup_session(self):
        """
        Setup Instagram session for authenticated requests
        
        Why: Authenticated requests have higher rate limits and
        can access more content
        """
        session_file = Path(config.INSTAGRAM_SESSION_FILE)
        
        if session_file.exists():
            try:
                self.loader.load_session_from_file(
                    config.INSTAGRAM_USERNAME,
                    session_file
                )
                logger.info("Loaded existing Instagram session")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")
                self._create_new_session()
        else:
            self._create_new_session()
    
    def _create_new_session(self):
        """Create new Instagram session if credentials available"""
        if config.INSTAGRAM_USERNAME and config.INSTAGRAM_PASSWORD:
            try:
                self.loader.login(
                    config.INSTAGRAM_USERNAME,
                    config.INSTAGRAM_PASSWORD
                )
                self.loader.save_session_to_file(config.INSTAGRAM_SESSION_FILE)
                logger.info("Created new Instagram session")
            except Exception as e:
                logger.error(f"Failed to login: {e}")
    
    def download_reel(self, url: str) -> Tuple[Dict[str, Any], Path]:
        """
        Download Instagram Reel and extract metadata
        
        Args:
            url: Instagram Reel URL
            
        Returns:
            Tuple of (metadata dict, video file path)
            
        Why: Returns both metadata and file path for further processing
        """
        shortcode = InstagramValidator.extract_shortcode(url)
        if not shortcode:
            raise ValueError(f"Invalid Instagram URL: {url}")
        
        try:
            # Get post object
            post = instaloader.Post.from_shortcode(
                self.loader.context,
                shortcode
            )
            
            # Verify it's a video
            if not post.is_video:
                raise ValueError(f"URL is not a video/reel: {url}")
            
            # Extract metadata
            metadata = {
                'shortcode': post.shortcode,
                'url': InstagramValidator.normalize_url(url),
                'caption': post.caption or "",
                'hashtags': list(post.caption_hashtags) if post.caption else [],
                'views': post.video_view_count,
                'likes': post.likes,
                'comments': post.comments,
                'duration': post.video_duration,
                'date_posted': post.date.isoformat(),
                'owner_username': post.owner_username,
                'is_sponsored': post.is_sponsored
            }
            
            # Download video
            target_dir = config.TEMP_DOWNLOAD_DIR / shortcode
            target_dir.mkdir(exist_ok=True)
            
            self.loader.download_post(post, target=str(target_dir))
            
            # Find downloaded video file
            video_files = list(target_dir.glob("*.mp4"))
            if not video_files:
                raise FileNotFoundError(f"No video file found after download")
            
            video_path = video_files[0]
            logger.info(f"Successfully downloaded reel: {shortcode}")
            
            return metadata, video_path
            
        except Exception as e:
            logger.error(f"Failed to download reel {url}: {e}")
            raise
