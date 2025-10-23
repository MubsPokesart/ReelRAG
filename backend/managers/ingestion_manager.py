import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict

from instagrapi import Client
from instagrapi.exceptions import MediaNotFound

# Define a base path relative to this file's location
BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionManager:
    """Manages the downloading of Instagram reels and fetching their metadata."""

    def __init__(self, client: Client, cache_dir: str = None):
        """
        Initializes the IngestionManager.

        Args:
            client: An authenticated instagrapi.Client instance.
            cache_dir: The directory to store downloaded reel videos.
        """
        self.client = client
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = BASE_DATA_DIR / "reel_videos"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, url: str) -> Path:
        """Generates a stable, hashed filename for a given URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.mp4"

    def get_reel_metadata(self, reel_url: str) -> Optional[Dict]:
        """
        Fetches metadata for a single Instagram reel.

        Args:
            reel_url: The URL of the reel.

        Returns:
            A dictionary containing reel metadata or None if not found.
        """
        try:
            logger.info(f"Fetching metadata for {reel_url}...")
            media_pk = self.client.media_pk_from_url(reel_url)
            media_info = self.client.media_info(media_pk)

            return {
                "media_pk": str(media_pk),
                "url": reel_url,
                "caption": media_info.caption_text or "",
                "username": media_info.user.username,
                "post_date": media_info.taken_at.strftime("%Y-%m-%d"),
            }
        except MediaNotFound:
            logger.warning(f"Reel not found or private: {reel_url}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {reel_url}: {e}")
            return None

    def download_reel(self, reel_url: str) -> Optional[Path]:
        """
        Downloads a reel video if it's not already in the cache.

        Args:
            reel_url: The URL of the reel to download.

        Returns:
            The local path to the downloaded video, or None if download fails.
        """
        cache_path = self._get_cache_path(reel_url)
        if cache_path.exists():
            logger.info(f"Using cached reel: {cache_path}")
            return cache_path

        try:
            media_pk = self.client.media_pk_from_url(reel_url)
            logger.info(f"Downloading reel: {reel_url}")
            
            # instagrapi downloads to a path with the media_pk, we rename it
            downloaded_path = self.client.video_download(media_pk, folder=self.cache_dir)

            if downloaded_path and downloaded_path.exists():
                # Rename to our stable hashed filename
                downloaded_path.rename(cache_path)
                logger.info(f"Reel downloaded and cached as: {cache_path}")
                return cache_path
            else:
                logger.error(f"Download failed for {reel_url}, file not found after download call.")
                return None

        except MediaNotFound:
            logger.warning(f"Cannot download reel, it may be private or deleted: {reel_url}")
            return None
        except Exception as e:
            logger.error(f"An error occurred during reel download {reel_url}: {e}")
            return None
