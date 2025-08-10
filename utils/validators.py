"""
Validation utilities for Instagram URLs and other inputs.
Ensures data integrity before processing.
"""
import re
from typing import Optional

class InstagramValidator:
    """Validates and extracts information from Instagram URLs"""
    
    REEL_PATTERN = r'https?://(?:www\.)?instagram\.com/(?:p|reel|reels)/([A-Za-z0-9_-]{11})/?'
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """
        Validate if URL is a valid Instagram Reel/Post URL
        
        Why: Prevents processing invalid URLs early in the pipeline
        """
        pattern = re.compile(cls.REEL_PATTERN)
        return bool(pattern.match(url))
    
    @classmethod
    def extract_shortcode(cls, url: str) -> Optional[str]:
        """
        Extract the shortcode (post ID) from Instagram URL
        
        Why: Shortcode is the unique identifier needed for API calls
        """
        pattern = re.compile(cls.REEL_PATTERN)
        match = pattern.match(url)
        return match.group(1) if match else None
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """
        Normalize Instagram URL to standard format
        
        Why: Ensures consistent storage and prevents duplicates
        """
        shortcode = cls.extract_shortcode(url)
        if shortcode:
            return f"https://www.instagram.com/reel/{shortcode}/"
        return url