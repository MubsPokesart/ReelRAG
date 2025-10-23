import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict

import whisper
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingManager:
    """Handles audio extraction, transcription, and cleaning for reels."""

    FILLER_WORDS = {
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'basically',
        'actually', 'literally', 'sort of', 'kind of'
    }
    CORRECTIONS = {
        'gonna': 'going to', 'wanna': 'want to', 'gotta': 'got to', 'shoulda': 'should have',
        'coulda': 'could have', 'woulda': 'would have', 'lemme': 'let me', 'gimme': 'give me',
        'dunno': "don't know", 'kinda': 'kind of', 'sorta': 'sort of'
    }

    def __init__(self, whisper_model: str = "base"):
        """
        Initializes the PreprocessingManager.

        Args:
            whisper_model: The name of the Whisper model to use (e.g., 'base', 'small').
        """
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.model = whisper.load_model(whisper_model)
        self.temp_dir = Path(tempfile.gettempdir()) / "reel_audio_cache"
        self.temp_dir.mkdir(exist_ok=True)

    def _clean_transcription(self, text: str) -> str:
        """Cleans raw transcription text by removing fillers and correcting slang."""
        if not text: return ""
        text_lower = text.lower()
        # Remove punctuation before checking for filler words
        words = text_lower.split()
        words_filtered = [word for word in words if re.sub(r'[^\w\s]', '', word) not in self.FILLER_WORDS]
        text_clean = ' '.join(words_filtered)

        # Apply corrections
        for mistake, correction in self.CORRECTIONS.items():
            text_clean = re.sub(r'\b' + mistake + r'\b', correction, text_clean)
        
        # Basic punctuation and capitalization improvements
        text_clean = re.sub(r'\s+([,.!?;])', r'\1', text_clean).strip()
        if text_clean and not any(text_clean.endswith(p) for p in ['.', '!', '?']):
            text_clean += '.'
        
        # Capitalize the first letter of the cleaned text
        if text_clean:
            text_clean = text_clean[0].upper() + text_clean[1:]

        return text_clean

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extracts audio from a video file and saves it as an MP3."""
        try:
            audio_path = self.temp_dir / f"{video_path.stem}.mp3"
            if audio_path.exists():
                logger.info(f"Using cached audio: {audio_path}")
                return audio_path

            logger.info(f"Extracting audio from: {video_path}")
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    logger.warning(f"No audio track in video: {video_path}")
                    return None
                video.audio.write_audiofile(str(audio_path), logger=None, verbose=False)
            return audio_path
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path}: {e}")
            return None

    def transcribe_reel(self, video_path: Path) -> Optional[str]:
        """
        Processes a single reel video to get a cleaned transcription.

        Args:
            video_path: The path to the reel video file.

        Returns:
            The cleaned transcription text, or None if a step fails.
        """
        audio_path = self._extract_audio(video_path)
        if not audio_path:
            return None

        try:
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.model.transcribe(str(audio_path), fp16=False, language="en")
            raw_text = result.get("text", "").strip()

            if not raw_text:
                logger.warning(f"Empty transcription for: {audio_path}")
                return ""

            cleaned_text = self._clean_transcription(raw_text)
            logger.info(f"Successfully transcribed {video_path.name}. Length: {len(cleaned_text)}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            return None
