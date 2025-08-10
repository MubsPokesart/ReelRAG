"""
Transcription processing using OpenAI Whisper.
Handles speech-to-text conversion and text refinement.
"""
import whisper
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from config import config

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """
    Processes audio files to generate transcriptions
    
    Why: Centralizes transcription logic and text processing
    """
    
    def __init__(self, model_size: str = None):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size or config.WHISPER_MODEL
        logger.info(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)
        
    def transcribe(self, audio_path: Path) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
            
        Why: Returns comprehensive results including confidence
        and language detection for quality assessment
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_path),
                language=None if config.WHISPER_LANGUAGE == "auto" else config.WHISPER_LANGUAGE,
                verbose=False
            )
            
            # Extract segments with timestamps if needed
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            # Refine the transcription
            refined_text = self._refine_transcription(result["text"])
            
            return {
                "raw_text": result["text"],
                "refined_text": refined_text,
                "language": result.get("language", "unknown"),
                "segments": segments,
                "duration": result.get("duration", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            raise
    
    def _refine_transcription(self, text: str) -> str:
        """
        Clean and refine transcription text
        
        Why: Removes common transcription artifacts and improves
        text quality for better embedding and search
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove filler words (customize based on your content)
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically']
        for filler in filler_words:
            # Case-insensitive replacement at word boundaries
            pattern = r'\b' + filler + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Fix common transcription errors
        replacements = {
            ' i ': ' I ',
            " i'm ": " I'm ",
            " i'll ": " I'll ",
            " i'd ": " I'd ",
            " i've ": " I've ",
            '  ': ' ',  # Remove double spaces
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Capitalize first letter of sentences
        text = '. '.join(s.strip().capitalize() for s in text.split('. ') if s)
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    @staticmethod
    def cleanup_audio(audio_path: Path):
        """
        Remove audio file after transcription
        
        Why: Saves disk space after processing
        """
        try:
            if audio_path.exists():
                audio_path.unlink()
                logger.info(f"Cleaned up audio: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio {audio_path}: {e}")