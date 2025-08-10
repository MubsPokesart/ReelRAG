import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    
    # Instagram Settings
    INSTAGRAM_SESSION_FILE = os.getenv("INSTAGRAM_SESSION_FILE", "session.json")
    INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME", "")
    INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "")
    
    # File Storage
    BASE_DIR = Path(__file__).parent
    TEMP_DOWNLOAD_DIR = BASE_DIR / "temp_downloads"
    AUDIO_OUTPUT_DIR = BASE_DIR / "audio_files"
    
    # Whisper Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")  # auto-detect or specify
    
    # Vector Database Settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers")  # or "openai"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Search Settings
    DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", 5))
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 20))
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.TEMP_DOWNLOAD_DIR.mkdir(exist_ok=True)
        cls.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()
config.create_directories()