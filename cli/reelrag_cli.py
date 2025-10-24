import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import json
import time
import random
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import typer
from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import LoginRequired

from backend.managers.ingestion_manager import IngestionManager
from backend.managers.preprocessing_manager import PreprocessingManager
from backend.managers.index_manager import IndexManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="ReelRAG: A CLI for ingesting, preprocessing, and indexing Instagram Reels.")

# --- Global Paths and Configuration ---
DATA_DIR = Path("data")
REELS_FILE = DATA_DIR / "reels.txt"
DATA_FILE = DATA_DIR / "output" / "data.json"
SETTINGS_FILE = DATA_DIR / "instagrapi_settings.json"
VIDEO_DIR = DATA_DIR / "reel_videos"
DB_DIR = DATA_DIR / "reels_db"

def _load_data() -> Dict[str, Any]:
    """Loads the central data JSON file."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning("data.json is corrupted, starting fresh.")
                return {}
    return {}

def _save_data(data: Dict[str, Any]):
    """Saves data to the central JSON file."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data saved to {DATA_FILE}")

def _get_insta_client() -> Client:
    """Initializes and logs in the Instagram client."""
    cl = Client()
    try:
        if SETTINGS_FILE.exists():
            cl.load_settings(SETTINGS_FILE)
            cl.get_timeline_feed() # Check if session is valid
            logger.info("Instagram session loaded and valid.")
        else:
            username = os.getenv('IG_USERNAME')
            password = os.getenv('IG_PASSWORD')
            if not username or not password:
                raise ValueError("IG_USERNAME and IG_PASSWORD not found in .env file.")
            cl.login(username, password)
            cl.dump_settings(SETTINGS_FILE)
            logger.info("Logged into Instagram and saved session.")
    except LoginRequired:
        logger.error("Session expired. Re-logging in.")
        username = os.getenv('IG_USERNAME')
        password = os.getenv('IG_PASSWORD')
        if not username or not password:
            raise ValueError("IG_USERNAME and IG_PASSWORD not found in .env file.")
        cl.login(username, password)
        cl.dump_settings(SETTINGS_FILE)
        logger.info("Re-logged into Instagram and saved session.")
    return cl

@app.command()
def ingest(reels_file: Path = REELS_FILE):
    """Ingests reel URLs, fetches metadata, and downloads videos."""
    load_dotenv()
    if not reels_file.exists():
        logger.error(f"Reels file not found at: {reels_file}")
        raise typer.Exit(code=1)

    with open(reels_file, 'r') as f:
        reel_urls = [line.strip() for line in f.read().splitlines() if "instagram.com" in line]

    logger.info(f"Found {len(reel_urls)} reels to process.")
    data = _load_data()
    
    try:
        client = _get_insta_client()
        ingestion_manager = IngestionManager(client, cache_dir=str(VIDEO_DIR))
    except Exception as e:
        logger.error(f"Failed to initialize Instagram client: {e}")
        raise typer.Exit(code=1)

    processed_count = 0
    for i, url in enumerate(reel_urls):
        media_pk = str(client.media_pk_from_url(url))
        if media_pk in data:
            logger.info(f"[{i+1}/{len(reel_urls)}] Reel {media_pk} metadata already exists. Skipping metadata fetch.")
        else:
            metadata = ingestion_manager.get_reel_metadata(url)
            if not metadata:
                continue
            data[metadata["media_pk"]] = metadata
        
        # Always attempt download in case it failed before
        ingestion_manager.download_reel(url)

        processed_count += 1
        if processed_count % 10 == 0:
            _save_data(data)
        
        time.sleep(random.uniform(2, 5)) # Throttle requests

    _save_data(data)
    logger.info("Ingestion complete.")

@app.command()
def preprocess():
    """Extracts audio and transcribes downloaded reels."""
    data = _load_data()
    if not data:
        logger.error("No data found. Run the 'ingest' command first.")
        raise typer.Exit(code=1)

    preprocessing_manager = PreprocessingManager(whisper_model="base")
    
    processed_count = 0
    for i, (media_pk, reel_data) in enumerate(data.items()):
        if reel_data.get("transcription"): # Skip if already transcribed
            logger.info(f"[{i+1}/{len(data)}] Reel {media_pk} already transcribed. Skipping.")
            continue

        url = reel_data['url']
        video_path = VIDEO_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.mp4"

        if not video_path.exists():
            logger.warning(f"Video for reel {media_pk} not found at {video_path}. Skipping.")
            continue

        logger.info(f"[{i+1}/{len(data)}] Preprocessing reel: {media_pk}")
        transcription = preprocessing_manager.transcribe_reel(video_path)

        if transcription is not None:
            data[media_pk]["transcription"] = transcription
            processed_count += 1

        if processed_count % 10 == 0:
            _save_data(data)

    _save_data(data)
    logger.info("Preprocessing complete.")

@app.command()
def index():
    """Embeds, clusters, and indexes reel content into ChromaDB."""
    load_dotenv() # For the GOOGLE_API_KEY
    data = _load_data()
    if not data:
        logger.error("No data found. Run 'ingest' and 'preprocess' first.")
        raise typer.Exit(code=1)

    index_manager = IndexManager(chroma_path=str(DB_DIR))
    index_manager.index_reels(data, use_clustering=True)
    logger.info("Indexing complete.")

if __name__ == "__main__":
    # Make sure the data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "output").mkdir(exist_ok=True)
    app()
