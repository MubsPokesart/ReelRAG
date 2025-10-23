import os, logging
import google.generativeai as genai
logger = logging.getLogger(__name__)

CANDIDATES = [
    os.getenv("MODEL_PRIMARY", "gemini-2.5-pro"),
    os.getenv("MODEL_FALLBACK", "gemini-2.5-flash"),
    "gemini-pro-latest",
    "gemini-flash-latest"
]

def configure_llm_from_env():
    """Return a GenerativeModel that supports generateContent, or None."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set; disabling LLM features.")
        return None
    genai.configure(api_key=api_key)
    try:
        available = set()
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.add(m.name.split("/")[-1])
        for name in CANDIDATES:
            if name in available:
                logger.info(f"Using Gemini model: {name}")
                from google.generativeai import GenerativeModel
                return GenerativeModel(name)
        logger.error(f"No preferred Gemini models available. Seen: {sorted(available)}")
    except Exception as e:
        logger.error(f"Failed to list/select Gemini model: {e}")
    return None
