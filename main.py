"""
Main FastAPI application entry point.
Configures and runs the Instagram Reels API server.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from config import config
from api.endpoints import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    
    Why: Ensures proper startup and shutdown procedures
    """
    # Startup
    logger.info("Starting Instagram Reels API")
    config.create_directories()
    yield
    # Shutdown
    logger.info("Shutting down Instagram Reels API")

# Create FastAPI app
app = FastAPI(
    title="Instagram Reels Transcription API",
    description="API for processing Instagram Reels with AI transcription and semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix=config.API_PREFIX)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Instagram Reels Transcription API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    
    Why: Provides consistent error responses
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True  # Set to False in production
    )