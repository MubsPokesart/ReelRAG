# ReelRAG
Python endpoint for processing Instagram Reels transcription, fed into a vector database, to do similarity search

## Implementation Explanation and Rationale

### Why This Architecture?

1. **Modular Design**: Each component has a single responsibility, making the system maintainable and testable.

2. **Asynchronous Processing**: Uses FastAPI's async capabilities and background tasks for non-blocking operations.

3. **Error Handling**: Comprehensive error handling at each layer prevents cascading failures.

4. **Scalability**: Can easily scale by:
   - Adding more workers
   - Using message queues (Redis/RabbitMQ) for processing
   - Implementing caching layers

### Step-by-Step Process Flow:

1. **URL Validation** (Necessary): Prevents processing invalid URLs and wasting resources.

2. **Instagram Scraping** (Necessary): Retrieves video and metadata. Instaloader is reliable and well-maintained.

3. **Audio Extraction** (Necessary): Whisper requires audio input, not video. This reduces file size by ~80%.

4. **Transcription** (Necessary): Core functionality for searchability.

5. **Text Refinement** (Necessary): Improves search quality by removing artifacts and normalizing text.

6. **Embedding Generation** (Necessary): Enables semantic search beyond keyword matching.

7. **Vector Storage** (Necessary): ChromaDB provides persistent storage with built-in similarity search.

8. **Cleanup** (Necessary): Prevents disk space issues from temporary files.
