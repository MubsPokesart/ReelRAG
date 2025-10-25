# AGENTS.md

## Purpose

This document serves as the authoritative guide for AI agents and developers working on the ReelRAG project. It defines the system architecture, coding standards, component responsibilities, and extension patterns that **MUST** be followed to maintain codebase quality and consistency.

---

## System Overview

ReelRAG is a full-stack retrieval-augmented generation system for analyzing Instagram Reels. The architecture consists of three distinct layers:

### 1. CLI Layer (Data Engineering)
**Location:** `/cli/reelrag_cli.py`

**Purpose:** Orchestrates the offline data pipeline from raw URLs to searchable embeddings.

**Commands:**
- `ingest`: Fetches reel metadata and downloads videos
- `preprocess`: Extracts audio and generates transcriptions
- `index`: Creates embeddings, clusters content, and persists to ChromaDB

### 2. Backend Layer (Flask API)
**Location:** `/backend/`

**Purpose:** Provides REST endpoints for hybrid retrieval, topic filtering, and report generation.

**Endpoints:**
- `POST /search`: Hybrid semantic/lexical search with optional Rocchio refinement
- `GET /topics`: Returns all unique topic tags from the corpus
- `POST /report`: Generates LLM-powered narrative summaries

### 3. Frontend Layer (React SPA)
**Location:** `/frontend/`

**Purpose:** User-facing interface for searching, filtering, and viewing results.

**Key Views:**
- `HomeView`: Main search interface with results display
- Component-based architecture with shared design system

---

## Architecture Principles

### Data Flow

```
User Query
    ↓
React Frontend (HomeView)
    ↓
SearchViewModel (useSearchViewModel.js)
    ↓
Flask Backend (/search endpoint)
    ↓
Retriever (core/retriever.py)
    ├→ Query Paraphrasing (Gemini)
    ├→ Hybrid Search (ChromaDB + BM25)
    ├→ Rocchio Refinement (optional)
    └→ Score Normalization & Reranking
    ↓
Formatted Results
    ↓
React UI (ResultCard components)
```

### Component Responsibilities

#### CLI Layer
- **IngestionManager**: Downloads and caches reel videos, handles Instagram API throttling
- **PreprocessingManager**: Extracts audio, transcribes with Whisper, cleans text
- **IndexManager**: Generates embeddings, performs clustering, assigns topic labels

#### Backend Core
- **Retriever**: Orchestrates the full retrieval pipeline (paraphrasing, hybrid search, refinement)
- **BM25Search**: Encapsulates keyword-based lexical search
- **RocchioRefinement**: Implements pseudo-relevance feedback for query refinement
- **GeminiUtils**: Centralized wrapper for all Gemini API calls

#### Backend Managers
- **ReportManager**: Generates narrative summaries from retrieved documents

#### Frontend
- **SearchViewModel**: Business logic for search and report generation
- **ResultCard**: Display component for individual reel results
- **SearchBar**: Input component with search and report generation triggers
- **ReportViewer**: Display component for generated reports

---

## Mandatory Programming Standards

These standards are **NON-NEGOTIABLE**. Every line of code must adhere to these principles. Violations will be rejected in code review.

### 1. File Length and Structure

**Rules:**
- **NEVER** allow a file to exceed **500 lines**
- If approaching **400 lines**, refactor immediately
- **1000+ lines is completely unacceptable**, even temporarily
- Use folders and naming conventions to group related small files

**Rationale:** Large files are difficult to test, review, and maintain. Small files force clear separation of concerns.

**Example Violation:**
```python
# DON'T: 800-line god class
class MassiveRetriever:
    def paraphrase(self): ...
    def embed(self): ...
    def search_chromadb(self): ...
    def search_bm25(self): ...
    def normalize_scores(self): ...
    def apply_rocchio(self): ...
    def format_results(self): ...
```

**Correct Approach:**
```python
# DO: Split into focused classes
# retriever.py (200 lines)
class Retriever:
    def __init__(self, paraphraser, semantic_search, lexical_search, refinement):
        self.paraphraser = paraphraser
        self.semantic_search = semantic_search
        self.lexical_search = lexical_search
        self.refinement = refinement
    
    def search(self, query, k, use_rocchio=False):
        # Orchestrates the pipeline
        ...

# paraphraser.py (100 lines)
class QueryParaphraser:
    def generate_paraphrases(self, query, n=3):
        ...

# score_normalizer.py (80 lines)
class ScoreNormalizer:
    def normalize_and_combine(self, semantic_scores, lexical_scores):
        ...
```

---

### 2. OOP-First Design

**Rules:**
- Every functionality must be encapsulated in a **dedicated class, struct, or protocol**
- Favor **composition over inheritance**
- Always design for **reuse**, not just "make it work"
- Avoid procedural scripts masquerading as modules

**Rationale:** Object-oriented design enables testability, dependency injection, and clear contracts.

**Example Violation:**
```python
# DON'T: Procedural spaghetti
def do_retrieval(query, k):
    paraphrases = call_gemini_for_paraphrases(query)
    embeddings = get_embeddings(paraphrases)
    chroma_results = query_chroma(embeddings)
    bm25_results = run_bm25(query)
    combined = combine_scores(chroma_results, bm25_results)
    return combined
```

**Correct Approach:**
```python
# DO: Class-based with clear responsibilities
class HybridRetriever:
    def __init__(self, paraphraser: QueryParaphraser, 
                 semantic_engine: SemanticSearch,
                 lexical_engine: BM25Search,
                 combiner: ScoreCombiner):
        self.paraphraser = paraphraser
        self.semantic_engine = semantic_engine
        self.lexical_engine = lexical_engine
        self.combiner = combiner
    
    def retrieve(self, query: str, k: int) -> List[ScoredDocument]:
        paraphrases = self.paraphraser.generate(query)
        semantic_scores = self.semantic_engine.search(paraphrases, k)
        lexical_scores = self.lexical_engine.search(query, k)
        return self.combiner.merge(semantic_scores, lexical_scores, k)
```

---

### 3. Single Responsibility Principle (SRP)

**Rules:**
- Every file, class, and function should do **ONE THING ONLY**
- If it has multiple responsibilities, **split it immediately**
- Each component should have a laser-focused concern

**Rationale:** SRP makes testing trivial, debugging straightforward, and changes localized.

**Example Violation:**
```python
# DON'T: Class doing too many things
class DataProcessor:
    def download_reel(self, url):
        ...
    def extract_audio(self, video_path):
        ...
    def transcribe(self, audio_path):
        ...
    def clean_text(self, transcript):
        ...
    def generate_embeddings(self, text):
        ...
    def cluster(self, embeddings):
        ...
```

**Correct Approach:**
```python
# DO: Separate classes per responsibility
class ReelDownloader:
    def download(self, url: str) -> Path:
        ...

class AudioExtractor:
    def extract(self, video_path: Path) -> Path:
        ...

class Transcriber:
    def transcribe(self, audio_path: Path) -> str:
        ...

class TextCleaner:
    def clean(self, raw_text: str) -> str:
        ...

class EmbeddingGenerator:
    def generate(self, text: str) -> np.ndarray:
        ...

class DocumentClusterer:
    def cluster(self, embeddings: np.ndarray) -> List[int]:
        ...
```

---

### 4. Modular Design

**Rules:**
- Code should connect like **Lego blocks** — interchangeable, testable, isolated
- Ask: "Can I reuse this class in a different project?" If not, refactor
- Reduce tight coupling; favor **dependency injection** and **protocols/interfaces**

**Rationale:** Modular code is portable, maintainable, and easier to reason about.

**Example Violation:**
```python
# DON'T: Tight coupling
class Retriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient("/hard/coded/path")
        self.gemini_key = "hard_coded_key_123"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

**Correct Approach:**
```python
# DO: Dependency injection
class Retriever:
    def __init__(self, 
                 vector_db: VectorDatabase,
                 llm_client: LLMClient,
                 embedding_model: EmbeddingModel,
                 logger: Logger):
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.logger = logger

# Now Retriever works with ANY implementation of these interfaces
```

---

### 5. Manager and Coordinator Patterns

**Rules:**
- Use **ViewModel**, **Manager**, and **Coordinator** naming conventions for logic separation:
  - **UI logic** → ViewModel
  - **Business logic** → Manager
  - **Navigation/state flow** → Coordinator
- **NEVER** mix views and business logic directly

**Rationale:** Clear separation of concerns makes code testable and prevents UI frameworks from infecting business logic.

**Current Implementation:**
- `IngestionManager`: Business logic for downloading reels
- `PreprocessingManager`: Business logic for audio extraction and transcription
- `IndexManager`: Business logic for embedding generation and clustering
- `ReportManager`: Business logic for LLM-powered report generation
- `useSearchViewModel`: Frontend viewmodel for search/report orchestration

**Example:**
```javascript
// DO: ViewModel handles UI state, delegates to API
const useSearchViewModel = () => {
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async (query) => {
        setLoading(true);
        const data = await apiClient.search(query); // Delegates to backend
        setResults(data);
        setLoading(false);
    };

    return { results, loading, handleSearch };
};
```

---

### 6. Function and Class Size

**Rules:**
- Keep functions under **30-40 lines**
- If a class exceeds **200 lines**, assess splitting into helper classes
- Long functions are a code smell indicating missing abstractions

**Example Violation:**
```python
# DON'T: 150-line function
def process_reel(url):
    # 20 lines of downloading
    # 30 lines of audio extraction
    # 40 lines of transcription
    # 30 lines of cleaning
    # 30 lines of embedding generation
    ...
```

**Correct Approach:**
```python
# DO: Compose small functions
def process_reel(url: str) -> ProcessedReel:
    video_path = download_reel(url)
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    clean_text = clean_transcript(transcript)
    embedding = generate_embedding(clean_text)
    return ProcessedReel(url, clean_text, embedding)
```

---

### 7. Naming and Readability

**Rules:**
- All names must be **descriptive and intention-revealing**
- Avoid vague names like: `data`, `info`, `helper`, `temp`, `handle`, `process`
- Use full words, not abbreviations (exception: widely-known acronyms like `URL`, `API`)

**Example Violation:**
```python
# DON'T
def proc(d):
    r = do_thing(d)
    return r
```

**Correct Approach:**
```python
# DO
def generate_embedding_from_text(clean_transcript: str) -> np.ndarray:
    normalized_text = self.text_normalizer.normalize(clean_transcript)
    embedding_vector = self.embedding_model.encode(normalized_text)
    return embedding_vector
```

---

### 8. Scalability Mindset

**Rules:**
- Always code as if **someone else will scale this**
- Include **extension points** (protocols, dependency injection) from day one
- Avoid hardcoding limits, paths, or assumptions

**Example:**
```python
# DO: Make scaling easy
class ReelProcessor:
    def __init__(self, 
                 batch_size: int = 10,
                 max_workers: int = 4,
                 cache_dir: Path = Path("./cache")):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_dir = cache_dir
    
    def process_batch(self, urls: List[str]) -> List[ProcessedReel]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_reel, url) for url in urls]
            return [f.result() for f in futures]
```

---

### 9. Avoid God Classes

**Rules:**
- **NEVER** let one file or class hold everything
- Split large classes into: UI, State, Handlers, Networking, etc.
- Prefer multiple small, focused classes over one massive class

**Example Violation:**
```python
# DON'T: God class
class ReelRAGSystem:
    def __init__(self):
        ...
    
    def download_reel(self, url): ...
    def extract_audio(self, video): ...
    def transcribe(self, audio): ...
    def embed(self, text): ...
    def cluster(self, embeddings): ...
    def search(self, query): ...
    def generate_report(self, results): ...
    def handle_ui_event(self, event): ...
```

**Correct Approach:**
- `ReelDownloader` (downloads)
- `AudioExtractor` (extracts)
- `Transcriber` (transcribes)
- `EmbeddingGenerator` (embeds)
- `Clusterer` (clusters)
- `Retriever` (searches)
- `ReportGenerator` (generates reports)
- `SearchViewModel` (handles UI events)

---

## Extension Patterns

### Adding a New Retrieval Strategy

**Scenario:** You want to add a re-ranking model to the retrieval pipeline.

**Steps:**
1. Create a new class: `/backend/core/reranker.py`
2. Implement a clean interface:
   ```python
   class Reranker:
       def rerank(self, query: str, candidates: List[Document], k: int) -> List[Document]:
           ...
   ```
3. Inject into `Retriever`:
   ```python
   class Retriever:
       def __init__(self, ..., reranker: Optional[Reranker] = None):
           self.reranker = reranker
       
       def search(self, query, k, use_reranking=False):
           results = self._hybrid_search(query, k)
           if use_reranking and self.reranker:
               results = self.reranker.rerank(query, results, k)
           return results
   ```

**Do NOT:**
- Add 200 lines of reranking logic directly into `Retriever`
- Hardcode model paths or API keys
- Break existing tests

---

### Adding a New Frontend Component

**Scenario:** You want to add a date range filter to the search interface.

**Steps:**
1. Create `/frontend/src/components/DateRangeFilter.jsx`
   ```jsx
   const DateRangeFilter = ({ startDate, endDate, onChange }) => {
       // Component logic
       return (
           <div className="date-range-filter">
               {/* UI elements */}
           </div>
       );
   };
   export default DateRangeFilter;
   ```
2. Add state management to `useSearchViewModel.js`:
   ```javascript
   const [dateRange, setDateRange] = useState({ start: null, end: null });
   ```
3. Update `/search` API call to include date range
4. Update backend `Retriever` to filter by date

**Do NOT:**
- Add 300 lines to `HomeView.jsx`
- Mix API logic with UI rendering
- Bypass the ViewModel layer

---

### Adding a New CLI Command

**Scenario:** You want to add a `validate` command to check data integrity.

**Steps:**
1. Create `/backend/managers/validation_manager.py`:
   ```python
   class ValidationManager:
       def __init__(self, data_dir: Path, logger: Logger):
           self.data_dir = data_dir
           self.logger = logger
       
       def validate(self) -> ValidationReport:
           ...
   ```
2. Add Typer command to `/cli/reelrag_cli.py`:
   ```python
   @app.command()
   def validate(data_dir: str = "./data"):
       """Validates data integrity of processed reels."""
       manager = ValidationManager(Path(data_dir), logger)
       report = manager.validate()
       typer.echo(report)
   ```

**Do NOT:**
- Add validation logic directly to `reelrag_cli.py`
- Create a 500-line validation script

---

## Testing Requirements

### Backend Tests
- **Location:** `/backend/tests/`
- **Framework:** pytest
- **Coverage:** Aim for 80%+ on core modules

**Example Test Structure:**
```python
# tests/test_bm25_search.py
import pytest
from backend.core.bm25_search import BM25Search

def test_bm25_returns_relevant_documents():
    corpus = {"doc1": "cat dog", "doc2": "bird fish"}
    searcher = BM25Search(corpus)
    results = searcher.search("cat", limit=1)
    assert results[0]["id"] == "doc1"

def test_bm25_handles_empty_query():
    corpus = {"doc1": "content"}
    searcher = BM25Search(corpus)
    results = searcher.search("", limit=10)
    assert len(results) == 0
```

### Frontend Tests
- **Location:** `/frontend/src/__tests__/`
- **Framework:** Jest + React Testing Library
- **Coverage:** All components and ViewModels

**Example Test:**
```javascript
// __tests__/SearchBar.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import SearchBar from '../components/SearchBar';

test('calls onSearch when button clicked', () => {
    const mockSearch = jest.fn();
    render(<SearchBar query="test" onSearch={mockSearch} />);
    fireEvent.click(screen.getByText('Search'));
    expect(mockSearch).toHaveBeenCalled();
});
```

---

## Environment Configuration

### Required Environment Variables

```bash
# .env file
IG_USERNAME="your_instagram_username"
IG_PASSWORD="your_instagram_password"
GOOGLE_API_KEY="your_google_gemini_api_key"
```

### Optional Configuration

#### Backend / CLI Configuration
- `CHROMA_DB_PATH`: Path to ChromaDB directory (default: `./data/reels_db`)
- `BATCH_SIZE`: Number of reels to process before autosave (default: 10)
- `EMBEDDING_MODEL`: SentenceTransformer model name (default: `all-MiniLM-L6-v2`)

#### Frontend Search Configuration
- `use_rocchio`: Boolean to enable/disable Rocchio query refinement (default: `false`)
- `semantic_weight`: Float (0.0-1.0) to balance semantic vs. keyword search (default: `0.7`)
- `num_paraphrases`: Integer (0-5) for number of LLM-generated query variations (default: `3`)
- `num_results`: Integer for number of documents to retrieve or use in reports (default: `5`)

---

## Common Pitfalls and Solutions

### Pitfall 1: Tight Coupling to External APIs
**Problem:** Hardcoding API calls throughout the codebase.

**Solution:** Create wrapper classes:
```python
# core/gemini_utils.py
class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def generate_paraphrases(self, query: str, n: int) -> List[str]:
        ...
    
    def generate_report(self, context: str, query: str) -> str:
        ...
```

### Pitfall 2: Monolithic Managers
**Problem:** 800-line `IngestionManager` doing everything.

**Solution:** Split into focused components:
- `ReelFetcher`: Fetches metadata from Instagram
- `ReelDownloader`: Downloads video files
- `CacheManager`: Manages local file cache
- `IngestionOrchestrator`: Coordinates the above

### Pitfall 3: UI Logic in Backend
**Problem:** Backend returning formatted HTML or UI-specific structures.

**Solution:** Backend returns pure data; frontend handles formatting:
```python
# Backend
return {"results": [{"id": "1", "score": 0.95, "text": "..."}]}

# Frontend
<ResultCard result={result} />  // Formats for display
```

### Pitfall 4: No Error Recovery
**Problem:** CLI crashes on first error, losing all progress.

**Solution:** Implement checkpointing:
```python
class IngestionManager:
    def ingest(self, urls: List[str], checkpoint_interval: int = 10):
        for i, url in enumerate(urls):
            try:
                self._process_reel(url)
            except Exception as e:
                self.logger.error(f"Failed to process {url}: {e}")
                continue
            
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(i + 1)
```

---

## Code Review Checklist

Before submitting a PR, verify:

- [ ] No file exceeds 400 lines (hard limit: 500)
- [ ] All classes follow Single Responsibility Principle
- [ ] No functions exceed 40 lines
- [ ] All dependencies are injected, not hardcoded
- [ ] Naming is descriptive and intention-revealing
- [ ] Tests are included for new functionality
- [ ] No God classes or procedural scripts
- [ ] Follows Manager/ViewModel/Coordinator patterns
- [ ] Documentation updated (docstrings, README)
- [ ] No tight coupling to external APIs

---

## Agent Behavior Guidelines

When working on this codebase as an AI agent:

1. **Always read this file first** before making changes
2. **Enforce standards ruthlessly** — reject any code that violates these rules
3. **Refactor aggressively** — if you see 400+ line files, split them
4. **Think modularity** — every new feature should be a new class
5. **Test everything** — no untested code
6. **Document decisions** — update this file when patterns evolve

---

## Project Contacts and Resources

- **Architecture Document:** `ReelRAG Architecture Decisions.pdf` (original design)
- **README:** `/README.md` (setup and usage instructions)
- **Original Prompt:** `ReelRAG-actual.yml` (historical reference)

---

## Changelog

### v1.0 (Current)
- Initial implementation of CLI, Flask backend, and React frontend
- Hybrid retrieval with semantic + lexical search
- Rocchio refinement and query paraphrasing
- Topic clustering and filtering
- LLM-powered report generation

### Future Enhancements
- Implement date range filtering
- Add re-ranking model
- Support multiple embedding models
- Add user authentication
- Deploy to production environment

---

**Remember:** The goal is not just to make code work, but to make it **maintainable, testable, and scalable**. Every line should be written with the next developer in mind.