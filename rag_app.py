"""
Advanced RAG-based Vibe Search Application
Implements a true Retrieval Augmented Generation system for finding places based on vibes
"""
import os
import pickle
import faiss
import numpy as np
import time
import logging
from typing import List, Dict, Optional
from functools import lru_cache
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI and embedding imports
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")  # Default embedding model (can be changed)

# Paths for data files
INDEX_FILE = os.path.join(BASE_DIR, "ingestion", f"final_index_{EMBEDDING_MODEL}.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "ingestion", f"final_metadata_{EMBEDDING_MODEL}.pkl")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Neighborhood mapping for query understanding
NEIGHBORHOODS = {
    "east village": ["east village", "alphabet city", "east side"],
    "west village": ["west village", "greenwich village", "west side", "village"],
    "midtown": ["midtown", "times square", "herald square", "garment district"],
    "soho": ["soho", "nolita", "little italy", "south of houston"],
    "tribeca": ["tribeca", "triangle below canal"],
    "upper east side": ["upper east side", "ues", "yorkville", "lenox hill"],
    "upper west side": ["upper west side", "uws", "lincoln square"],
    "brooklyn": ["brooklyn", "williamsburg", "bushwick", "dumbo", "greenpoint"],
    "queens": ["queens", "astoria", "long island city", "lic", "flushing"],
}

# Query expansion templates for specific vibes
QUERY_EXPANSIONS = {
    "date": "romantic intimate cozy dinner candlelit atmosphere",
    "work": "wifi laptop quiet study spacious productivity coffee",
    "sunny": "outdoor patio terrace sunshine bright open air",
    "weekend": "fun relaxing activities events lively exciting",
    "brunch": "breakfast eggs mimosas weekend morning relaxed",
    "coffee": "cafe espresso latte cappuccino tea",
    "dinner": "restaurant evening meal food",
    "lunch": "quick meal food casual midday",
    "drinks": "bar cocktail wine beer alcohol happy hour",
    "dance": "dancing club music dj party nightlife",
    "cheap": "affordable inexpensive budget reasonable",
    "fancy": "upscale elegant high-end luxury",
    "quiet": "peaceful calm relaxing tranquil",
    "lively": "energetic bustling vibrant loud",
    "unique": "special interesting different unusual",
    "outdoor": "patio terrace garden fresh air",
    "cozy": "warm comfortable intimate snug",
    "group": "large party friends gathering",
    "view": "skyline scenic overlook rooftop",
}

# Models for API requests
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20
    neighborhood: Optional[str] = None
    vibes: Optional[List[str]] = None
    match_mode: Optional[str] = "vibe" # Field: "vibe"(default) or "strict"

    
class Place(BaseModel):
    place_id: str
    name: str
    neighborhood: str
    latitude: float
    longitude: float
    emoji: str
    short_description: Optional[str] = None
    image_url: Optional[str] = None
    score: float
    explanation: Optional[str] = None
    vibe_tags: Optional[List[str]] = None
    
class SearchResponse(BaseModel):
    places: List[Place]
    query: str
    expanded_query: Optional[str] = None
    processing_time: float
    message: Optional[str] = None
    
# ---------------------------------------------------------
# Initialize models and data
# ---------------------------------------------------------
# Load index and metadata with error handling

try:
    logger.info(f"Loading index from {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)
    logger.info(f"Loading metadata from {METADATA_FILE}")
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    logger.info(f"Loaded index with {len(metadata)} places")
except Exception as e:
    logger.error(f"Error loading index or metadata: {e}")
    # Create dummy data for testing
    index = None
    metadata = []
    logger.warning("Using empty index and metadata! Run ingestion/rag_index.py first.")

# Initialize the embedding model
embedding_model_map = {
    "minilm": "all-MiniLM-L6-v2",           # <-- Default
    "bge-small": "BAAI/bge-small-en-v1.5",  # <-- Balance b/w performance and speed
    "bge-base": "BAAI/bge-base-en-v1.5",    # <-- Highly suitable for speed users
    "bge-large": "BAAI/bge-large-en-v1.5",  # <-- Currently Best Model
    "mpnet": "all-mpnet-base-v2"            # <-- Just in case!
}


try:
    model_name = embedding_model_map.get(EMBEDDING_MODEL, "BAAI/bge-large-en-v1.5")
    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    embedding_model = None
    logger.warning("No embedding model available! Search will not work properly.")

# ---------------------------------------------------------
# Core search functions 
# ---------------------------------------------------------

def expand_query(query: str) -> str:
    """Expand the search query using predefined expansions"""
    expanded_terms = []
    query_lower = query.lower()
    
    # Look for vibe terms in the query
    for term, expansion in QUERY_EXPANSIONS.items():
        if term in query_lower:
            expanded_terms.append(expansion)
    
    # If we found expansions, combine them with the original query
    if expanded_terms:
        expanded_query = f"{query} {' '.join(expanded_terms)}"
        logger.info(f"Expanded query: {expanded_query}")
        return expanded_query
    
    return query

def get_explanation(query: str, place: Dict) -> str:
    """Generate an explanation for why a place matches the query"""
    vibe_tags = place.get('vibe_tags', [])
    name = place.get('name', '')
    neighborhood = place.get('neighborhood', '')
    
    # Default explanation
    explanation = f"Matches your search for '{query}'"
    
    # For specific vibes in the query
    query_lower = query.lower()
    
    # Check for neighborhood mentions
    if neighborhood and any(hood in query_lower for hood in [neighborhood.lower(), *NEIGHBORHOODS.get(neighborhood.lower(), [])]):
        explanation = f"Located in {neighborhood}, matching your location preference"
    
    # Check for vibe matches
    elif "date" in query_lower and any(tag in ['date_night', 'romantic', 'intimate'] for tag in vibe_tags):
        explanation = f"Romantic spot perfect for a date night"
    elif "work" in query_lower and any(tag in ['work_friendly', 'quiet', 'spacious'] for tag in vibe_tags):
        explanation = f"Great place to work with good atmosphere"
    elif "sunny" in query_lower and any(tag in ['outdoor_vibes', 'patio'] for tag in vibe_tags):
        explanation = f"Offers excellent outdoor space for sunny days"
    elif "cozy" in query_lower and any(tag in ['quiet_relaxing', 'cozy'] for tag in vibe_tags):
        explanation = f"Cozy atmosphere perfect for relaxing"
    elif "unique" in query_lower and any(tag in ['unique_special', 'special'] for tag in vibe_tags):
        explanation = f"Unique, special place with distinctive character"
    elif "dancing" in query_lower and any(tag in ['dancing_music', 'nightlife'] for tag in vibe_tags):
        explanation = f"Great spot for dancing and nightlife"
    elif "drinks" in query_lower and any(tag in ['drinks_focus', 'bar'] for tag in vibe_tags):
        explanation = f"Known for their excellent drink options"
    elif "coffee" in query_lower and any(tag in ['coffee_tea', 'cafe'] for tag in vibe_tags):
        explanation = f"Popular cafe with great coffee options"
    elif "cheap" in query_lower and any(tag in ['budget_friendly', 'affordable'] for tag in vibe_tags):
        explanation = f"Budget-friendly place that won't break the bank"
    elif "group" in query_lower and any(tag in ['group_hangout', 'social'] for tag in vibe_tags):
        explanation = f"Great for groups and social gatherings"
    
    return explanation


def extract_neighborhood(query: str) -> Optional[str]:
    """Extract neighborhood from query"""
    query_lower = query.lower()
    
    # Check for direct neighborhood mentions
    for hood, variations in NEIGHBORHOODS.items():
        for variant in variations:
            if variant in query_lower:
                return hood
    
    return None

def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a text string and pad/adapt to match combined (text+image) dimension"""
    if not embedding_model:
        return np.random.rand(1536).astype('float32')  # 1536 because 1024 + 512
        
    embedding = embedding_model.encode(text)
    embedding = embedding / np.linalg.norm(embedding)

    # Handle different text embedding dimensions based on model
    text_dim = embedding.shape[0]
    
    if text_dim == 384:  # minilm
        # Pad with zeros for the image part (512-d)
        padding = np.zeros(1152, dtype='float32')  # 1536 - 384 = 1152
        combined_embedding = np.hstack([embedding, padding])
    elif text_dim == 768:  # bge-base or mpnet
        # Pad with zeros for the image part
        padding = np.zeros(768, dtype='float32')  # 1536 - 768 = 768
        combined_embedding = np.hstack([embedding, padding])
    elif text_dim == 1024:  # bge-large
        # Pad with zeros for the image part
        padding = np.zeros(512, dtype='float32')  # 1536 - 1024 = 512
        combined_embedding = np.hstack([embedding, padding])
    else:
        raise ValueError(f"Unexpected text embedding size {text_dim}")
    
    return combined_embedding.astype('float32')


def search_places(query: str, limit: int = 20, neighborhood_filter: Optional[str] = None, 
                 vibe_filters: Optional[List[str]] = None, match_mode: str = "vibe", timeout: float = 5.0) -> List[Dict]:
    """Search for places matching the query"""
    start_time = time.time()
    
    # Validate inputs
    if not query.strip():
        return []
    
    if not index or not metadata or len(metadata) == 0:
        logger.error("No index or metadata available")
        return []
    
    query_length = len(query.split())

    if match_mode == "strict":
        if query_length <= 3:
            lexical_bonus_weight = 0.35
        elif query_length <= 6:
            lexical_bonus_weight = 0.20
        else:
            lexical_bonus_weight = 0.10
    else:  # "vibe" mode
        lexical_bonus_weight = 0.05

        
    try:
        # Expand query for better retrieval
        expanded_query = expand_query(query)
        
        # Generate embedding for the query
        query_embedding = embed_text(expanded_query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        search_limit = min(limit * 4, len(metadata))  # Get more for filtering
        D, I = index.search(query_embedding, search_limit)
        
        # Process results
        results = []
        for score, idx in zip(D[0], I[0]):
            if time.time() - start_time > timeout:
                logger.warning(f"Search timeout after {len(results)} results")
                break

            if idx < 0 or idx >= len(metadata):
                continue

            place = metadata[idx]

            # --- FILTERS ---
            if neighborhood_filter:
                place_hood = place.get('neighborhood', '').lower()
                if neighborhood_filter.lower() not in place_hood:
                    continue

            if vibe_filters:
                place_vibes = place.get('vibe_tags', [])
                if not any(vibe in place_vibes for vibe in vibe_filters):
                    continue

            # --- LEXICAL MATCHING and BONUS ---
            text_fields = []
            if place.get("name"):
                text_fields.append(place["name"].lower())
            if place.get("short_description"):
                text_fields.append(place["short_description"].lower())
            if place.get("tags"):
                text_fields.extend([tag.lower() for tag in place["tags"]])
            if place.get("vibe_tags"):
                text_fields.extend([vibe.lower() for vibe in place["vibe_tags"]])

            query_words = set(query.lower().split())
            match_count = 0
            for word in query_words:
                for field in text_fields:
                    if word in field:
                        match_count += 1
                        break

            if len(query_words) > 0:
                lexical_match_score = match_count / len(query_words)
            else:
                lexical_match_score = 0.0

            adjusted_score = float(score) + lexical_bonus_weight * lexical_match_score

            # --- BUILD RESULT OBJECT ---
            explanation = get_explanation(query, place)

            result = {
                'place_id': place.get('place_id', ''),
                'name': place.get('name', ''),
                'neighborhood': place.get('neighborhood', ''),
                'latitude': place.get('latitude', 0.0),
                'longitude': place.get('longitude', 0.0),
                'emoji': place.get('emoji', '📍'),
                'short_description': place.get('short_description', ''),
                'image_url': place.get('image_url', ''),
                'vibe_tags': place.get('vibe_tags', []),
                'score': adjusted_score,                      # USE adjusted_score here!
                'semantic_score': float(score),                # (optional)
                'lexical_match_score': lexical_match_score,    # (optional)
                'explanation': explanation
            }

            # --- APPEND ---
            results.append(result)

            if len(results) >= limit:
                break
        
        # Sort by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------
app = FastAPI(title="Vibe Search RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates for HTML rendering
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
        "index_size": index.ntotal if index else 0,
        "metadata_count": len(metadata)
    }

# Main search API endpoint
@app.post("/api/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    """Search for places matching a query"""
    start_time = time.time()
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Extract neighborhood from query if not provided explicitly
    neighborhood = request.neighborhood
    if not neighborhood:
        neighborhood = extract_neighborhood(request.query)
        
    # Search for places
    results = search_places(
        query=request.query,
        limit=request.limit,
        neighborhood_filter=neighborhood,
        vibe_filters=request.vibes,
        match_mode=request.match_mode
    )
    
    # Calculate processing time
    elapsed = time.time() - start_time
    
    # Create response
    places = [Place(**place) for place in results]
    
    return SearchResponse(
        places=places,
        query=request.query,
        expanded_query=expand_query(request.query),
        processing_time=elapsed,
        message=f"Found {len(places)} places in {elapsed:.2f} seconds"
    )

# Web UI route
@app.get("/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Serve the web UI"""
    try:
        return templates.TemplateResponse(
            "rag_ui.html", 
            {"request": request, "embedding_model": EMBEDDING_MODEL}
        )
    except Exception as e:
        logger.error(f"Error serving template: {e}")
        # Fallback HTML
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vibe Search RAG</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1>Vibe Search RAG</h1>
                <p>Template not found. Please check your installation or use the API directly.</p>
                <p>Try the API at <code>/api/search</code> with a POST request.</p>
                <p>Or check the health endpoint at <a href="/health">/health</a>.</p>
            </div>
        </body>
        </html>
        """)

# ---------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vibe Search RAG server")
    uvicorn.run("rag_app:app", host="0.0.0.0", port=8000, reload=True)