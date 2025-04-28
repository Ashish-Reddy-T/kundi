"""
Enhanced RAG indexing script for Vibe Search
Uses sentence-transformers for semantic understanding of place data
Incorporates visual attributes from place_image_analyzer
"""
import os
import pandas as pd
import numpy as np
import pickle
import faiss
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Constants
EMBED_MODELS = {
    'minilm': 'all-MiniLM-L6-v2',    # Fast, good for testing (384 dimensions)
    'bge-small': 'BAAI/bge-small-en-v1.5',  # Fast, good quality (384 dimensions)
    'bge-base': 'BAAI/bge-base-en-v1.5',    # Good balance (768 dimensions)
    'bge-large': 'BAAI/bge-large-en-v1.5',  # High quality but slower (1024 dimensions)
    'mpnet': 'all-mpnet-base-v2',     # High quality (768 dimensions)
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACES_CSV = os.path.join(BASE_DIR, 'places.csv')
REVIEWS_CSV = os.path.join(BASE_DIR, 'reviews.csv')
MEDIA_CSV = os.path.join(BASE_DIR, 'media.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ingestion')
VISUAL_ATTRIBUTES = os.path.join(OUTPUT_DIR, 'visual_attributes.pkl')

# Vibe categories for metadata enrichment
VIBE_CATEGORIES = {
    "date_night": [
        "romantic", "intimate", "cozy", "candlelit", "date", "charming"
    ],
    "work_friendly": [
        "wifi", "laptop", "outlets", "quiet", "work", "study", "spacious"
    ],
    "outdoor_vibes": [
        "outdoor", "patio", "terrace", "garden", "rooftop", "fresh air",
        "alfresco", "sunshine", "sunny"
    ],
    "group_hangout": [
        "group", "friends", "party", "social", "gathering", "fun", "lively"
    ],
    "food_focus": [
        "food", "restaurant", "delicious", "menu", "chef", "cuisine", "eat",
        "dining", "dinner", "lunch", "brunch"
    ],
    "drinks_focus": [
        "bar", "drink", "cocktail", "beer", "wine", "alcohol", "happy hour"
    ],
    "coffee_tea": [
        "coffee", "cafe", "espresso", "latte", "cappuccino", "tea", "matcha"
    ],
    "dancing_music": [
        "dance", "club", "dj", "music", "live", "performance", "party"
    ],
    "quiet_relaxing": [
        "quiet", "peaceful", "calm", "relaxing", "tranquil", "serene"
    ],
    "upscale_fancy": [
        "upscale", "fancy", "elegant", "luxury", "high-end", "fine dining"
    ],
    "casual_lowkey": [
        "casual", "relaxed", "laid-back", "informal", "simple"
    ],
    "unique_special": [
        "unique", "special", "quirky", "interesting", "eclectic", "hidden gem"
    ],
    "trendy_cool": [
        "trendy", "hip", "cool", "stylish", "instagram", "fashionable"
    ],
    "budget_friendly": [
        "affordable", "cheap", "budget", "inexpensive", "reasonable"
    ]
}

# Visual attribute to vibe mapping
VISUAL_TO_VIBE_MAPPING = {
    "atmosphere:upscale/fancy": "upscale_fancy",
    "atmosphere:casual/relaxed": "casual_lowkey",
    "atmosphere:romantic": "date_night",
    "atmosphere:family-friendly": "group_hangout",
    "atmosphere:hipster/trendy": "trendy_cool",
    "atmosphere:cozy/intimate": "date_night",
    "atmosphere:lively/energetic": "group_hangout",
    "atmosphere:quiet/peaceful": "quiet_relaxing",
    "vibes:date night spot": "date_night",
    "vibes:group hangout": "group_hangout",
    "vibes:work-friendly": "work_friendly",
    "vibes:instagram-worthy": "trendy_cool",
    "vibes:hidden gem": "unique_special",
    "vibes:special occasion": "upscale_fancy",
    "vibes:everyday casual": "casual_lowkey",
    "vibes:budget-friendly": "budget_friendly",
    "setting:outdoor patio": "outdoor_vibes",
    "setting:rooftop": "outdoor_vibes",
    "setting:garden/green space": "outdoor_vibes",
    "place_type:cafe": "coffee_tea",
    "place_type:cocktail bar": "drinks_focus",
    "place_type:restaurant": "food_focus",
    "place_type:rooftop lounge": "outdoor_vibes",
    "place_type:nightclub": "dancing_music"
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build vector index for Vibe Search")
    parser.add_argument(
        "--model", 
        choices=list(EMBED_MODELS.keys()), 
        default="minilm",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--skip_visual", 
        action="store_true",
        help="Skip incorporating visual attributes"
    )
    return parser.parse_args()

def clean_tags(tags_str):
    """Clean tags string from the CSV format"""
    if pd.isna(tags_str):
        return []
    # Convert "{tag1,tag2}" format to list
    tags = tags_str.replace("{", "").replace("}", "").split(",")
    return [t.strip() for t in tags if t.strip()]

def extract_vibe_tags(text):
    """Extract vibe tags based on keyword matching"""
    text = text.lower()
    detected_vibes = []
    
    for vibe_name, keywords in VIBE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                detected_vibes.append(vibe_name)
                break
                
    return detected_vibes

def load_visual_attributes():
    """Load visual attributes from file if available"""
    if os.path.exists(VISUAL_ATTRIBUTES):
        try:
            print(f"Loading visual attributes from {VISUAL_ATTRIBUTES}")
            with open(VISUAL_ATTRIBUTES, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading visual attributes: {e}")
            
    print("No visual attributes found, proceeding without them")
    return []

def extract_visual_vibes(visual_attr):
    """Extract vibe tags from visual analysis"""
    visual_vibes = set()
    
    if not visual_attr or not visual_attr.get('top_attributes'):
        return []
        
    top_attrs = visual_attr.get('top_attributes', {})
    
    # Map visual attributes to vibes
    for category, attribute in top_attrs.items():
        mapping_key = f"{category}:{attribute}"
        if mapping_key in VISUAL_TO_VIBE_MAPPING:
            visual_vibes.add(VISUAL_TO_VIBE_MAPPING[mapping_key])
    
    # Check for any vibes specifically detected
    if 'analysis' in visual_attr and 'vibes' in visual_attr['analysis']:
        for vibe_item in visual_attr['analysis']['vibes']:
            vibe_tag = vibe_item['tag']
            mapping_key = f"vibes:{vibe_tag}"
            if mapping_key in VISUAL_TO_VIBE_MAPPING:
                visual_vibes.add(VISUAL_TO_VIBE_MAPPING[mapping_key])
            
    return list(visual_vibes)

def create_document(place, reviews_df, visual_attr=None):
    """Create a rich text representation of a place including visual attributes"""
    # Basic place info
    name = str(place['name']) if pd.notna(place['name']) else ''
    neighborhood = str(place['neighborhood']) if pd.notna(place['neighborhood']) else ''
    desc = str(place['short_description']) if pd.notna(place['short_description']) else ''
    
    # Clean and process tags
    tags = clean_tags(place.get('tags', ''))
    tags_str = ' '.join(tags)
    
    # Get reviews for this place
    place_reviews = reviews_df[reviews_df['place_id'] == place['place_id']]['review_text'].tolist()
    reviews_text = ' '.join([str(r) for r in place_reviews[:5] if pd.notna(r)])
    
    # Add visual attributes if available
    visual_text = ""
    if visual_attr:
        top_attrs = visual_attr.get('top_attributes', {})
        if top_attrs:
            visual_terms = []
            for category, attribute in top_attrs.items():
                visual_terms.append(f"{category}: {attribute}")
            
            visual_text = "Visual attributes: " + ". ".join(visual_terms)
    
    # Create a rich document for embedding
    doc = f"Place: {name}. {desc}. Located in {neighborhood}. Type: {tags_str}. " 
    
    if visual_text:
        doc += f"{visual_text}. "
        
    doc += f"Reviews: {reviews_text[:500]}"
    
    return doc

def main():
    args = parse_args()
    model_name = EMBED_MODELS[args.model]
    batch_size = args.batch_size
    
    # Output files based on model name
    text_embeddings_file = os.path.join(OUTPUT_DIR, f"text_embeddings_{args.model}.npy")
    metadata_file = os.path.join(OUTPUT_DIR, f"metadata_{args.model}.pkl")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {BASE_DIR}...")
    places_df = pd.read_csv(PLACES_CSV)
    reviews_df = pd.read_csv(REVIEWS_CSV)
    media_df = pd.read_csv(MEDIA_CSV)
    
    print(f"Loaded {len(places_df)} places, {len(reviews_df)} reviews, {len(media_df)} media items")

    # Load visual attributes if available
    visual_attributes = [] if args.skip_visual else load_visual_attributes()
    visual_attr_by_place = {}
    
    if visual_attributes:
        print(f"Creating lookup for {len(visual_attributes)} visual attributes")
        for attr in visual_attributes:
            if 'place_id' in attr and attr['place_id']:
                visual_attr_by_place[attr['place_id']] = attr
                
        print(f"Found visual attributes for {len(visual_attr_by_place)} places")

    # Load the text model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Prepare documents and metadata
    print("Creating documents...")
    documents = []
    metadata = []

    for _, row in tqdm(places_df.iterrows(), total=len(places_df), desc="Preparing documents"):
        place_id = str(row['place_id'])
        
        # Get visual attributes for this place
        visual_attr = visual_attr_by_place.get(place_id)
        
        # Create text document
        doc = create_document(row, reviews_df, visual_attr)
        documents.append(doc)

        # Get image URL
        media_urls = media_df[media_df['place_id'] == place_id]['media_url'].tolist()
        image_url = media_urls[0] if media_urls else ''

        # Extract textual vibes
        combined_text = f"{row['name']} {row['short_description']} {row.get('tags', '')} "
        combined_text += ' '.join(reviews_df[reviews_df['place_id'] == place_id]['review_text'].head(5).tolist())
        text_vibes = extract_vibe_tags(combined_text)
        
        # Extract visual vibes
        visual_vibes = extract_visual_vibes(visual_attr) if visual_attr else []
        
        # Combine text and visual vibes
        all_vibes = list(set(text_vibes + visual_vibes))
        
        # Create metadata
        meta = {
            'place_id': place_id,
            'name': str(row['name']) if pd.notna(row['name']) else '',
            'neighborhood': str(row['neighborhood']) if pd.notna(row['neighborhood']) else '',
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'emoji': str(row['emoji']) if pd.notna(row['emoji']) else '📍',
            'short_description': str(row['short_description']) if pd.notna(row['short_description']) else '',
            'tags': clean_tags(row.get('tags', '')),
            'image_url': image_url,
            'vibe_tags': all_vibes,
            'text_vibes': text_vibes,        # Separate for reference
            'visual_vibes': visual_vibes     # Separate for reference
        }
        
        # Add visual attributes if available
        if visual_attr and 'top_attributes' in visual_attr:
            meta['visual_attributes'] = visual_attr['top_attributes']
        
        metadata.append(meta)
    
    # Generate text embeddings
    print("Generating text embeddings...")
    text_embeddings = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding text"):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        text_embeddings.append(batch_embeddings)

    # Stack into single array
    text_embeddings = np.vstack(text_embeddings).astype('float32')

    # Save text embeddings
    print(f"Saving text embeddings to {text_embeddings_file}")
    np.save(text_embeddings_file, text_embeddings)

    # Save metadata
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print("Step 1 completed successfully! ✅ Text embeddings ready.")

if __name__ == "__main__":
    main()