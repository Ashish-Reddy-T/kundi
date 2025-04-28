"""
RAG indexing - Step 2: Generate Image Embeddings
Enhanced version that uses place_image_analyzer.py for CLIP embeddings
"""

import os
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import torch
import logging
import sys

# Import from the PlaceImageAnalyzer class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from place_image_analyzer import PlaceImageAnalyzer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACES_CSV = os.path.join(BASE_DIR, 'places.csv')
MEDIA_CSV = os.path.join(BASE_DIR, 'media.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ingestion')
OUTPUT_NPY = os.path.join(OUTPUT_DIR, 'image_embeddings.npy')
VISUAL_ATTRIBUTES = os.path.join(OUTPUT_DIR, 'visual_attributes.pkl')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate image embeddings with enhanced CLIP")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=400,
        help="Maximum image width for processing"
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=400,
        help="Maximum image height for processing"
    )
    parser.add_argument(
        "--use_existing",
        action="store_true",
        help="Use existing place_clip_analysis_data.pkl if available"
    )
    parser.add_argument(
        "--analysis_path",
        type=str,
        default=os.path.join(BASE_DIR, "place_clip_analysis_data.pkl"),
        help="Path to existing analysis data"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading media.csv and places.csv from {BASE_DIR}...")
    places_df = pd.read_csv(PLACES_CSV)
    media_df = pd.read_csv(MEDIA_CSV)

    print(f"Loaded {len(places_df)} places and {len(media_df)} media entries")

    # Initialize image analyzer
    analyzer = PlaceImageAnalyzer(max_image_size=(args.max_width, args.max_height))
    
    # Check if we should use existing analysis data
    if args.use_existing and os.path.exists(args.analysis_path):
        print(f"Loading existing analysis data from {args.analysis_path}")
        try:
            with open(args.analysis_path, "rb") as f:
                analyzer.result_data = pickle.load(f)
            print(f"Loaded {len(analyzer.result_data['urls'])} pre-analyzed images")
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Will generate new embeddings")

    # Create a lookup for place_id to first image URL
    print("Creating place to image mapping...")
    place_to_image = {}
    for _, row in tqdm(places_df.iterrows(), total=len(places_df)):
        place_id = row['place_id']
        urls = media_df[media_df['place_id'] == place_id]['media_url'].tolist()
        if urls:
            place_to_image[place_id] = urls[0]
    
    print(f"Found images for {len(place_to_image)} out of {len(places_df)} places")

    # Process all places in order they appear in places.csv
    image_embeddings = []
    visual_attributes = []

    for _, row in tqdm(places_df.iterrows(), total=len(places_df), desc="Processing places"):
        place_id = row['place_id']
        image_url = place_to_image.get(place_id, '')
        
        if not image_url:
            # No image available, use zero vector
            image_embeddings.append(np.zeros(512, dtype='float32'))
            visual_attributes.append({
                'place_id': place_id,
                'url': '',
                'analysis': {},
                'top_attributes': {}
            })
            continue
        
        # Check if image already analyzed
        if image_url in analyzer.result_data["urls"]:
            idx = analyzer.result_data["urls"].index(image_url)
            embed = analyzer.result_data["embeddings"][idx]
            analysis = analyzer.result_data["analysis_results"][idx]
            
            # Make sure embedding is 1D (some might be 2D with shape (1, 512))
            if embed.ndim > 1:
                embed = embed.squeeze()
                
            image_embeddings.append(embed)
            visual_attributes.append({
                'place_id': place_id,
                'url': image_url,
                'analysis': analysis,
                'top_attributes': analyzer.create_summary(analysis)
            })
        else:
            # Process new image
            result = analyzer.analyze_image(url=image_url, place_id=place_id)
            
            if "error" in result:
                # Handle error case with zero vector
                print(f"Error analyzing image for place {place_id}: {result['error']}")
                image_embeddings.append(np.zeros(512, dtype='float32'))
                visual_attributes.append({
                    'place_id': place_id,
                    'url': image_url,
                    'analysis': {},
                    'top_attributes': {}
                })
            else:
                # Get the embedding that was just added
                idx = analyzer.result_data["urls"].index(image_url)
                embed = analyzer.result_data["embeddings"][idx]
                analysis = analyzer.result_data["analysis_results"][idx]
                
                # Make sure embedding is 1D
                if embed.ndim > 1:
                    embed = embed.squeeze()
                    
                image_embeddings.append(embed)
                visual_attributes.append({
                    'place_id': place_id,
                    'url': image_url,
                    'analysis': analysis,
                    'top_attributes': analyzer.create_summary(analysis)
                })

    # Convert to numpy array and save
    image_embeddings = np.array(image_embeddings, dtype='float32')
    print(f"Saving {len(image_embeddings)} image embeddings to {OUTPUT_NPY}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save embeddings
    np.save(OUTPUT_NPY, image_embeddings)
    
    # Save visual attributes
    print(f"Saving visual attributes to {VISUAL_ATTRIBUTES}")
    with open(VISUAL_ATTRIBUTES, 'wb') as f:
        pickle.dump(visual_attributes, f)
    
    # Also save a JSON version for easier inspection
    import json
    json_path = VISUAL_ATTRIBUTES.replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        safe_attributes = []
        for item in visual_attributes:
            safe_item = {k: v for k, v in item.items()}
            if 'analysis' in safe_item:
                for category in safe_item['analysis']:
                    for attr in safe_item['analysis'][category]:
                        if 'confidence' in attr:
                            attr['confidence'] = float(attr['confidence'])
            safe_attributes.append(safe_item)
        
        json.dump(safe_attributes, f, indent=2)

    print("✅ Image embeddings and visual attributes saved successfully!")

if __name__ == "__main__":
    main()