"""
Enhanced RAG indexing - Step 3: Combine Text + Image Embeddings with weighting
Creates a FAISS index from combined text and visual embeddings with configurable weights
"""

import os
import numpy as np
import pickle
import faiss
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Combine text and image embeddings")
    parser.add_argument(
        "--model",
        choices=["minilm", "bge-small", "bge-base", "bge-large", "mpnet"],
        default="bge-large",
        help="Which text model's embeddings to combine with images"
    )
    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.7,
        help="Weight for text embeddings (0.0-1.0)"
    )
    parser.add_argument(
        "--image_weight",
        type=float,
        default=0.3,
        help="Weight for image embeddings (0.0-1.0)"
    )
    return parser.parse_args()

def normalize_embedding(embedding):
    """Normalize embedding to unit length"""
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norm, 1e-8)  # Avoid division by zero

def main():
    args = parse_args()
    model_name = args.model
    
    text_weight = max(0.0, min(1.0, args.text_weight))  # Clamp to [0,1]
    image_weight = max(0.0, min(1.0, args.image_weight))
    
    # Normalize weights to sum to 1
    total_weight = text_weight + image_weight
    if total_weight > 0:
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
    else:
        # Default if both weights are 0
        text_weight = 0.7
        image_weight = 0.3
    
    logger.info(f"Using weights: text={text_weight:.2f}, image={image_weight:.2f}")

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'ingestion')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_emb_file = os.path.join(OUTPUT_DIR, f'text_embeddings_{model_name}.npy')
    image_emb_file = os.path.join(OUTPUT_DIR, 'image_embeddings.npy')
    metadata_file = os.path.join(OUTPUT_DIR, f'metadata_{model_name}.pkl')

    index_file = os.path.join(OUTPUT_DIR, f'final_index_{model_name}.faiss')
    final_metadata_file = os.path.join(OUTPUT_DIR, f'final_metadata_{model_name}.pkl')

    # Load embeddings
    logger.info("Loading embeddings...")
    text_embeddings = np.load(text_emb_file)
    image_embeddings = np.load(image_emb_file)

    logger.info(f"Loaded text embeddings: {text_embeddings.shape}")
    logger.info(f"Loaded image embeddings: {image_embeddings.shape}")

    if text_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError(f"Mismatch between number of text embeddings ({text_embeddings.shape[0]}) and image embeddings ({image_embeddings.shape[0]})!")

    # Normalize embeddings separately first
    logger.info("Normalizing embeddings...")
    text_embeddings_norm = normalize_embedding(text_embeddings)
    image_embeddings_norm = normalize_embedding(image_embeddings)
    
    # Apply weights and combine
    logger.info("Combining embeddings with weights...")
    combined_embeddings = np.hstack([
        text_embeddings_norm * text_weight,
        image_embeddings_norm * image_weight
    ]).astype('float32')
    
    logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")

    # Normalize the final combined vectors
    logger.info("Normalizing combined embeddings...")
    faiss.normalize_L2(combined_embeddings)

    # Create FAISS index
    logger.info("Creating FAISS index...")
    d = combined_embeddings.shape[1]  # Dimension of the combined embeddings
    
    # Use inner product (cosine similarity) since vectors are normalized
    index = faiss.IndexFlatIP(d)
    index.add(combined_embeddings)
    
    logger.info(f"Created index with {index.ntotal} vectors of dimension {d}")

    # Save FAISS index
    logger.info(f"Saving final index to {index_file}")
    faiss.write_index(index, index_file)

    # Save Metadata (unchanged)
    logger.info(f"Saving metadata to {final_metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    with open(final_metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    logger.info("✅ All Done! Combined index created successfully.")
    
    # Create a JSON-friendly version for easier inspection
    try:
        import json
        
        info = {
            "index_stats": {
                "vectors": index.ntotal,
                "dimension": d,
                "text_embedding_size": text_embeddings.shape[1],
                "image_embedding_size": image_embeddings.shape[1]
            },
            "weights": {
                "text_weight": text_weight,
                "image_weight": image_weight
            },
            "model": model_name,
            "places_count": len(metadata)
        }
        
        info_file = os.path.join(OUTPUT_DIR, f'index_info_{model_name}.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved index info to {info_file}")
    except Exception as e:
        logger.error(f"Error creating index info: {e}")

if __name__ == "__main__":
    main()