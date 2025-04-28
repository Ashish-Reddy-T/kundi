#!/usr/bin/env python3
"""
Place-Specific CLIP Image Analyzer
Analyzes images of restaurants, bars, parks, attractions, museums, and rooftop lounges
using OpenAI's CLIP model with tailored categories and attributes.
"""

import argparse
import pickle
import logging
import time
import os
import sys
import traceback
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
import faiss
import json
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Force using CPU to avoid GPU memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class PlaceImageAnalyzer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", embedding_dim=512, max_image_size=(400, 400)):
        """Initialize the CLIP model and processor with compatible versions"""
        logger.info(f"Loading CLIP model: {model_name}")
        self.device = "cpu"  # Force CPU usage for stability
        logger.info(f"Using device: {self.device}")
        
        # Dynamically import to avoid loading until needed
        try:
            # Import specific version to avoid compatibility issues
            from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
            
            # Load model
            self.model = CLIPModel.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create processor components separately to avoid compatibility issues
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model_name,
                do_resize=True,
                size={"height": 224, "width": 224},
                do_center_crop=True,
                do_normalize=True
            )
            
            # Manual implementation of processor function
            self.model_loaded = True
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            logger.error(traceback.format_exc())
            self.model_loaded = False
        
        self.max_image_size = max_image_size
        self.embedding_dim = embedding_dim
        
        # Tailored categories for places
        self.categories = {
            "place_type": [
                "restaurant", "cafe", "cocktail bar", "pub/sports bar", "wine bar", 
                "rooftop lounge", "nightclub", "food truck", "food hall/market",
                "museum/gallery", "park/garden", "historic landmark", "theater/performance venue",
                "bookstore", "boutique shop", "hotel", "beach", "viewpoint/scenic spot",
                "outdoor recreation area", "farmer's market"
            ],
            
            "setting": [
                "indoor dining room", "outdoor patio", "rooftop", "waterfront", 
                "garden/green space", "urban streetscape", "historic building",
                "modern building", "mall/shopping center", "warehouse/industrial space",
                "basement/underground", "courtyard", "beachfront", "mountainside",
                "lakeside", "riverside"
            ],
            
            "atmosphere": [
                "upscale/fancy", "casual/relaxed", "romantic", "family-friendly",
                "hipster/trendy", "retro/nostalgic", "cozy/intimate", "lively/energetic",
                "quiet/peaceful", "bustling/busy", "social/communal", "formal/elegant",
                "artistic/creative", "sporty/athletic", "natural/organic", "industrial/raw",
                "minimalist/modern", "classic/traditional", "festive/celebratory"
            ],
            
            "lighting": [
                "bright natural light", "moody/dim lighting", "candlelit", "string lights/fairy lights",
                "neon/colorful lighting", "warm ambient lighting", "cool/blue tones",
                "dramatic spotlights", "fireplace/fire pit", "lanterns/pendants"
            ],
            
            "colors_and_materials": [
                "wood-dominant", "metal/industrial", "brick/exposed walls", "stone/marble",
                "glass/reflective surfaces", "leather upholstery", "colorful/vibrant",
                "neutral/earth tones", "monochromatic", "pastel colors",
                "bold/contrasting colors", "concrete/brutalist", "greenery/plants",
                "textile/soft furnishings", "water features", "tile/mosaic"
            ],
            
            "furniture_and_decor": [
                "modern furniture", "vintage/antique furniture", "minimalist decor",
                "maximalist/eclectic", "open kitchen/bar seating", "booth seating",
                "communal tables", "lounge furniture", "artistic installations",
                "wall art/murals", "hanging plants", "bookshelves/library",
                "exposed kitchen", "stage/performance area", "retail displays",
                "exhibition space"
            ],
            
            "view": [
                "city skyline view", "water view", "garden/nature view", "street view", 
                "mountain view", "no notable view", "interior courtyard view",
                "people-watching setup"
            ],
            
            "crowd": [
                "empty/no people", "sparsely populated", "comfortably filled", 
                "crowded/bustling", "waiting line visible", "mix of demographics",
                "primarily young crowd", "family-oriented", "business/professional",
                "tourist-heavy"
            ],
            
            "time_context": [
                "daytime", "evening/night", "golden hour", "breakfast/brunch setting",
                "lunch service", "dinner service", "late night", "special event"
            ],
            
            "food_and_drink_focus": [
                "plated food visible", "drinks/cocktails featured", "coffee focus",
                "wine/beer prominent", "desserts displayed", "casual fare",
                "fine dining presentation", "buffet/self-service", "street food",
                "farm-to-table aesthetic", "international cuisine", "traditional cuisine"
            ],
            
            "vibes": [
                "date night spot", "group hangout", "work-friendly", "instagram-worthy",
                "hidden gem", "tourist attraction", "local favorite", "special occasion",
                "everyday casual", "wellness/healthy", "indulgent/comfort food",
                "cultural experience", "outdoor adventure", "urban exploration",
                "relaxation retreat", "party atmosphere", "educational/informative",
                "luxurious experience", "budget-friendly", "pet-friendly"
            ]
        }
        
        # Define structured prompts with emphasis on contrast
        self.structured_prompts = {}
        for category, attributes in self.categories.items():
            # Format each attribute as "a photo of [attribute]" to improve CLIP performance
            self.structured_prompts[category] = [f"a photo of {attr}" for attr in attributes]
        
        # Setup FAISS index for similarity search
        self.setup_faiss()
        
        # Storage for our analysis results
        self.result_data = {
            "urls": [],
            "embeddings": [],
            "analysis_results": [],
            "place_ids": []  # Associate with place_id if available
        }
        
        # Load previous data if available
        self.load_data()

    def setup_faiss(self):
        """Initialize FAISS index"""
        logger.info("Setting up FAISS index")
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance (Euclidean)
            self.faiss_loaded = True
        except Exception as e:
            logger.error(f"Error setting up FAISS: {e}")
            self.faiss_loaded = False
        
    def load_data(self, file_path="place_clip_analysis_data.pkl"):
        """Load previous analysis data if available"""
        if os.path.exists(file_path):
            logger.info(f"Loading previous data from {file_path}")
            try:
                with open(file_path, "rb") as f:
                    self.result_data = pickle.load(f)
                
                # Rebuild FAISS index with loaded data
                if len(self.result_data["embeddings"]) > 0 and self.faiss_loaded:
                    embeddings = np.vstack(self.result_data["embeddings"])
                    self.index.add(embeddings)
                    logger.info(f"Loaded {len(self.result_data['urls'])} previous images into FAISS index")
            except Exception as e:
                logger.error(f"Error loading previous data: {e}")
                logger.info("Starting fresh")
        else:
            logger.info("No previous data found, starting fresh")
    
    def clear_data(self):
        """Clear all stored data"""
        self.result_data = {
            "urls": [],
            "embeddings": [],
            "analysis_results": [],
            "place_ids": []
        }
        # Reset FAISS index
        if self.faiss_loaded:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info("All analysis data cleared")
    
    def save_data(self, file_path="place_clip_analysis_data.pkl"):
        """Save analysis data to pickle file"""
        try:
            logger.info(f"Saving data to {file_path}")
            with open(file_path, "wb") as f:
                pickle.dump(self.result_data, f)
                
            # Also save a JSON version for easier inspection
            json_path = file_path.replace('.pkl', '.json')
            json_data = {
                "urls": self.result_data["urls"],
                "analysis_results": self.result_data["analysis_results"],
                "place_ids": self.result_data["place_ids"]
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def load_image_from_url(self, url):
        """Load image from URL with size limits and error handling"""
        logger.info(f"Loading image from {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if needed (handles PNG with transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            logger.info(f"Image loaded successfully, original size: {img.size}")
            
            # Resize image if larger than max_image_size
            if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                # Calculate new size maintaining aspect ratio
                ratio = min(self.max_image_size[0] / img.size[0], self.max_image_size[1] / img.size[1])
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                logger.info(f"Image resized to {new_size}")
                
            return img
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def process_image(self, image):
        """Custom implementation of processor to avoid compatibility issues"""
        if image is None:
            return None
            
        try:
            # Convert PIL image to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Process image with image processor
            image_inputs = self.image_processor(images=image, return_tensors="pt")
            return image_inputs
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
            
    def process_text(self, texts):
        """Process text inputs"""
        try:
            text_inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            return text_inputs
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None
    
    def get_image_features(self, image):
        """Extract features from image using CLIP with memory management"""
        if not self.model_loaded or image is None:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
            
        logger.info("Getting image features")
        try:
            # Process image
            inputs = self.process_image(image)
            if inputs is None:
                return np.zeros((1, self.embedding_dim), dtype=np.float32)
                
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Free up memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features.cpu().numpy()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory. Try using a smaller image size.")
            else:
                logger.error(f"Runtime error: {e}")
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting image features: {e}")
            logger.error(traceback.format_exc())
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
    
    def get_text_similarities(self, image_features, text_prompts, top_k=5):
        """Calculate similarities between image and text prompts"""
        if not self.model_loaded:
            return []
            
        try:
            # Process text inputs
            text_inputs = self.process_text(text_prompts)
            if text_inputs is None:
                return []
                
            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarities
            similarities = (100.0 * torch.matmul(
                torch.from_numpy(image_features), 
                text_features.cpu().T
            )).squeeze()
            
            # Handle case where similarities is a 0-dim tensor
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)
                
            # Get top matches
            if len(text_prompts) <= top_k:
                values, indices = similarities, torch.arange(len(text_prompts))
            else:
                values, indices = similarities.topk(min(top_k, len(text_prompts)))
            
            results = []
            for value, idx in zip(values.tolist(), indices.tolist()):
                if idx < len(text_prompts):  # Avoid index out of range
                    # Strip the "a photo of" part from the prompt for cleaner results
                    original_prompt = text_prompts[idx]
                    clean_tag = original_prompt.replace("a photo of ", "")
                    
                    results.append({
                        "tag": clean_tag,
                        "confidence": float(value)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error calculating text similarities: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def analyze_image(self, url, place_id=None, force=False):
        """Analyze image and return structured attributes with improved scoring"""
        start_time = time.time()
        
        # Check if URL already analyzed
        if url in self.result_data["urls"] and not force:
            logger.info(f"URL already analyzed: {url}")
            idx = self.result_data["urls"].index(url)
            return {
                "url": url,
                "place_id": self.result_data["place_ids"][idx] if len(self.result_data["place_ids"]) > idx else None,
                "analysis": self.result_data["analysis_results"][idx],
                "processing_time": 0,
                "note": "Retrieved from cache"
            }
        
        # If forcing re-analysis, remove previous entry
        if url in self.result_data["urls"] and force:
            logger.info(f"Forcing re-analysis of URL: {url}")
            idx = self.result_data["urls"].index(url)
            self.result_data["urls"].pop(idx)
            self.result_data["embeddings"].pop(idx)
            self.result_data["analysis_results"].pop(idx)
            if len(self.result_data["place_ids"]) > idx:
                self.result_data["place_ids"].pop(idx)
                
            # Reset FAISS index
            if self.faiss_loaded:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                if len(self.result_data["embeddings"]) > 0:
                    embeddings = np.vstack(self.result_data["embeddings"])
                    self.index.add(embeddings)
        
        # Load and process image
        image = self.load_image_from_url(url)
        if image is None:
            return {
                "url": url,
                "place_id": place_id,
                "error": "Failed to load image",
                "analysis": {},
                "processing_time": time.time() - start_time
            }
        
        # Get image features
        image_features = self.get_image_features(image)
        
        # Analyze across all categories
        logger.info("Analyzing image across all categories")
        analysis_results = {}
        
        for category, prompts in self.structured_prompts.items():
            # Analyze each category
            results = self.get_text_similarities(image_features, prompts, top_k=min(5, len(prompts)))
            
            # Apply additional normalization to increase contrast between scores
            if results:
                # Get max confidence
                max_conf = max(item["confidence"] for item in results)
                
                # Only keep results with confidence at least 70% of the max
                threshold = max_conf * 0.7
                results = [item for item in results if item["confidence"] >= threshold]
                
                # Rescale confidences to increase contrast
                if max_conf > 0:
                    for item in results:
                        # Apply sigmoid-like rescaling to emphasize differences
                        normalized = (item["confidence"] - threshold) / (max_conf - threshold)
                        item["confidence"] = float(round(normalized * 100, 2))
            
            analysis_results[category] = results
        
        # Add to FAISS index if available
        if self.faiss_loaded:
            try:
                self.index.add(image_features)
            except Exception as e:
                logger.error(f"Error adding to FAISS index: {e}")
        
        # Store results
        self.result_data["urls"].append(url)
        self.result_data["embeddings"].append(image_features)
        self.result_data["analysis_results"].append(analysis_results)
        self.result_data["place_ids"].append(place_id)
        
        # Save updated data
        self.save_data()
        
        processing_time = time.time() - start_time
        
        # Create a summary of top attributes across categories
        top_attributes = self.create_summary(analysis_results)
        
        return {
            "url": url,
            "place_id": place_id,
            "analysis": analysis_results,
            "summary": top_attributes,
            "processing_time": processing_time
        }
    
    def create_summary(self, analysis_results):
        """Create a summary of top attributes across all categories"""
        top_summary = {}
        
        for category, results in analysis_results.items():
            if results and len(results) > 0:
                # Just take the top result for the summary
                top_summary[category] = results[0]["tag"]
        
        return top_summary
    
    def find_similar_places(self, url, k=5):
        """Find similar places to the given URL"""
        if not self.faiss_loaded:
            return []
            
        # Check if URL exists in our database
        if url not in self.result_data["urls"]:
            logger.warning(f"URL not found in database: {url}")
            
            # Alternative: analyze the image first, then find similar
            image = self.load_image_from_url(url)
            if image is None:
                return []
                
            image_features = self.get_image_features(image)
        else:
            # Get the image features from database
            idx = self.result_data["urls"].index(url)
            image_features = self.result_data["embeddings"][idx]
        
        try:
            # Search FAISS index
            D, I = self.index.search(image_features, k+1)  # +1 because the image itself will be included
            
            # Remove the query image from results
            results = []
            for i, distance in zip(I[0], D[0]):
                if i < len(self.result_data["urls"]):
                    similar_url = self.result_data["urls"][i]
                    
                    # Skip if this is the query image
                    if similar_url == url:
                        continue
                        
                    # Get place_id if available
                    place_id = None
                    if len(self.result_data["place_ids"]) > i:
                        place_id = self.result_data["place_ids"][i]
                    
                    results.append({
                        "url": similar_url,
                        "place_id": place_id,
                        "distance": float(distance),
                        "similarity": 1.0 / (1.0 + float(distance))  # Convert distance to similarity score
                    })
                    
                    if len(results) >= k:
                        break
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar images: {e}")
            return []
    
    def batch_analyze_from_places_csv(self, csv_file, media_csv=None, limit=None, force=False):
        """Analyze images from places.csv with associated media"""
        try:
            # Load places CSV
            logger.info(f"Loading places data from {csv_file}")
            places_df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(places_df)} places")
            
            # If media CSV provided, use it to get image URLs
            if media_csv:
                logger.info(f"Loading media data from {media_csv}")
                media_df = pd.read_csv(media_csv)
                logger.info(f"Loaded {len(media_df)} media entries")
                
                # Join places with their first media item
                analysis_queue = []
                for _, place in places_df.iterrows():
                    place_id = place['place_id']
                    place_media = media_df[media_df['place_id'] == place_id]
                    
                    if not place_media.empty:
                        # Take the first media URL
                        media_url = place_media.iloc[0]['media_url']
                        if media_url:
                            analysis_queue.append({
                                'place_id': place_id,
                                'name': place['name'],
                                'url': media_url
                            })
            else:
                # Assume places CSV has image URLs directly
                analysis_queue = []
                for _, place in places_df.iterrows():
                    if 'image_url' in place and place['image_url']:
                        analysis_queue.append({
                            'place_id': place['place_id'],
                            'name': place['name'],
                            'url': place['image_url']
                        })
            
            # Apply limit if specified
            if limit and limit > 0:
                analysis_queue = analysis_queue[:limit]
                
            logger.info(f"Prepared {len(analysis_queue)} images for analysis")
            
            # Process each image
            results = []
            for idx, item in enumerate(analysis_queue):
                logger.info(f"Processing {idx+1}/{len(analysis_queue)}: {item['name']}")
                
                result = self.analyze_image(
                    url=item['url'],
                    place_id=item['place_id'],
                    force=force
                )
                
                results.append({
                    'place_id': item['place_id'],
                    'name': item['name'],
                    'analysis': result
                })
                
                # Add a small delay to prevent rate limiting
                time.sleep(0.5)
            
            logger.info(f"Completed analysis of {len(results)} places")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def generate_attributes_report(self, output_file="place_attributes_report.csv"):
        """Generate a CSV report with places and their top attributes"""
        try:
            if not self.result_data["urls"]:
                logger.warning("No analysis data available for report")
                return False
                
            # Prepare data for report
            report_data = []
            
            for idx, url in enumerate(self.result_data["urls"]):
                # Get place_id if available
                place_id = self.result_data["place_ids"][idx] if idx < len(self.result_data["place_ids"]) else None
                
                # Get analysis results
                analysis = self.result_data["analysis_results"][idx]
                
                # Create row with place info
                row = {"url": url, "place_id": place_id}
                
                # Add top attribute from each category to the row
                for category, results in analysis.items():
                    if results and len(results) > 0:
                        category_key = f"{category}"
                        row[category_key] = results[0]["tag"]
                        row[f"{category}_confidence"] = results[0]["confidence"]
                
                report_data.append(row)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(report_data)
            df.to_csv(output_file, index=False)
            logger.info(f"Report saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False
            
    def enrich_places_csv(self, input_csv, output_csv, media_csv=None):
        """
        Enhance places.csv with visual attributes detected from images
        Args:
            input_csv: Path to input places CSV
            output_csv: Path to write enriched CSV
            media_csv: Optional path to media CSV with place images
        """
        try:
            logger.info(f"Loading data from {input_csv}")
            places_df = pd.read_csv(input_csv)
            
            # Get image URLs for places
            if media_csv:
                media_df = pd.read_csv(media_csv)
                
                # Create lookup of first image for each place
                place_images = {}
                for _, row in media_df.iterrows():
                    place_id = row['place_id']
                    if place_id not in place_images and pd.notna(row['media_url']):
                        place_images[place_id] = row['media_url']
            else:
                # Assume image URLs are in places_df
                place_images = {}
                if 'image_url' in places_df.columns:
                    for _, row in places_df.iterrows():
                        if pd.notna(row['place_id']) and pd.notna(row['image_url']):
                            place_images[row['place_id']] = row['image_url']
            
            # Add empty columns for visual attributes
            visual_columns = []
            for category in self.categories.keys():
                col_name = f"visual_{category}"
                places_df[col_name] = None
                visual_columns.append(col_name)
            
            # Add confidence columns
            confidence_columns = []
            for category in self.categories.keys():
                col_name = f"visual_{category}_confidence"
                places_df[col_name] = None
                confidence_columns.append(col_name)
                
            # Add detected vibe tags
            places_df['detected_vibes'] = None
            
            # Process each place with an image
            processed_count = 0
            for idx, row in places_df.iterrows():
                place_id = row['place_id']
                
                if place_id in place_images:
                    url = place_images[place_id]
                    
                    # Check if we've already analyzed this image
                    if url in self.result_data["urls"]:
                        result_idx = self.result_data["urls"].index(url)
                        analysis = self.result_data["analysis_results"][result_idx]
                        
                        # Update the dataframe with analysis results
                        for category, results in analysis.items():
                            if results and len(results) > 0:
                                col_name = f"visual_{category}"
                                conf_name = f"visual_{category}_confidence"
                                
                                places_df.at[idx, col_name] = results[0]["tag"]
                                places_df.at[idx, conf_name] = results[0]["confidence"]
                        
                        # Extract vibe tags specifically
                        if 'vibes' in analysis and analysis['vibes']:
                            vibes = [v["tag"] for v in analysis['vibes']]
                            places_df.at[idx, 'detected_vibes'] = ", ".join(vibes)
                            
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logger.info(f"Processed {processed_count} places")
            
            logger.info(f"Enriched {processed_count} places with visual attributes")
            
            # Save enriched CSV
            places_df.to_csv(output_csv, index=False)
            logger.info(f"Saved enriched places data to {output_csv}")
            
            return True
        except Exception as e:
            logger.error(f"Error enriching places CSV: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def generate_vibe_tags_for_places(self, output_file="place_vibes.csv"):
        """Generate a CSV with place_id and associated vibe tags"""
        try:
            if not self.result_data["urls"]:
                logger.warning("No analysis data available for vibes")
                return False
                
            # Prepare data for vibes report
            vibes_data = []
            
            for idx, url in enumerate(self.result_data["urls"]):
                # Get place_id if available
                place_id = self.result_data["place_ids"][idx] if idx < len(self.result_data["place_ids"]) else None
                
                if not place_id:
                    continue
                
                # Get analysis results
                analysis = self.result_data["analysis_results"][idx]
                
                # Extract vibes specifically
                if 'vibes' in analysis and analysis['vibes']:
                    for vibe in analysis['vibes']:
                        vibes_data.append({
                            "place_id": place_id,
                            "vibe_tag": vibe["tag"],
                            "confidence": vibe["confidence"]
                        })
            
            # Convert to DataFrame and save
            if vibes_data:
                df = pd.DataFrame(vibes_data)
                df.to_csv(output_file, index=False)
                logger.info(f"Vibes report saved to {output_file}")
                return True
            else:
                logger.warning("No vibe data found in analyzed images")
                return False
                
        except Exception as e:
            logger.error(f"Error generating vibes report: {e}")
            return False
            
    def generate_similar_places_recommendations(self, output_dir="similar_places"):
        """Generate similarity-based place recommendations"""
        try:
            if not self.result_data["urls"] or not self.faiss_loaded:
                logger.warning("No data available for similarity recommendations")
                return False
                
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Find similar places for each place
            all_similarities = {}
            
            for idx, url in enumerate(self.result_data["urls"]):
                place_id = self.result_data["place_ids"][idx] if idx < len(self.result_data["place_ids"]) else None
                
                if not place_id:
                    continue
                    
                # Find similar places
                similar_places = self.find_similar_places(url, k=10)
                
                if similar_places:
                    # Only keep entries with place_id
                    similar_with_ids = [p for p in similar_places if p.get('place_id')]
                    
                    if similar_with_ids:
                        all_similarities[place_id] = similar_with_ids
            
            # Save all similarities to JSON
            json_path = os.path.join(output_dir, "all_similar_places.json")
            with open(json_path, 'w') as f:
                json.dump(all_similarities, f, indent=2)
                
            # Create CSV format
            csv_data = []
            for place_id, similars in all_similarities.items():
                for similar in similars:
                    csv_data.append({
                        "source_place_id": place_id,
                        "similar_place_id": similar["place_id"],
                        "similarity_score": similar["similarity"]
                    })
            
            # Save CSV
            if csv_data:
                csv_path = os.path.join(output_dir, "similar_places.csv")
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved similarity recommendations to {csv_path}")
                
            logger.info(f"Generated similarity recommendations for {len(all_similarities)} places")
            return True
            
        except Exception as e:
            logger.error(f"Error generating similarity recommendations: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def export_vibe_embeddings(self, output_file="vibe_embeddings.npy", metadata_file="vibe_metadata.json"):
        """Export place embeddings for use in external systems"""
        try:
            if not self.result_data["urls"]:
                logger.warning("No data available for embeddings export")
                return False
                
            # Collect embeddings and metadata
            embeddings = []
            metadata = []
            
            for idx, url in enumerate(self.result_data["urls"]):
                place_id = self.result_data["place_ids"][idx] if idx < len(self.result_data["place_ids"]) else None
                
                if not place_id:
                    continue
                    
                embeddings.append(self.result_data["embeddings"][idx])
                
                # Get top attributes for metadata
                analysis = self.result_data["analysis_results"][idx]
                top_attrs = self.create_summary(analysis)
                
                metadata.append({
                    "place_id": place_id,
                    "url": url,
                    "top_attributes": top_attrs
                })
            
            if embeddings:
                # Save embeddings
                embeddings_array = np.vstack(embeddings)
                np.save(output_file, embeddings_array)
                
                # Save metadata
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                logger.info(f"Exported {len(embeddings)} place embeddings to {output_file}")
                logger.info(f"Exported metadata to {metadata_file}")
                return True
            else:
                logger.warning("No valid embeddings to export")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
            logger.error(traceback.format_exc())
            return False

def main():
    parser = argparse.ArgumentParser(description="Analyze place images using CLIP")
    parser.add_argument("--url", type=str, help="URL of the image to analyze")
    parser.add_argument("--place_id", type=str, help="Optional place ID to associate with the image")
    parser.add_argument("--find_similar", action="store_true", help="Find similar places to the given URL")
    parser.add_argument("--force", action="store_true", help="Force re-analysis of the image even if previously analyzed")
    parser.add_argument("--clear", action="store_true", help="Clear all stored data before analysis")
    parser.add_argument("--batch", action="store_true", help="Perform batch analysis from places.csv")
    parser.add_argument("--places_csv", type=str, default="places.csv", help="Path to places CSV file")
    parser.add_argument("--media_csv", type=str, default=None, help="Path to media CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of places to analyze in batch mode")
    parser.add_argument("--report", action="store_true", help="Generate attributes report")
    parser.add_argument("--vibes_report", action="store_true", help="Generate vibe tags report")
    parser.add_argument("--enrich", action="store_true", help="Enrich places CSV with visual attributes")
    parser.add_argument("--enriched_output", type=str, default="places_enriched.csv", help="Output path for enriched CSV")
    parser.add_argument("--export_embeddings", action="store_true", help="Export place embeddings")
    parser.add_argument("--recommendations", action="store_true", help="Generate place recommendations")
    parser.add_argument("--output", type=str, default="place_clip_analysis_data.pkl", help="Output file path")
    parser.add_argument("--max_width", type=int, default=400, help="Maximum image width for processing")
    parser.add_argument("--max_height", type=int, default=400, help="Maximum image height for processing")
    
    args = parser.parse_args()
    
    # Initialize analyzer with custom image size limits
    analyzer = PlaceImageAnalyzer(max_image_size=(args.max_width, args.max_height))
    
    # Clear data if requested
    if args.clear:
        analyzer.clear_data()
    
    if args.find_similar and args.url:
        similar_places = analyzer.find_similar_places(args.url)
        logger.info("\n===== Similar Places =====")
        for i, place in enumerate(similar_places):
            place_id_info = f"(place_id: {place['place_id']})" if place['place_id'] else ""
            logger.info(f"{i+1}. {place['url']} {place_id_info} (similarity: {place['similarity']:.2f})")
    
    elif args.batch:
        logger.info("Starting batch analysis of places")
        analyzer.batch_analyze_from_places_csv(
            csv_file=args.places_csv,
            media_csv=args.media_csv,
            limit=args.limit,
            force=args.force
        )
        
    elif args.report:
        logger.info("Generating attributes report")
        analyzer.generate_attributes_report()
        
    elif args.vibes_report:
        logger.info("Generating vibe tags report")
        analyzer.generate_vibe_tags_for_places()
        
    elif args.enrich:
        logger.info("Enriching places CSV with visual attributes")
        analyzer.enrich_places_csv(
            input_csv=args.places_csv,
            output_csv=args.enriched_output,
            media_csv=args.media_csv
        )
        
    elif args.export_embeddings:
        logger.info("Exporting place embeddings")
        analyzer.export_vibe_embeddings()
        
    elif args.recommendations:
        logger.info("Generating place recommendations")
        analyzer.generate_similar_places_recommendations()
        
    elif args.url:
        # Analyze the image
        results = analyzer.analyze_image(args.url, place_id=args.place_id, force=args.force)
        
        # Check for errors
        if "error" in results:
            logger.error(f"Analysis failed: {results['error']}")
            sys.exit(1)
        
        # Display results
        logger.info("\n===== Place Image Analysis Results =====")
        logger.info(f"URL: {results['url']}")
        if results['place_id']:
            logger.info(f"Place ID: {results['place_id']}")
        logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
        
        logger.info("\n===== Place Summary =====")
        for category, attribute in results.get('summary', {}).items():
            logger.info(f"{category.replace('_', ' ').title()}: {attribute}")
        
        logger.info("\n===== Detailed Analysis =====")
        for category, attributes in results["analysis"].items():
            logger.info(f"\n{category.replace('_', ' ').title()}:")
            for i, attr in enumerate(attributes):
                logger.info(f"  {i+1}. {attr['tag']} ({attr['confidence']:.2f}%)")
    
    else:
        parser.print_help()
        logger.info("Please provide an action (--url, --batch, --report, etc.)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

"""
STEP:1 [MANDATORY]
Start command for processing: python3 place_image_analyzer.py --batch --places_csv "yourFileLocation" --media_csv "yourFileLocation" 
(Took me around 15 minutes on CPU(because mine is a MAC) for the 1500 places, but if you'd use a GPU based machine - a 4090 - you should be able to create the .pkl file in way under 4 minutes)
This defaultly generates the "place_clip_analysis_data.pkl" file!

STEP:2 (if you'd like a report); [OPTIONAL]
Then run this command for report generation: python3 place_image_analyzer.py --report
Saves to `place_attributes_report.csv` by default!

STEP:3 (if you'd like a csv with the correct attributes (Really GOOD); [OPTIONAL]
This is required for the main normalised data: 
```
python3 place_image_analyzer.py --enrich --places_csv "places.csv" --media_csv "media.csv" --enriched_output "places_with_visuals.csv"
```
It saves to places_with_visuals.csv which you can change and then see the outputs which are really _perfecto_!
"""

"""
Basic Commands (for better understanding the content (Provides beautiful analysis of your imageFiles))

1)    --url [URL]: Analyzes a single image from the specified URL

    Example: python place_image_analyzer_complete.py --url "https://example.com/image.jpg"
    Runs a one-off analysis on an individual image


2)    --place_id [ID]: Associates the analyzed image with a specific place ID

    Example: python place_image_analyzer_complete.py --url "https://example.com/image.jpg" --place_id "place123"
    Links the analysis to a place ID in your database


3)   --force: Forces re-analysis of images even if they've been previously analyzed

    Example: python place_image_analyzer_complete.py --url "https://example.com/image.jpg" --force
    Overwrites previous analysis results for an image


4)    --clear: Clears all stored data before starting a new analysis

    Example: python place_image_analyzer_complete.py --clear --batch
    Removes all previous analyses and starts fresh



Batch Processing Commands

5)    --batch: Processes multiple places from a CSV file

    Example: python place_image_analyzer_complete.py --batch
    Analyzes all places from places.csv (and their associated images)


6)    --places_csv [FILE]: Specifies the input places CSV file (default: "places.csv")

    Example: python place_image_analyzer_complete.py --batch --places_csv "my_places.csv"
    Uses a custom places CSV file for batch processing


7)    --media_csv [FILE]: Specifies the media CSV file with image URLs

    Example: python place_image_analyzer_complete.py --batch --places_csv "places.csv" --media_csv "media.csv"
    Uses a separate file containing image URLs for each place


8)    --limit [NUMBER]: Limits batch processing to a specific number of places

    Example: python place_image_analyzer_complete.py --batch --limit 100
    Only processes the first 100 places (useful for testing)



Output and Report Commands

9)    --report: Generates a comprehensive attributes report (what you've already run)

    Example: python place_image_analyzer_complete.py --report
    Creates place_attributes_report.csv with all attributes and confidence scores


10)    --vibes_report: Generates a report focused on vibe tags

    Example: python place_image_analyzer_complete.py --vibes_report
    Creates place_vibes.csv with detected vibes for each place


11)    --enrich: Enhances your original places.csv with visual attributes

    Example: python place_image_analyzer_complete.py --enrich
    Adds visual attribute columns to your places data


12)    --enriched_output [FILE]: Sets the output file for enriched CSV (default: "places_enriched.csv")

    Example: python place_image_analyzer_complete.py --enrich --enriched_output "places_with_visuals.csv"
    Saves the enriched dataset to a custom filename


13)    --export_embeddings: Exports place embeddings for use in external systems

    Example: python place_image_analyzer_complete.py --export_embeddings
    Creates vibe_embeddings.npy and vibe_metadata.json files


14)    --recommendations: Generates place recommendations based on visual similarity

    Example: python place_image_analyzer_complete.py --recommendations
    Creates a similar_places.csv file with place pairs and similarity scores


15)    --output [FILE]: Sets the main output file (default: "place_clip_analysis_data.pkl")

    Example: python place_image_analyzer_complete.py --batch --output "my_analysis.pkl"
    Changes the name of the main storage file



Image Settings

16)    --max_width [PIXELS]: Sets maximum image width for processing (default: 400)

    Example: python place_image_analyzer_complete.py --batch --max_width 600
    Uses larger images for potentially better analysis (but slower)


17)    --max_height [PIXELS]: Sets maximum image height for processing (default: 400)

    Example: python place_image_analyzer_complete.py --batch --max_height 600
    Works alongside max_width to control image size



Finding Similar Places

18)    --find_similar: Finds visually similar places to a given image URL

    Example: python place_image_analyzer_complete.py --url "https://example.com/image.jpg" --find_similar
    Returns a list of similar places based on visual features
"""