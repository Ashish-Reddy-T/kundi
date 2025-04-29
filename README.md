# Vibe Search™ - Multimodal RAG-Based Place Recommender

A sophisticated place recommendation system built for the lovely _DATATHON_. This implementation uses modern Retrieval Augmented Generation (RAG) techniques with multimodal understanding to find places in NYC that match specific "vibes" and intangible qualities.

---

## Key Features

- **Multimodal Understanding**: Combines both textual and visual analysis to truly understand place characteristics
- **Visual Attribute Detection**: Analyzes images to detect atmospheres, lighting, colors, settings, and more
- **Semantic Understanding**: Capture the true meaning behind queries like "where to find hot guys" or "cafes to cowork from"
- **True RAG Implementation**: Uses modern embedding models and visual CLIP analysis for nuanced understanding
- **Vibe Detection**: Automatically identifies vibes and atmospheres from both text and visual data
- **Contextual Explanations**: Explains why each place matches your query across both text and visual dimensions
- **Fast Results**: Optimized to deliver results quickly despite complex processing
- **Beautiful UI**: Modern, responsive interface to display results

---

## Start

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Ashish-Reddy-T/kundi.git
cd kundi

# ![PLEASE MOVE THE `places.csv`, `media.csv` and `reviews.csv` files into this folder]!

# Direct Run (Will install dependencies)
python rag_run.py --build-index --auto-yes # (Uses bge-large and 0.7:text_ratio :: 0.3:image_ratio)
```

### Complete Implementation Steps

For new users, follow these steps to set up the full pipeline:

1. **Analyze images with the CLIP-based analyzer**:
```bash
# Process all place images and generate visual attributes
python place_image_analyzer.py --batch --places_csv "places.csv" --media_csv "media.csv"

# Generate a report of visual attributes (optional)
python place_image_analyzer.py --report

# Export embeddings (optional)
python place_image_analyzer.py --export_embeddings
```

2. **Generate image embeddings for RAG**:
```bash
# Use the enhanced script with the analyzed data
python ingestion/rag_index_images.py --use_existing --analysis_path "place_clip_analysis_data.pkl"
```

3. **Generate text embeddings with visual attributes**:
```bash
# Generate text embeddings incorporating visual attributes
python ingestion/rag_index.py --model bge-large
```

4. **Combine text and image embeddings**:
```bash
# Combine with tuned weights
python ingestion/rag_index_combine.py --model bge-large --text_weight 0.7 --image_weight 0.3
# Weights can be altered according to your will (range:[0,1])
```

5. **Run the RAG application**:
```bash
# Run the application with the combined data
python run_rag.py --embedding-model bge-large
```

6. **Access the web interface**:
Open your browser and go to: http://localhost:8000

### Quick Run (After Initial Setup)

Once the initial setup is complete, you can simply run:

```bash
python run_rag.py --embedding-model bge-large
```

---


## Technologies Used

### Multimodal Understanding
- **CLIP Visual Analysis**: Deep visual understanding of places
  - Detects visual attributes like atmosphere, setting, colors, lighting
  - Generates consistent visual embeddings
  - Enhances search with visual context
- **Combined Text-Visual Embeddings**: Weighted fusion of textual and visual understanding

### Embedding Models
- **Sentence Transformers**: Modern semantic embedding models
  - Options: MiniLM, BGE, MPNet
  - Captures nuanced meaning in text

### Vector Database
- **FAISS**: High-performance similarity search 
  - Fast retrieval even with thousands of items
  - Efficient cosine similarity calculations

### Visual Analysis
- **PlaceImageAnalyzer**: Advanced CLIP-based image analysis
  - Detects 11 categories of visual attributes
  - Generates detailed atmospheric understanding
  - Maps visual attributes to vibes

### LLM Integration
- **LangChain**: Framework for LLM applications
  - Structured prompts for context and explanations
  - Support for multiple LLM providers

### Optional LLMs
- **Local LLMs**: Via llama-cpp-python
  - Works offline with models like Llama, Mistral, etc.
- **OpenAI API**: For enhanced explanations
  - Set OPENAI_API_KEY in .env file to use
 
---

## Architecture

This system follows a sophisticated multimodal RAG architecture:

1. **Indexing Phase**:
   - Process place data, reviews, and media
   - Analyze images with CLIP for visual attributes
   - Extract vibe attributes from both text and images
   - Generate high-quality text and image embeddings
   - Combine embeddings with appropriate weights
   - Build optimized vector index

2. **Retrieval Phase**:
   - Process user query with semantic understanding
   - Expand query to capture related concepts
   - Retrieve relevant places using multimodal vector similarity
   - Apply contextual filtering (neighborhoods, vibes)

3. **Generation Phase**:
   - Generate explanations for why places match the query
   - Provide context about each place's vibe and atmosphere
   - Create a cohesive, helpful response highlighting both visual and textual matches

---

## How Vibe Search Works

### 1. Multimodal Ingestion & Embedding
The system processes place data from multiple sources:
- **Structured Data**: Name, location, tags, etc.
- **Unstructured Data**: Reviews, descriptions
- **Visual Data**: Images analyzed with CLIP to detect visual attributes
  - Place type (restaurant, cafe, bar, etc.)
  - Setting (indoor, outdoor, rooftop, etc.)
  - Atmosphere (upscale, casual, romantic, etc.)
  - Lighting (bright, dim, candlelit, etc.)
  - Colors and materials
  - Furniture and decor
  - View characteristics
  - Crowd dynamics
  - Time context
  - Food and drink focus
  - Overall vibes

This data is processed to extract vibe attributes from both text and images, which are combined with the original data and embedded using state-of-the-art models.

### 2. Query Understanding & Expansion
When you search for something like "cafes to cowork from", the system:
- **Understands the Concept**: Recognizes "coworking" implies wifi, quiet, outlets, etc.
- **Expands the Query**: Adds related terms to improve results
- **Extracts Constraints**: Identifies location filters, price ranges, etc.

### 3. Multimodal Similarity Search & Filtering
The system searches for places that match the expanded query:
- **Vector Similarity**: Finds places with similar text AND visual characteristics
- **Balanced Retrieval**: Weights visual and textual importance appropriately
- **Filtering**: Applies neighborhood and vibe filters
- **Ranking**: Orders results by relevance across both modalities

### 4. Explanation Generation
For each result, the system generates an explanation:
- **Contextual Understanding**: Why this place matches your query
- **Visual Highlights**: Emphasizes relevant visual attributes detected
- **Text Highlights**: Features from descriptions and reviews
- **Natural Language**: Presents information conversationally

---

## Advanced Usage

### Command Line Options

```bash

# Run with specific embedding model
python run_rag.py --build-index --embedding-model minilm

# Adjust text vs image importance
python ingestion/rag_index_combine.py --model minilm --text_weight 0.7 --image_weight 0.3

# Run on specific port
python run_rag.py --port 8080
```

### Visual Analysis Options

```bash
# Generate visual embeddings using existing analysis
python ingestion/rag_index_images.py --use_existing --analysis_path "place_clip_analysis_data.pkl"

# Analyze images with specific settings
python place_image_analyzer.py --batch --places_csv "places.csv" --media_csv "media.csv" --max_width 600 --max_height 600

# Generate visual attribute reports
python place_image_analyzer.py --report
python place_image_analyzer.py --vibes_report
```

### Environment Variables

Create a `.env` file to configure:

```
EMBEDDING_MODEL=bge-large
LLM_PROVIDER=local
OPENAI_API_KEY=your_key_here
```

---

## Example Queries

- "cafes to cowork from"
- "matcha latte in the east village"
- "where can I spend a sunny day?"
- "romantic restaurants with dim lighting"
- "dance-y bars that have disco balls"
- "restaurants with outdoor seating and string lights"
- "cozy cafes with warm ambient lighting"
- "upscale cocktail bars with a view"

---

## Technical Details

### Embedding Models

This implementation supports multiple embedding models:

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| MiniLM | 384 | ⚡⚡⚡ | ⭐⭐ |
| BGE-Small | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| BGE-Base | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
| BGE-Large | 1024 | ⚡ | ⭐⭐⭐⭐⭐ |
| MPNet | 768 | ⚡⚡ | ⭐⭐⭐⭐ |

- Though strictly speaking, there is no real difference for this dataset; significant changes can only be seen if the dataset is altered!
  
Ex: `bge-large` for us ran queries under 0.1 seconds most of the times

### Visual Analysis Categories

The system analyzes images across 11 categories of visual attributes:

1. **Place Type**: restaurant, cafe, bar, etc.
2. **Setting**: indoor, outdoor, rooftop, waterfront, etc.
3. **Atmosphere**: upscale, casual, romantic, lively, etc.
4. **Lighting**: bright, dim, candlelit, colored, etc.
5. **Colors & Materials**: wood, brick, colorful, monochrome, etc.
6. **Furniture & Decor**: modern, vintage, minimalist, etc.
7. **View**: skyline, water, garden, none, etc.
8. **Crowd**: empty, sparse, filled, busy, etc.
9. **Time Context**: daytime, evening, golden hour, etc.
10. **Food & Drink Focus**: plated food, cocktails, coffee, etc.
11. **Vibes**: date night, group hangout, instagram-worthy, etc.

### Vibe Categories

The system automatically categorizes places into vibes from both text and visual cues:

- date_night
- work_friendly
- outdoor_vibes
- group_hangout
- food_focus
- drinks_focus
- coffee_tea
- dancing_music
- quiet_relaxing
- upscale_fancy
- casual_lowkey
- unique_special
- trendy_cool
- budget_friendly

---

## Troubleshooting

### "Unexpected text embedding size" Error
If you see this error when using a different embedding model, you may need to update the `embed_text` function in `rag_app.py` to handle the dimensionality of that specific embedding model.

### Image Processing Issues
If you encounter issues with image processing:
- Check that your `places.csv` and `media.csv` have the correct format
- Try reducing the `--max_width` and `--max_height` parameters
- Ensure you have adequate memory for CLIP model loading

### FAISS Index Issues
If the FAISS index fails to build or search:
- Check that the text and image embeddings match in count
- Try rebuilding the index with the `--build-index` option
- Ensure the embedding dimensions match what's expected in `rag_app.py`

---

## Performance Optimizations

1. **Vectorization**: Fast similarity search with FAISS
2. **Cached Embeddings**: Reuse query embeddings for similar searches
3. **Cached Explanations**: Store explanations for common patterns
4. **Timeout Handling**: Enforce limits on search and explanation time
5. **Batched Processing**: Process data in optimal batches
6. **Weighted Embeddings**: Balance text and visual importance
7. **Image Resizing**: Process images at optimal dimensions for speed/quality balance

---

_Made with LOVE from the __VibeLabs__ team for the wonderful sponsor __CORNER___ 💖
