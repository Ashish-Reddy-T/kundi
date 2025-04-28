"""
Wrapper for Ollama API integration for Vibe Search
Provides a simple client for using Ollama models for query expansion and explanation
"""
import requests
import json
import time
import logging
from typing import List, Dict, Optional, Union
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Simple client for Ollama API with error handling and fallbacks"""
    
    def __init__(self, model: str = "llama3:latest", base_url: str = "http://localhost:11434",
                timeout: float = 2.0):
        """Initialize the Ollama client
        
        Args:
            model: Model identifier to use (default llama3:latest)
            base_url: Ollama API base URL
            timeout: Default timeout for requests in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.cache = {}
        
        # Test connection
        try:
            self._send_request("Hi, test")
            logger.info(f"Ollama client initialized with model {model}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.warning("Ollama not available. Using fallback methods.")
    
    @lru_cache(maxsize=100)
    def _send_request(self, prompt: str) -> str:
        """Send a request to Ollama API with error handling
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated text response
        """
        start_time = time.time()
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            elapsed = time.time() - start_time
            logger.debug(f"Ollama request completed in {elapsed:.2f}s")
            
            return result
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out")
            return ""
        except Exception as e:
            logger.error(f"Error in Ollama request: {e}")
            return ""
    
    def expand_search_query(self, query: str) -> str:
        """Expand a search query with relevant terms
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query with relevant terms
        """
        # Return from cache if exists
        cache_key = f"expand:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Define a clear, structured prompt
        prompt = f"""You are a helpful assistant that expands search queries to improve semantic search.
Given the query: "{query}"
Generate 5-10 related keywords that would help find relevant places in New York City. 
Focus on vibes, atmosphere, and qualities that match this query.
Return ONLY the keywords separated by spaces, no explanations or formatting:"""
        
        try:
            # Call the model
            result = self._send_request(prompt)
            
            # Clean and validate result
            terms = result.strip().lower().split()
            expanded = " ".join(terms[:15])  # Limit to 15 terms
            
            # Cache and return
            self.cache[cache_key] = expanded
            return expanded
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            # Use rule-based fallbacks
            fallbacks = {
                "coffee": "cafe relaxing cozy wifi",
                "date": "romantic intimate dinner",
                "bar": "drinks cocktails nightlife"
            }
            for key, expansion in fallbacks.items():
                if key in query.lower():
                    return expansion
            return ""
            
    def analyze_search_results(self, query: str, results: List[Dict]) -> Dict[str, str]:
        """Generate explanations for search results
        
        Args:
            query: Original search query
            results: List of search results
            
        Returns:
            Dictionary mapping place names to explanations
        """
        # Limited to top 5 results to reduce load
        results = results[:5]
        
        # Return from cache if exists
        cache_key = f"explain:{query}:{len(results)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare minimal context from results
        context = []
        for i, place in enumerate(results):
            name = place.name if hasattr(place, 'name') else place.get('name', f"Place {i}")
            hood = place.neighborhood if hasattr(place, 'neighborhood') else place.get('neighborhood', '')
            desc = place.short_description if hasattr(place, 'short_description') else place.get('short_description', '')
            vibe_tags = place.vibe_tags if hasattr(place, 'vibe_tags') else place.get('vibe_tags', [])
            
            context.append(f"Place {i+1}: {name} in {hood}. {desc}. Vibes: {', '.join(vibe_tags)}")
        
        # Define a clear, structured prompt
        prompt = f"""The user searched for: "{query}"

Here are some matching places:
{chr(10).join(context)}

For each place, generate a short one-sentence explanation (10-15 words) for why it matches their search.
Format as a JSON object with place names as keys and explanations as values. Just return the JSON with no other text."""
        
        try:
            # Call the model with a longer timeout
            result = self._send_request(prompt)
            
            # Try to parse JSON response
            try:
                explanation_dict = json.loads(result)
                # Clean up the keys to match place names
                clean_dict = {}
                for place in results:
                    name = place.name if hasattr(place, 'name') else place.get('name', '')
                    # Find closest matching key
                    best_match = name
                    for key in explanation_dict.keys():
                        if name.lower() in key.lower() or key.lower() in name.lower():
                            best_match = key
                            break
                    if best_match in explanation_dict:
                        clean_dict[name] = explanation_dict[best_match]
                
                # Cache and return
                self.cache[cache_key] = clean_dict
                return clean_dict
            except json.JSONDecodeError:
                # Fall back to simple parsing
                lines = result.split('\n')
                explanation_dict = {}
                for place in results:
                    name = place.name if hasattr(place, 'name') else place.get('name', '')
                    for line in lines:
                        if name in line and ":" in line:
                            _, explanation = line.split(":", 1)
                            explanation_dict[name] = explanation.strip().strip('"')
                            break
                
                # Cache and return
                self.cache[cache_key] = explanation_dict
                return explanation_dict
                
        except Exception as e:
            logger.error(f"Result analysis error: {e}")
            return {}
            
    def analyze_image(self, image_url: str) -> List[str]:
        """Extract vibe terms from an image
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            List of vibe terms extracted from the image
        """
        # Return from cache if exists
        cache_key = f"image:{image_url}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # This is a dummy implementation since Ollama doesn't support image analysis
        # For a real implementation, you would use a multimodal model or separate API
        logger.warning("Image analysis not supported in current implementation")
        
        # Return common vibe terms based on keywords in the URL
        url_lower = image_url.lower()
        image_terms = []
        
        vibe_mappings = {
            "restaurant": ["dining", "food", "eating"],
            "cafe": ["coffee", "cafe", "casual"],
            "bar": ["drinks", "nightlife", "social"],
            "outdoor": ["patio", "outdoor", "sunny"],
            "cozy": ["intimate", "warm", "comfortable"],
            "modern": ["trendy", "contemporary", "stylish"]
        }
        
        for key, terms in vibe_mappings.items():
            if key in url_lower:
                image_terms.extend(terms)
                
        # Cache and return
        result = image_terms[:5]  # Limit to 5 terms
        self.cache[cache_key] = result
        return result