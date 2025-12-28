#!/usr/bin/env python3
"""
AI News Scraper using Crawl4AI and Claude as specified in functional requirements.
Implements the exact scraping process: Crawl4AI -> Claude -> Structured Output
"""

import os
import sys
import re
import json
import random
import asyncio
import hashlib
import logging
import tempfile
import traceback  # ✅ ADD THIS MISSING IMPORT
import requests
import feedparser
import aiohttp
from bs4 import BeautifulSoup
import ollama  # ✅ NEW: Ollama import

# Configure logging early to avoid import issues
logging.basicConfig(level=logging.INFO)

# Detect Railway deployment environment
IS_RAILWAY_DEPLOYMENT = (
    os.environ.get('RAILWAY_ENVIRONMENT') or 
    os.environ.get('RAILWAY_PROJECT_ID') or 
    os.environ.get('RAILWAY_SERVICE_ID') or
    '/home/appuser' in os.path.expanduser('~') or
    os.path.exists('/app') or  # Common Railway app directory
    'railway' in os.environ.get('HOSTNAME', '').lower()
)

# For Railway deployment, force fallback mode to avoid permission issues
if IS_RAILWAY_DEPLOYMENT:
    CRAWL4AI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("🚂 Railway deployment detected - using fallback scraping mode")
else:
    # Set up writable directories for local/other deployments
    temp_base_dir = tempfile.mkdtemp(prefix='crawl4ai_')
    os.environ['CRAWL4AI_CACHE_DIR'] = temp_base_dir
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = os.path.join(temp_base_dir, 'browsers')
    os.environ['CRAWL4AI_BASE_DIRECTORY'] = temp_base_dir

import json
import asyncio
import hashlib
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp

# Configure logging
logger = logging.getLogger(__name__)

# Crawl4AI imports - only for non-Railway environments
if not IS_RAILWAY_DEPLOYMENT:
    try:
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, CosineStrategy
        from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
        CRAWL4AI_AVAILABLE = True
        logger.info(f"✅ Crawl4AI available - using cache dir: {temp_base_dir}")
    except ImportError as e:
        CRAWL4AI_AVAILABLE = False
        logger.warning(f"⚠️ Crawl4AI not installed - falling back to basic scraping: {e}")
else:
    # Set dummy variables for Railway deployment
    temp_base_dir = "/tmp"
    CRAWL4AI_AVAILABLE = False

@dataclass
class ScrapedArticle:
    """Structured article data from scraping"""
    title: str
    author: Optional[str]
    summary: str
    content: str
    date: Optional[str]
    url: str
    source: str
    significance_score: float = 5.0
    complexity_level: str = "Medium"
    reading_time: int = 1
    published_date: Optional[str] = None
    scraped_date: Optional[str] = None
    content_type_label: str = "article"
    topic_category_label: str = "Generative AI"
    keywords: Optional[List[str]] = None
    publisher_id: Optional[int] = None
    llm_processed: Optional[str] = None
    image_url: Optional[str] = None          # ✅ ADD THIS
    image_source: Optional[str] = None       # ✅ ADD THIS

class Crawl4AIScraper:
    """AI News Scraper using Crawl4AI and Claude LLM"""
    
    def __init__(self):
        self.extraction_warnings = []  # Track extraction warnings for reporting
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.claude_api_url = "https://api.anthropic.com/v1/messages"

        # ✅ NEW: Ollama Configuration
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')  # Default to Llama 3.2 3B
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')  # Default Ollama host
        
        # ✅ FIX: Use a verified HuggingFace model that's available on Inference API
        # Popular models on HuggingFace Inference API (free tier):
        # - mistralai/Mistral-7B-Instruct-v0.3 (updated version)
        # - meta-llama/Meta-Llama-3-8B-Instruct (requires approval)
        # - HuggingFaceH4/zephyr-7b-beta (open, works well)
        # - microsoft/Phi-3-mini-4k-instruct (smaller, faster)
        
        # Using Mistral v0.3 which is more widely available
        self.huggingface_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.huggingface_api_url = f"https://api-inference.huggingface.co/models/{self.huggingface_model_name}"

        logger.info(f"🤗 HuggingFace Model: {self.huggingface_model_name}")
        logger.info(f"🌐 HuggingFace URL: {self.huggingface_api_url}")
        logger.info(f"   API Key: {'✅ Set' if self.huggingface_api_key else '❌ Missing'}")

        self.headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Debug: Log API key status
        logger.info(f"🔑 API Keys Status:")
        logger.info(f"   - ANTHROPIC_API_KEY: {'✅ Set' if self.anthropic_api_key else '❌ Not set'}")
        logger.info(f"   - GOOGLE_API_KEY: {'✅ Set' if self.google_api_key else '❌ Not set'}")
        logger.info(f"   - HUGGINGFACE_API_KEY: {'✅ Set' if self.huggingface_api_key else '❌ Not set'}")
        logger.info(f"🦙 Ollama Model: {self.ollama_model}")
        logger.info(f"🌐 Ollama Host: {self.ollama_host}")

        self._test_ollama_connection()
        
        if not self.anthropic_api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not set - Claude processing will not work")
        
        if not self.google_api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set - Gemini processing will not be available")
        
        if not self.huggingface_api_key:
            logger.warning("⚠️ HUGGINGFACE_API_KEY not set - HuggingFace processing will not be available")

    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama and verify model availability."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"🔍 Ollama response: {data}")
                
                # ✅ FIX: Handle different response structures
                if isinstance(data, dict):
                    models_list = data.get('models', [])
                elif isinstance(data, list):
                    models_list = data
                else:
                    logger.error(f"❌ Unexpected Ollama response format: {type(data)}")
                    return False
                
                # ✅ FIX: Safely extract model names
                available_models = []
                for model in models_list:
                    if isinstance(model, dict):
                        # Handle both 'name' and 'model' keys
                        model_name = model.get('name') or model.get('model')
                        if model_name:
                            available_models.append(model_name)
                    elif isinstance(model, str):
                        available_models.append(model)
                
                logger.info(f"🔍 Available Ollama models: {available_models}")
                
                # ✅ FIX: Check if model exists (with or without tag)
                model_base = self.ollama_model.split(':')[0]  # Extract base name
                model_exists = any(
                    self.ollama_model in m or model_base in m 
                    for m in available_models
                )
                
                if model_exists:
                    logger.info(f"✅ Ollama model '{self.ollama_model}' is available")
                    return True
                else:
                    logger.warning(f"⚠️ Model '{self.ollama_model}' not found in Ollama")
                    logger.info(f"💡 Available models: {available_models}")
                    logger.info(f"💡 Install model: ollama pull {self.ollama_model}")
                    return False
            else:
                logger.error(f"❌ Ollama returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Cannot connect to Ollama at {self.ollama_host}")
            logger.info("💡 Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            logger.error(f"❌ Ollama connection test failed: {e}", exc_info=True)
            logger.info("💡 Make sure Ollama is running: ollama serve")
            logger.info(f"💡 Install model: ollama pull {self.ollama_model}")
            return False

    def _log_extraction_warning(self, field: str, url: str, error: str):
        """Log extraction warning and track for summary reporting"""
        warning_msg = f"⚠️ {field} extraction failed for {url}: {error}"
        logger.warning(warning_msg)
        self.extraction_warnings.append({
            'field': field,
            'url': url,
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def get_extraction_summary(self) -> dict:
        """Get summary of extraction warnings"""
        if not self.extraction_warnings:
            return {'total_warnings': 0, 'message': 'All field extractions successful'}
        
        field_counts = {}
        for warning in self.extraction_warnings:
            field = warning['field']
            field_counts[field] = field_counts.get(field, 0) + 1
        
        return {
            'total_warnings': len(self.extraction_warnings),
            'field_breakdown': field_counts,
            'message': f"Field extraction completed with {len(self.extraction_warnings)} warnings across {len(field_counts)} field types"
        }
    
    # User agents for rotation to avoid 403 blocks
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]
    
    async def scrape_with_crawl4ai(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Use Crawl4AI to scrape article URLs and extract core content,
        automatically converting it into clean Markdown or JSON.
        """
        try:
            # logger.info(f"🕷️ Scraping URL with Crawl4AI: {url}")
            
            if CRAWL4AI_AVAILABLE:
                # Use full Crawl4AI implementation with browser automation
                return await self._crawl4ai_full_scrape(url)
            else:
                # Fallback to basic scraping
                return await self._fallback_scrape(url)
                
        except Exception as e:
            logger.error(f"❌ Crawl4AI scraping failed for {url}: {str(e)}")
            return None
    
    async def _crawl4ai_full_scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Full Crawl4AI implementation with browser automation and advanced extraction"""
        try:
            # Use the module-level temp directory that was set before imports
            global temp_base_dir
            
            # Create session-specific subdirectories
            session_dir = os.path.join(temp_base_dir, f'session_{os.getpid()}_{asyncio.current_task().get_name() if asyncio.current_task() else "main"}')
            os.makedirs(session_dir, exist_ok=True)
            
            # Configure Crawl4AI with advanced options and proper permissions
            crawler_strategy = AsyncPlaywrightCrawlerStrategy(
                headless=True,
                browser_type="chromium",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                },
                user_data_dir=os.path.join(session_dir, 'user_data'),
                downloads_path=os.path.join(session_dir, 'downloads')
            )
            
            # Define extraction strategy for structured content
            extraction_strategy = JsonCssExtractionStrategy({
                "article": {
                    "title": "h1, title, .title, [data-title], .headline, .entry-title, .post-title",
                    "author": ".author, .byline, [data-author], .writer, .post-author",
                    "date": "time, .date, .published, [datetime], .post-date, .article-date",
                    "content": "article, .content, .post-content, .entry-content, .article-body, main, .main-content",
                    "description": "meta[name='description'], .description, .excerpt, .summary",
                    "tags": ".tags, .categories, .keywords, .topics, .tag",
                    "image": "img[src], .featured-image img, .article-image img"
                }
            })
            
            # Initialize the async crawler with Railway-compatible settings
            crawler = None
            try:
                logger.info(f"🔧 Initializing AsyncWebCrawler for {url}")
                crawler = AsyncWebCrawler(
                    crawler_strategy=crawler_strategy,
                    always_by_pass_cache=True,
                    verbose=False,
                    cache_mode="disabled",
                    timeout=30
                )
                
                # ✅ ADD NULL CHECK FOR CRAWLER
                if crawler is None:
                    logger.error(f"❌ AsyncWebCrawler initialization returned None for {url}")
                    return await self._fallback_scrape(url)
                
                logger.info(f"✅ AsyncWebCrawler initialized successfully for {url}")
                
                # Perform the crawl with advanced options
                logger.info(f"🌐 Starting crawl for {url}")
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy,
                    bypass_cache=True,
                    process_iframes=True,
                    remove_overlay_elements=True,
                    simulate_user=True,
                    override_navigator=True,
                    wait_for="body",
                    delay_before_return_html=2.0,
                    css_selector="body",
                    screenshot=False,
                    magic=True
                )
                
                # ✅ ADD NULL CHECK FOR RESULT
                if result is None:
                    logger.error(f"❌ Crawl result is None for {url} - crawler.arun() returned None")
                    return await self._fallback_scrape(url)
                
                # ✅ NOW SAFE TO ACCESS result.success
                if not result.success:
                    logger.error(f"❌ Crawl4AI failed for {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                    return await self._fallback_scrape(url)
                
                logger.info(f"✅ Crawl completed successfully for {url}")
                
                # Extract structured data
                extracted_data = {}
                
                # Parse extracted JSON if available
                if result.extracted_content:
                    try:
                        json_data = json.loads(result.extracted_content)
                        if isinstance(json_data, list) and json_data:
                            extracted_data = json_data[0].get("article", {})
                        elif isinstance(json_data, dict):
                            extracted_data = json_data.get("article", {})
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Could not parse extracted JSON for {url}")
                
                # Get clean markdown content
                markdown_content = result.markdown or ""
                
                # Extract title with exception handling
                title = "No title found"
                try:
                    title = (
                        extracted_data.get("title") or 
                        result.metadata.get("title") or
                        "No title found"
                    )
                    if isinstance(title, list):
                        title = title[0] if title else "No title found"
                    title = str(title).strip()
                except Exception as e:
                    self._log_extraction_warning("Title", url, str(e))
                    title = "No title found"
                
                # Extract description with exception handling
                description = ""
                try:
                    description = (
                        extracted_data.get("description") or
                        result.metadata.get("description") or
                        ""
                    )
                    if isinstance(description, list):
                        description = description[0] if description else ""
                    description = str(description).strip()
                except Exception as e:
                    self._log_extraction_warning("Description", url, str(e))
                    description = ""
                
                # Extract author with exception handling
                author = None
                try:
                    author = extracted_data.get("author")
                    if isinstance(author, list):
                        author = author[0] if author else None
                    if author:
                        author = str(author).strip()
                except Exception as e:
                    self._log_extraction_warning("Author", url, str(e))
                    author = None
                
                # Extract publication date with exception handling
                pub_date = None
                try:
                    pub_date = extracted_data.get("date")
                    if isinstance(pub_date, list):
                        pub_date = pub_date[0] if pub_date else None
                    if pub_date:
                        pub_date = str(pub_date).strip()
                    
                    # If no date found, use current date as fallback
                    if not pub_date:
                        pub_date = datetime.now(timezone.utc).isoformat()
                        logger.info(f"📅 Using current date as fallback for {url}")
                        
                except Exception as e:
                    self._log_extraction_warning("Date", url, str(e))
                    # Use current date as fallback
                    pub_date = datetime.now(timezone.utc).isoformat()
                    logger.info(f"📅 Using current date as fallback due to extraction error for {url}")
                
                # Extract main content with exception handling
                content = ""
                full_clean_content = ""
                try:
                    if extracted_data.get("content"):
                        content = extracted_data["content"]
                        if isinstance(content, list):
                            content = " ".join(content)
                    else:
                        content = markdown_content
                    
                    content = str(content).strip()
                    
                    # --- STEP 1: Full Cleaning ---
                    full_clean_content = self._clean_content(content)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Content extraction failed for {url}: {str(e)}")
                    content = ""
                    full_clean_content = ""

                # Calculate reading time with exception handling
                reading_time_minutes = 1
                try:
                    # --- STEP 2: Calculate Reading Time (on FULL content) ---
                    word_count = len(re.findall(r'\b\w+\b', full_clean_content))
                    reading_time_minutes = int((word_count / 225) + 0.5) if word_count > 0 else 1
                    
                    # Ensure a minimum reading time of 1 minute
                    if reading_time_minutes == 0:
                        reading_time_minutes = 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Reading time calculation failed for {url}: {str(e)}")
                    reading_time_minutes = 1
                
                # Limit content for LLM processing with exception handling
                try:
                    # --- STEP 3: Limit Content (for LLM prompt) ---
                    content = full_clean_content[:4000]
                except Exception as e:
                    logger.warning(f"⚠️ Content truncation failed for {url}: {str(e)}")
                    content = content[:4000] if content else "" 
 
                # Extract tags/topics with exception handling
                tags = []
                try:
                    tags = extracted_data.get("tags", [])
                    if isinstance(tags, str):
                        tags = [tags]
                    
                    # Add fallback keywords if no tags found
                    if not tags:
                        tags = self._generate_fallback_keywords(url, title, content)
                        
                    # Ensure tags is a list
                    if not isinstance(tags, list):
                        tags = []
                        
                except Exception as e:
                    logger.warning(f"⚠️ Tags extraction failed for {url}: {str(e)}")
                    try:
                        tags = self._generate_fallback_keywords(url, title, content)
                    except Exception as fallback_e:
                        logger.warning(f"⚠️ Fallback keywords generation failed for {url}: {str(fallback_e)}")
                        tags = ["Technology", "AI News"]
                
                logger.info(f"🔄 Crawl4  full scrapre result for URL : {url}")

                # Detect content type with exception handling
                content_type = "Blogs"
                try:
                    content_type = self._detect_content_type(url, result.html, title, content, result.media)
                except Exception as e:
                    logger.warning(f"⚠️ Content type detection failed for {url}: {str(e)}")
                    content_type = "Blogs"

                # ✅ Extract images from HTML
                try:
                    soup_for_images = BeautifulSoup(result.html, 'html.parser')
                    image_data = self._extract_images(soup_for_images, url)
                except Exception as img_error:
                    logger.warning(f"⚠️ Image extraction failed in Crawl4AI: {img_error}")
                    image_data = {'image_url': None, 'image_source': None}

            except Exception as crawler_e:
                logger.error(f"❌ Crawler operations failed for {url}: {str(crawler_e)}")
                return await self._fallback_scrape(url)

            extracted_result = {
                "title": title.strip(),
                "description": description.strip(),
                "content": content.strip(),
                "author": author,
                "date": pub_date,
                "tags": tags,
                "url": url,
                "reading_time": reading_time_minutes,
                "markdown": markdown_content[:3000],  # Keep some markdown for reference
                "extracted_time": datetime.now(timezone.utc).isoformat(),
                "extraction_method": "crawl4ai_full",
                "links": result.links[:10] if result.links else [],  # Extract some links
                "media": result.media[:5] if result.media else [],  # Extract some media
                "content_type": content_type,  # Add detected content type
                "image_url": image_data.get('image_url'),  # ✅ ADD THIS
                "image_source": image_data.get('image_source')  # ✅ ADD THIS
            }

            logger.info(f"🔄 Crawl4  full scrapre result for title : {extracted_result.get('title', 'Unknown')[:50]}... description: {extracted_result.get('description', 'Unknown')[:50]}...")
            
            
            # Cleanup session directory for Railway deployment
            try:
                import shutil
                shutil.rmtree(session_dir, ignore_errors=True)
            except:
                pass  # Ignore cleanup errors
                
            return extracted_result
                
        except Exception as e:
            logger.error(f"❌ Full Crawl4AI scraping failed for {url}: {str(e)}")
            # Cleanup session directory on error
            try:
                import shutil
                shutil.rmtree(session_dir, ignore_errors=True)
            except:
                pass
            return await self._fallback_scrape(url)
            
        finally:
            # Modern Crawl4AI versions handle cleanup automatically
            # if crawler:
            #     try:
            #         await crawler.aclose()
            #         logger.debug(f"✅ Crawler properly closed for {url}")
            #     except Exception as cleanup_error:
            #         logger.warning(f"⚠️ Error closing crawler: {cleanup_error}")
            pass
    
    def _extract_youtube_metadata(self, url: str) -> dict:
        """Extract YouTube video metadata using YouTube API or webpage scraping"""
        try:
            import re
            import requests
            from urllib.parse import urlparse, parse_qs
            
            # Extract video ID from URL
            video_id = None
            if 'youtube.com/watch' in url:
                parsed_url = urlparse(url)
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
            elif 'youtu.be/' in url:
                video_id = url.split('youtu.be/')[-1].split('?')[0]
            
            if not video_id:
                logger.warning(f"⚠️ Could not extract video ID from {url}")
                return {}
            
            logger.info(f"🎥 Extracting metadata for YouTube video ID: {video_id}")
            
            # Try to extract metadata from the YouTube page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            html = response.text
            
            metadata = {}
            
            # Extract title
            title_match = re.search(r'"title":"([^"]+)"', html)
            if title_match:
                metadata['title'] = title_match.group(1).replace('\\u0026', '&')
            
            # Extract description
            desc_match = re.search(r'"shortDescription":"([^"]+)"', html)
            if desc_match:
                metadata['description'] = desc_match.group(1)[:500]  # Limit description length
            
            # Extract duration (if available) and convert to minutes for database
            duration_match = re.search(r'"lengthSeconds":"([^"]+)"', html)
            if duration_match:
                duration_seconds = int(duration_match.group(1))
                duration_minutes = max(1, duration_seconds // 60)  # At least 1 minute
                metadata['duration'] = duration_minutes  # Store as integer minutes for database
                metadata['duration_display'] = f"{duration_seconds // 60}:{duration_seconds % 60:02d}"  # Keep formatted version for display
            
            # Extract channel name
            channel_match = re.search(r'"ownerChannelName":"([^"]+)"', html)
            if channel_match:
                metadata['channel'] = channel_match.group(1)
            
            # Extract view count (if available)
            views_match = re.search(r'"viewCount":"([^"]+)"', html)
            if views_match:
                metadata['views'] = views_match.group(1)
            
            logger.info(f"🎥 YouTube metadata extracted: {len(metadata)} fields")
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract YouTube metadata for {url}: {e}")
            return {}

    def _extract_podcast_metadata(self, url: str) -> dict:
        """Extract podcast episode metadata from various podcast platforms"""
        try:
            import re
            import requests
            from urllib.parse import urlparse
            
            logger.info(f"🎧 Extracting metadata for podcast URL: {url}")
            
            # Try to extract metadata from the podcast page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            html = response.text
            
            metadata = {}
            
            # Extract title - try multiple patterns
            title_patterns = [
                r'<title[^>]*>([^<]+)</title>',
                r'"name":"([^"]+)"',
                r'"title":"([^"]+)"',
                r'<h1[^>]*>([^<]+)</h1>',
                r'<meta property="og:title" content="([^"]*)"',
            ]
            
            for pattern in title_patterns:
                title_match = re.search(pattern, html, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    # Clean up common title suffixes
                    title = re.sub(r'\s*\|\s*Spotify.*$', '', title)
                    title = re.sub(r'\s*\|\s*Apple Podcasts.*$', '', title)
                    title = re.sub(r'\s*\|\s*SoundCloud.*$', '', title)
                    metadata['title'] = title[:200]  # Limit title length
                    break
            
            # Extract description
            desc_patterns = [
                r'<meta name="description" content="([^"]*)"',
                r'<meta property="og:description" content="([^"]*)"',
                r'"description":"([^"]+)"',
                r'<p[^>]*class="[^"]*description[^"]*"[^>]*>([^<]+)</p>',
            ]
            
            for pattern in desc_patterns:
                desc_match = re.search(pattern, html, re.IGNORECASE)
                if desc_match:
                    metadata['description'] = desc_match.group(1)[:500]  # Limit description length
                    break
            
            # Extract duration - try to find audio duration
            duration_patterns = [
                r'"duration[^"]*":"?([0-9:]+)"?',
                r'"durationMs":"?([0-9]+)"?',
                r'"lengthSeconds":"?([0-9]+)"?',
                r'duration[^>]*>([0-9:]+)<',
                r'([0-9]+:[0-9]+:[0-9]+)',  # HH:MM:SS format
                r'([0-9]+:[0-9]+)',         # MM:SS format
            ]
            
            for pattern in duration_patterns:
                duration_match = re.search(pattern, html)
                if duration_match:
                    duration_str = duration_match.group(1)
                    try:
                        # Convert duration to minutes (integer) for database
                        if 'Ms' in pattern and duration_str.isdigit():
                            # Milliseconds to minutes
                            duration_minutes = max(1, int(duration_str) // 60000)
                        elif duration_str.isdigit():
                            # Seconds to minutes
                            duration_minutes = max(1, int(duration_str) // 60)
                        elif ':' in duration_str:
                            # Time format (HH:MM:SS or MM:SS)
                            time_parts = duration_str.split(':')
                            if len(time_parts) == 3:  # HH:MM:SS
                                hours, minutes, seconds = map(int, time_parts)
                                duration_minutes = hours * 60 + minutes + (1 if seconds > 0 else 0)
                            elif len(time_parts) == 2:  # MM:SS
                                minutes, seconds = map(int, time_parts)
                                duration_minutes = minutes + (1 if seconds > 0 else 0)
                            else:
                                duration_minutes = 30  # Default fallback
                        else:
                            duration_minutes = 30  # Default fallback
                        
                        metadata['duration'] = max(1, duration_minutes)  # At least 1 minute
                        metadata['duration_display'] = duration_str  # Keep original for display
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Extract channel name
            channel_patterns = [
                r'"creator[^"]*":"([^"]+)"',
                r'"author[^"]*":"([^"]+)"',
                r'"artist[^"]*":"([^"]+)"',
                r'<meta name="author" content="([^"]*)"',
                r'"podcast[^"]*name":"([^"]+)"',
                r'"show[^"]*name":"([^"]+)"',
            ]
            
            for pattern in channel_patterns:
                channel_match = re.search(pattern, html, re.IGNORECASE)
                if channel_match:
                    metadata['host'] = channel_match.group(1)[:100]  # Limit host name length
                    break
            
            # Extract published date
            date_patterns = [
                r'"publishedAt":"([^"]+)"',
                r'"releaseDate":"([^"]+)"',
                r'"datePublished":"([^"]+)"',
                r'<meta property="article:published_time" content="([^"]*)"',
                r'"published":"([^"]+)"',
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, html)
                if date_match:
                    try:
                        from datetime import datetime
                        date_str = date_match.group(1)
                        # Try to parse the date
                        if 'T' in date_str:
                            # ISO format
                            parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            metadata['published_date'] = parsed_date.isoformat()
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Set default values if not found
            if 'duration' not in metadata:
                metadata['duration'] = 30  # Default 30 minutes for podcasts
            if 'host' not in metadata:
                metadata['host'] = 'Unknown'
            
            logger.info(f"🎧 Extracted podcast metadata: title={metadata.get('title', 'N/A')[:50]}...")
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract podcast metadata for {url}: {e}")
            return {}

    def _get_random_user_agent(self) -> str:
        """Get a random user agent for HTTP requests"""
        import random
        return random.choice(self.USER_AGENTS)
    
    async def _retry_request_with_different_agent(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Retry HTTP request with different user agents"""
        import random
        import time
        
        for attempt in range(max_retries):
            try:
                user_agent = self._get_random_user_agent()
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0'
                }
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            logger.debug(f"✅ Success with user agent attempt {attempt + 1} for {url}")
                            return await response.text()
                        elif response.status == 403:
                            logger.warning(f"⚠️ HTTP 403 attempt {attempt + 1}/{max_retries} for {url}")
                            if attempt < max_retries - 1:
                                # Wait with exponential backoff
                                wait_time = random.uniform(1, 3) * (2 ** attempt)
                                await asyncio.sleep(wait_time)
                                continue
                        else:
                            logger.warning(f"⚠️ HTTP {response.status} for {url}")
                            return None
                            
            except Exception as e:
                logger.warning(f"⚠️ Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(1, 2))
                    
        logger.error(f"❌ All retry attempts failed for {url}")
        return None
    
    async def _fallback_scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Enhanced fallback scraping with user agent rotation"""
        import random
        
        # Special handling for YouTube videos
        if 'youtube.com/watch' in url or 'youtu.be/' in url:
            logger.info(f"🎥 Using YouTube-specific metadata extraction for: {url}")
            youtube_metadata = self._extract_youtube_metadata(url)
            if youtube_metadata:
                return {
                    'title': youtube_metadata.get('title', 'YouTube Video'),
                    'content': youtube_metadata.get('description', 'YouTube video content'),
                    'summary': youtube_metadata.get('description', '')[:200] + '...' if youtube_metadata.get('description') else '',
                    'author': youtube_metadata.get('channel', 'Unknown'),
                    'url': url,
                    'published_date': datetime.now(timezone.utc).isoformat(),
                    'reading_time': youtube_metadata.get('duration', 5),  # Duration in minutes (integer) for database
                    'extraction_method': 'youtube_metadata_extraction',
                    'content_type_label': 'videos',  # Explicitly set as video
                    'keywords': ['youtube', 'video', 'ai', 'technology']
                }
        
        # Special handling for podcast platforms
        podcast_platforms = ['spotify.com/episode', 'soundcloud.com', 'anchor.fm', 'podcasts.apple.com', 
                           'podcasts.google.com', 'stitcher.com', 'overcast.fm', 'pocketcasts.com']
        if any(platform in url.lower() for platform in podcast_platforms):
            logger.info(f"🎧 Using podcast-specific metadata extraction for: {url}")
            podcast_metadata = self._extract_podcast_metadata(url)
            if podcast_metadata:
                return {
                    'title': podcast_metadata.get('title', 'Podcast Episode'),
                    'content': podcast_metadata.get('description', 'Podcast episode content'),
                    'summary': podcast_metadata.get('description', '')[:200] + '...' if podcast_metadata.get('description') else '',
                    'author': podcast_metadata.get('host', 'Unknown'),
                    'url': url,
                    'published_date': podcast_metadata.get('published_date', datetime.now(timezone.utc).isoformat()),
                    'reading_time': podcast_metadata.get('duration', 30),  # Duration in minutes (integer) for database
                    'extraction_method': 'podcast_metadata_extraction',
                    'content_type_label': 'podcasts',  # Explicitly set as podcast
                    'keywords': ['podcast', 'audio', 'ai', 'technology']
                }
        
        for attempt in range(3):  # Try up to 3 times with different user agents
            try:
                # logger.info(f"🔄 Using enhanced fallback scraping for: {url} (attempt {attempt + 1})")
                
                # Use random user agent for each attempt
                user_agent = self._get_random_user_agent()
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Referer': 'https://google.com/',
                }
                
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, timeout=30, allow_redirects=True) as response:
                        if response.status == 403 and attempt < 2:
                            # HTTP 403 - try again with different user agent
                            logger.warning(f"⚠️ HTTP 403 for {url}, retrying with different user agent (attempt {attempt + 1})")
                            await asyncio.sleep(random.uniform(1, 3))  # Random delay
                            continue
                        elif response.status != 200:
                            logger.error(f"❌ Failed to fetch {url}: HTTP {response.status}")
                            if attempt < 2:
                                await asyncio.sleep(random.uniform(1, 2))
                                continue
                            return None
                        
                        html_content = await response.text()
                        
                        # Extract basic content using BeautifulSoup
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript", "form"]):
                            element.decompose()
                        
                        # Extract title with exception handling
                        title = "No title found"
                        try:
                            title = self._extract_title(soup)
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback title extraction failed for {url}: {str(e)}")
                            title = "No title found"
                        
                        # Extract meta description with exception handling
                        description = ""
                        try:
                            description = self._extract_description(soup)
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback description extraction failed for {url}: {str(e)}")
                            description = ""
                        
                        # Extract author with exception handling
                        author = None
                        try:
                            author = self._extract_author(soup)
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback author extraction failed for {url}: {str(e)}")
                            author = None
                        
                        # Extract publication date with exception handling
                        pub_date = None
                        try:
                            pub_date = self._extract_date(soup)
                            
                            # If no date found, use current date as fallback
                            if not pub_date:
                                pub_date = datetime.now(timezone.utc).isoformat()
                                logger.info(f"📅 Fallback scraper using current date for {url}")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback date extraction failed for {url}: {str(e)}")
                            # Use current date as fallback
                            pub_date = datetime.now(timezone.utc).isoformat()
                            logger.info(f"📅 Fallback scraper using current date due to extraction error for {url}")
                        
                        # Extract and process content with exception handling
                        content = ""
                        full_clean_content = ""
                        reading_time_minutes = 1
                        try:
                            # Extract main content with priority selectors
                            content = self._extract_content(soup)
                            # Clean the full content
                            full_clean_content = self._clean_content(content)
                            
                            # Calculate word count and reading time (225 WPM)
                            word_count = len(re.findall(r'\b\w+\b', full_clean_content))
                            reading_time_minutes = int((word_count / 225) + 0.5) if word_count > 0 else 1
                            
                            if reading_time_minutes == 0:
                                reading_time_minutes = 1
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback content extraction failed for {url}: {str(e)}")
                            content = ""
                            full_clean_content = ""
                            reading_time_minutes = 1

                        # Extract tags/keywords with exception handling
                        tags = []
                        try:
                            tags = self._extract_tags(soup)
                            if not tags:
                                tags = self._generate_fallback_keywords(url, title, full_clean_content)
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback tags extraction failed for {url}: {str(e)}")
                            try:
                                tags = self._generate_fallback_keywords(url, title, full_clean_content)
                            except Exception as fallback_e:
                                logger.warning(f"⚠️ Fallback keywords generation failed for {url}: {str(fallback_e)}")
                                tags = ["Technology", "AI News"]
                        
                        # Detect content type with exception handling
                        content_type = "Blogs"
                        try:
                            content_type = self._detect_content_type(url, html_content, title, full_clean_content, [])
                        except Exception as e:
                            logger.warning(f"⚠️ Fallback content type detection failed for {url}: {str(e)}")
                            content_type = "Blogs"
                        
                        image_data = self._extract_images(soup, url)

                        return {
                            "title": title.strip(),
                            "description": description.strip(),
                            "content": full_clean_content.strip()[:4000],
                            "author": author,
                            "date": pub_date,
                            "tags": tags,
                            "url": url,
                            "reading_time": reading_time_minutes,
                            "extracted_date": datetime.now(timezone.utc).isoformat(),
                            "extraction_method": "enhanced_fallback_with_rotation",
                            "content_type": content_type,  # Add detected content type
                            "image_url": image_data.get('image_url'),  # ✅ ADD THIS
                            "image_source": image_data.get('image_source')  # ✅ ADD THIS
                        }
                        
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Fallback scraping attempt {attempt + 1} failed for {url}: {str(e)}, retrying...")
                    await asyncio.sleep(random.uniform(1, 2))
                    continue
                else:
                    logger.error(f"❌ Enhanced fallback scraping failed for {url}: {str(e)}")
                    
        # All attempts failed
        logger.error(f"❌ All fallback scraping attempts failed for {url}")
        return None
    
    def _extract_title(self, soup) -> str:
        """Extract title with multiple fallback methods"""
        # Try og:title first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content']
        
        # Try h1 tags
        h1_tag = soup.find('h1')
        if h1_tag and len(h1_tag.get_text().strip()) > 10:
            return h1_tag.get_text().strip()
        
        # Try title tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            # Remove common site suffixes
            for suffix in [' | ', ' - ', ' :: ']:
                if suffix in title:
                    title = title.split(suffix)[0]
                    break   
            return title
        
        return "No title found"
    
    def _extract_description(self, soup) -> str:
        """Extract description with multiple methods"""
        # Try og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc['content']
        
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        
        # Try excerpt or summary classes
        for selector in ['.excerpt', '.summary', '.description', '.lead']:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()[:200]
        
        return ""
    
    def _extract_author(self, soup) -> Optional[str]:
        """Extract author with multiple methods"""
        # Try meta author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta['content']
        
        # Try JSON-LD structured data
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, dict) and 'author' in data:
                    author = data['author']
                    if isinstance(author, dict) and 'name' in author:
                        return author['name']
                    elif isinstance(author, str):
                        return author
            except:
                pass
        
        # Try common author selectors
        author_selectors = [
            '.author', '.byline', '[data-author]', '.writer', '.post-author',
            '.author-name', '.by-author', '.article-author'
        ]
        
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                author_text = elem.get_text().strip()
                # Clean up common prefixes
                for prefix in ['By ', 'by ', 'Author: ', 'Written by ']:
                    if author_text.startswith(prefix):
                        author_text = author_text[len(prefix):]
                return author_text
        
        return None
    
    def _extract_date(self, soup) -> Optional[str]:
        """Extract publication date"""
        # Try datetime attributes
        time_elem = soup.find('time')
        if time_elem and time_elem.get('datetime'):
            return time_elem['datetime']
        
        # Try meta published time
        pub_time = soup.find('meta', property='article:published_time')
        if pub_time and pub_time.get('content'):
            return pub_time['content']
        
        return None
    
    def _extract_content(self, soup) -> str:
        """Extract main content with priority selectors"""
        # Priority content selectors
        content_selectors = [
            'article', '[role="main"]', 'main', '.content', '.post-content', 
            '.entry-content', '.article-body', '.story-body', '.post-body',
            '.main-content', '.article-content', '.blog-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove nested unwanted elements
                for unwanted in content_elem(['script', 'style', 'nav', 'footer', 'aside', '.ad', '.advertisement']):
                    unwanted.decompose()
                
                content = content_elem.get_text()
                if len(content.strip()) > 100:  # Ensure meaningful content
                    return self._clean_content(content)[:4000]
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            content = body.get_text()
            return self._clean_content(content)[:4000]
        
        return ""

    def _extract_images(self, soup, url: str) -> Dict[str, Optional[str]]:
        """
        Extract article images with multiple fallback methods
        Returns dict with 'image_url' and 'image_source'
        """
        try:
            article_images = []
            
            # Priority 1: Open Graph image (most reliable for social sharing)
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                img_url = og_image['content']
                article_images.append(('og', img_url))
                logger.debug(f"🖼️ Found OG image: {img_url[:80]}...")
            
            # Priority 2: Twitter card image
            if not article_images:
                twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
                if twitter_image and twitter_image.get('content'):
                    img_url = twitter_image['content']
                    article_images.append(('twitter', img_url))
                    logger.debug(f"🖼️ Found Twitter image: {img_url[:80]}...")
            
            # Priority 3: JSON-LD structured data image
            if not article_images:
                try:
                    
                    json_ld = soup.find('script', type='application/ld+json')
                    if json_ld:
                        import json
                        data = json.loads(json_ld.string)
                        if isinstance(data, list):
                            data = data[0] if data else {}
                        
                        if isinstance(data, dict) and 'image' in data:
                            image_data = data['image']
                            if isinstance(image_data, str):
                                article_images.append(('jsonld', image_data))
                            elif isinstance(image_data, dict) and 'url' in image_data:
                                article_images.append(('jsonld', image_data['url']))
                            elif isinstance(image_data, list) and image_data:
                                article_images.append(('jsonld', image_data[0] if isinstance(image_data[0], str) else image_data[0].get('url', '')))
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    logger.debug(f"⚠️ JSON-LD image extraction failed: {e}")
                    pass
            
            # Priority 4: Article featured/hero image
            if not article_images:
                featured_selectors = [
                    'img.featured-image',
                    'img.hero-image',
                    'img.main-image',
                    'img.article-image',
                    '.featured-image img',
                    '.hero-image img',
                    '.post-thumbnail img',
                    '[class*="featured"] img',
                    '[class*="hero"] img'
                ]
                
                for selector in featured_selectors:
                    featured_img = soup.select_one(selector)
                    if featured_img and featured_img.get('src'):
                        img_url = featured_img['src']
                        article_images.append(('featured', img_url))
                        logger.debug(f"🖼️ Found featured image: {img_url[:80]}...")
                        break
            
            # Priority 5: First large image in article content
            if not article_images:
                # Look inside article/main content areas
                content_areas = soup.select('article, main, [role="main"], .content, .post-content')
                for area in content_areas:
                    for img in area.find_all('img'):
                        src = img.get('src', '')
                        
                        # Skip small images, icons, logos, tracking pixels
                        if not src or any(x in src.lower() for x in ['icon', 'logo', 'pixel', 'tracker', '1x1', 'avatar', 'emoji', 'blank.gif']):
                            continue
                        
                        # Check image dimensions (if available)
                        width = img.get('width')
                        height = img.get('height')
                        
                        try:
                            # Only use images >= 400px wide
                            if width and int(width) >= 400:
                                article_images.append(('content', src))
                                logger.debug(f"🖼️ Found large image in content: {src[:80]}...")
                                break
                            elif not width:  # No width specified, assume it's good
                                # Additional check: skip if alt text suggests it's not a main image
                                alt = img.get('alt', '').lower()
                                if not any(skip in alt for skip in ['icon', 'logo', 'avatar', 'button']):
                                    article_images.append(('content', src))
                                    logger.debug(f"🖼️ Found image (no dimensions): {src[:80]}...")
                                    break
                        except (ValueError, TypeError):
                            continue
                    
                    if article_images:
                        break
            
            # Get the best image URL
            if article_images:
                source_type, image_url = article_images[0]
                
                # Make relative URLs absolute
                if image_url and not image_url.startswith(('http://', 'https://', 'data:', '//')):
                    from urllib.parse import urljoin
                    image_url = urljoin(url, image_url)
                    logger.debug(f"🔗 Converted to absolute URL: {image_url[:80]}...")
                
                # Handle protocol-relative URLs (//example.com/image.jpg)
                if image_url.startswith('//'):
                    from urllib.parse import urlparse
                    page_scheme = urlparse(url).scheme or 'https'
                    image_url = f"{page_scheme}:{image_url}"
                
                # Log success
                logger.info(f"✅ Image extracted ({source_type}): {image_url[:100]}...")
                
                return {
                    'image_url': image_url,
                    'image_source': 'scraped'
                }
            else:
                logger.warning(f"⚠️ No image found for URL")
                return {
                    'image_url': None,
                    'image_source': None
                }
                
        except Exception as e:
            logger.error(f"❌ Image extraction failed: {str(e)}")
            return {
                'image_url': None,
                'image_source': None
            }
    
    def _extract_tags(self, soup) -> List[str]:
        """Extract tags and keywords from multiple sources"""
        tags = []
        
        # 1. Try JSON-LD structured data (most reliable)
        try:
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    
                    # Extract keywords from JSON-LD
                    if 'keywords' in data:
                        keywords = data['keywords']
                        if isinstance(keywords, list):
                            tags.extend([k.strip() for k in keywords if isinstance(k, str)])
                        elif isinstance(keywords, str):
                            tags.extend([k.strip() for k in keywords.split(',')])
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # 2. Try Parsely meta tags (common on news sites)
        parsely_tags = soup.find('meta', attrs={'name': 'parsely-tags'})
        if parsely_tags and parsely_tags.get('content'):
            tags.extend([tag.strip() for tag in parsely_tags['content'].split(',') if tag.strip() and not tag.strip().startswith('pagetype:')])
        
        # 3. Try Open Graph article:tag
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('article:tag'))
        for og_tag in og_tags:
            if og_tag.get('content'):
                tags.append(og_tag['content'].strip())
        
        # 4. Try meta keywords (traditional)
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            tags.extend([tag.strip() for tag in keywords_meta['content'].split(',')])
        
        # 5. Try arXiv specific tags (for academic papers)
        if 'arxiv.org' in soup.find('meta', property='og:url')['content'] if soup.find('meta', property='og:url') else '':
            # arXiv subjects are often in the abstract or metadata
            subjects = soup.find_all('span', class_='primary-subject')
            for subj in subjects:
                if subj.get_text():
                    tags.append(subj.get_text().strip())
            
            # arXiv MSC classes
            msc_elements = soup.find_all('td', class_='msc-classes')
            for msc in msc_elements:
                if msc.get_text():
                    for msc_item in msc.get_text().split(','):
                        if msc_item.strip():
                            tags.append(msc_item.strip())
        
        # 6. Try WordPress/blog specific tags
        wp_tags = soup.find_all('a', rel='tag')
        for wp_tag in wp_tags:
            if wp_tag.get_text():
                tags.append(wp_tag.get_text().strip())
        
        # 7. Try category/taxonomy links
        for selector in ['.post-categories a', '.category a', '.taxonomy a', '.post-tags a']:
            elements = soup.select(selector)
            for elem in elements:
                tag_text = elem.get_text().strip()
                if tag_text and len(tag_text) < 50:
                    tags.append(tag_text)
        
        # 8. Try tag/category elements (general fallback)
        for selector in ['.tags', '.categories', '.keywords', '.topics', '.tag', '.entry-tags', '.post-meta-tags']:
            elements = soup.select(f'{selector} a, {selector} span, {selector} li')
            for elem in elements:
                tag_text = elem.get_text().strip()
                if tag_text and len(tag_text) < 50:
                    tags.append(tag_text)
        
        # 9. Extract from title/heading for technical content
        url = soup.find('meta', property='og:url')['content'] if soup.find('meta', property='og:url') else ''
        if any(domain in url for domain in ['quantumcomputingreport.com', 'nvidia.com', 'robohub.org']):
            title = soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else ''
            if title:
                # Extract technical terms from title
                technical_words = []
                words = title.split()
                for word in words:
                    # Keep technical terms (AI, ML, quantum, etc.)
                    if (len(word) > 2 and 
                        (word.lower() in ['ai', 'ml', 'quantum', 'robot', 'gpu', 'cpu', 'nvidia', 'openai'] or
                         any(char.isupper() for char in word) or
                         word.lower().endswith(('tech', 'ing', 'tion', 'ment')))):
                        technical_words.append(word)
                tags.extend(technical_words[:3])  # Limit to avoid noise
        
        # Clean and filter tags
        cleaned_tags = []
        for tag in tags:
            if tag and len(tag.strip()) > 1 and len(tag.strip()) < 50:
                # Filter out common non-keywords
                tag_lower = tag.lower().strip()
                # Expanded filter list for better quality
                excluded_terms = {
                    'theverge', 'the verge', 'pagetype:story', 'story', 'home', 'blog', 
                    'news', 'article', 'post', 'page', 'main', 'content', 'read more',
                    'click here', 'subscribe', 'newsletter', 'email', 'follow', 'share',
                    'comments', 'reply', 'admin', 'author', 'editor', 'published',
                    'updated', 'tags', 'categories', 'topics', 'related', 'more'
                }
                
                if (tag_lower not in excluded_terms and 
                    not tag_lower.startswith(('http', 'www', 'mailto:')) and
                    not tag_lower.isdigit() and
                    len(tag_lower) > 2):
                    cleaned_tags.append(tag.strip())
        
        # Return what we found
        return list(set(cleaned_tags))[:10]  # Limit to 10 unique tags
    
    def _generate_fallback_keywords(self, url: str, title: str, content: str) -> List[str]:
        """Generate fallback keywords when no tags are found"""
        import re
        from urllib.parse import urlparse
        
        fallback_keywords = []
        
        # 1. Extract domain-based keywords
        domain = urlparse(url).netloc.lower()
        if 'openai' in domain:
            fallback_keywords.extend(['OpenAI', 'ChatGPT', 'Artificial Intelligence'])
        elif 'nvidia' in domain:
            fallback_keywords.extend(['NVIDIA', 'GPU', 'AI Hardware'])
        elif 'google' in domain or 'deepmind' in domain:
            fallback_keywords.extend(['Google', 'Machine Learning', 'AI Research'])
        elif 'microsoft' in domain:
            fallback_keywords.extend(['Microsoft', 'Azure', 'AI Platform'])
        elif 'arxiv' in domain:
            fallback_keywords.extend(['Research Paper', 'Academic', 'AI Research'])
        elif 'github' in domain:
            fallback_keywords.extend(['Open Source', 'Code', 'Development'])
        elif 'techcrunch' in domain or 'verge' in domain:
            fallback_keywords.extend(['Tech News', 'Technology'])
            
        # 2. Extract from title and content
        text_to_analyze = f"{title} {content[:500]}"  # First 500 chars of content
        
        # Find common AI/tech terms
        tech_pattern = r'\b(?:AI|ML|GPU|CPU|API|IoT|5G|quantum|neural|machine learning|deep learning|algorithm|blockchain|cryptocurrency|robot|automation|cloud|edge computing|cybersecurity|artificial intelligence|natural language|computer vision|data science|big data)\b'
        tech_terms = re.findall(tech_pattern, text_to_analyze, re.IGNORECASE)
        fallback_keywords.extend([term.title() for term in tech_terms])
        
        # 3. Find capitalized words (potential proper nouns/companies)
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{2,15}\b', text_to_analyze)
        # Filter out common words
        meaningful_caps = [word for word in capitalized_words if word not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How', 'Why', 'And', 'But', 'For', 'With']]
        fallback_keywords.extend(meaningful_caps[:3])
        
        # 4. Add generic AI keywords if nothing found
        if not fallback_keywords:
            fallback_keywords = ['Artificial Intelligence', 'Technology', 'Innovation']
            
        # Remove duplicates and limit
        return list(set(fallback_keywords))[:8]
    
    def _detect_content_type(self, url: str, html: str, title: str, content: str, media: list) -> str:
        """Detect content type: Videos, Podcasts, or Blogs"""
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Video detection
        video_indicators = [
            # URL patterns
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com', 'twitch.tv',
            'video', '/watch', '/videos/', 'livestream', 'webinar',
            # Title/content patterns
            'watch:', 'video:', 'tutorial:', 'demo:', 'webinar:', 'live stream',
            'how to', 'walkthrough', 'demonstration', 'presentation'
        ]
        
        # Podcast detection
        podcast_indicators = [
            # URL patterns
            'podcast', 'spotify.com', 'anchor.fm', 'soundcloud.com', 'apple.com/podcasts',
            'castbox.fm', 'stitcher.com', 'overcast.fm', 'pocketcasts.com',
            '/audio/', '/episodes/', '/shows/',
            # Title/content patterns
            'podcast:', 'episode:', 'ep.', 'interview:', 'listen:', 'audio:',
            'conversation with', 'talks with', 'discussing', 'interview'
        ]
        
        # Check for video indicators
        video_score = 0
        for indicator in video_indicators:
            if indicator in url_lower:
                video_score += 3
            if indicator in title_lower:
                video_score += 2
            if indicator in content_lower[:500]:  # Check first 500 chars
                video_score += 1
        
        # Check media elements (if available)
        if media:
            for media_item in media:
                if isinstance(media_item, dict):
                    media_url = media_item.get('src', '').lower()
                    if any(ext in media_url for ext in ['.mp4', '.webm', '.avi', '.mov']):
                        video_score += 2
        
        # Check HTML for video elements
        if html:
            if '<video' in html or 'youtube.com/embed' in html or 'vimeo.com/video' in html:
                video_score += 3
            if 'iframe' in html and any(platform in html for platform in ['youtube', 'vimeo']):
                video_score += 2
        
        # Check for podcast indicators
        podcast_score = 0
        for indicator in podcast_indicators:
            if indicator in url_lower:
                podcast_score += 3
            if indicator in title_lower:
                podcast_score += 2
            if indicator in content_lower[:500]:
                podcast_score += 1
        
        # Check for audio elements
        if media:
            for media_item in media:
                if isinstance(media_item, dict):
                    media_url = media_item.get('src', '').lower()
                    if any(ext in media_url for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                        podcast_score += 2
        
        if html:
            if '<audio' in html or 'soundcloud.com/player' in html:
                podcast_score += 3
        
        # Determine content type based on scores
        if video_score >= 3:
            return "Videos"
        elif podcast_score >= 3:
            return "Podcasts"
        else:
            return "Blogs"  # Default fallback
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize text content"""
        if not content:
            return ""
        
        # Split into lines and clean
        lines = (line.strip() for line in content.splitlines())
        
        # Remove empty lines and join with spaces
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()

    def _get_standardized_prompt(self, scraped_data: Dict[str, Any]) -> str:
        """
        Standardized prompt for all LLM models to ensure consistent results.
        Used by Claude, Gemini, and HuggingFace.
        """
        extraction_method = scraped_data.get('extraction_method', 'unknown')
        author_info = scraped_data.get('author', 'Not specified')
        tags_info = scraped_data.get('tags', [])
        
        return f"""You are an expert Artificial Intelligence technology expert analyst and content classifier. Analyze the following scraped content and extract key information in JSON format.

Content Title: {scraped_data.get('title', '')}
Content Description: {scraped_data.get('description', '')}
Author: {author_info}
Existing Tags: {tags_info}
Content Text: {scraped_data.get('content', '')[:4000]}
Source URL: {scraped_data.get('url', '')}
Extraction Method: {extraction_method}

IMPORTANT: 
1. Look carefully for publication dates in the content text, meta information, or URL patterns. Common date formats include "Published October 10, 2025", "Oct 10, 2025", "2025-10-10", etc.
2. Analyze the URL for content type clues: YouTube/Vimeo/video platforms = Videos, Spotify/podcast platforms = Podcasts, news sites = Blogs.

Please analyze this content and return ONLY a valid JSON object with the following structure:
{{
    "title": "Clear, concise headline for the article (use original title if good)",
    "author": "Author name if found, or null",
    "summary": "4-5 sentence summary of the key points and significance",
    "date": "Publication date in ISO format (YYYY-MM-DD) if found, or null. Look for dates in content text like 'Published October 10, 2025' or 'Oct 10, 2025'",
    "content_type_label": "Classify the content type: 'Videos' if from YouTube/video platforms or contains video content, 'Podcasts' if audio content, otherwise 'Blogs'",
    "significance_score": "Number from 1-10 indicating importance of this AI/tech news",
    "complexity_level": "Classify the complexity into one of the following levels: Low, Medium, High depending on technical depth of content",
    "key_topics": ["list of 3-5 key AI tech topics covered in this content"],
    "topic_category_label": "Classify the core subject matter into ONLY ONE of the following 10 categories: Generative AI,AI Applications,AI Start Ups,AI Infrastructure,Cloud Computing,Machine Learning,AI Safety and Governance,Robotics,Internet Of Things (IoT),Quantum AI,Future Technology"
}}
Focus on Artificial Intelligence, machine learning, technology, and innovation content. Be accurate and provide meaningful analysis.
If this is not AI/tech related content, set significance_score to 1-3.

Return ONLY the JSON object, nothing else."""

    async def process_with_claude(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data from Crawl4AI to Claude for processing.
        Use standardized prompt to get structured output with key details.
        """
        try:
            logger.info(f"🤖 CLAUDE PROCESSING: {scraped_data.get('title', 'Unknown')[:60]}...")
            
            # Use standardized prompt
            prompt = self._get_standardized_prompt(scraped_data)

            # Call Claude API - using Claude 3 Haiku
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1000,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            if not self.anthropic_api_key:
                # Enhanced fallback for demo purposes
                publisher_id_fallback = scraped_data.get('publisher_id')
                logger.warning("⚠️ Using fallback processing (no Anthropic API key)")
                logger.info(f"🔗 Fallback: Preserving publisher_id {publisher_id_fallback} for {scraped_data.get('url', 'Unknown URL')}")
                return ScrapedArticle(
                    title=scraped_data.get('title', 'AI News Article'),
                    author=scraped_data.get('author'),
                    summary=scraped_data.get('description', 'AI news and developments'),
                    content=scraped_data.get('content', '')[:1000],
                    date=scraped_data.get('date') or scraped_data.get('extracted_date'),
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=6.0,
                    complexity_level="Medium",
                    reading_time=scraped_data.get('reading_time', 1),
                    publisher_id=publisher_id_fallback,  # Ensure publisher_id is preserved from scraped_data
                    published_date=scraped_data.get('date') or scraped_data.get('extracted_date'),
                    scraped_date=scraped_data.get('extracted_date'),
                    content_type_label=scraped_data.get('content_type', 'Blogs'),
                    topic_category_label="AI News & Updates",
                    keywords=scraped_data.get('tags', []),
                    image_url=scraped_data.get('image_url'),
                    image_source=scraped_data.get('image_source')
                )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.claude_api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status != 200:
                        logger.error(f"❌ CLAUDE API ERROR: HTTP {response.status}")
                        response_text = await response.text()
                        logger.error(f"❌ Claude error response: {response_text}")
                        return None
                    
                    result = await response.json()
                    
                    # Extract and parse the LLM response
                    content = result.get('content', [{}])[0].get('text', '{}')
                    
                    try:
                        # Clean up the response to extract JSON
                        content_clean = content.strip()
                        
                        # Remove markdown code blocks if present
                        if content_clean.startswith('```json'):
                            content_clean = content_clean[7:]  # Remove ```json
                        if content_clean.startswith('```'):
                            content_clean = content_clean[3:]   # Remove ```
                        if content_clean.endswith('```'):
                            content_clean = content_clean[:-3]  # Remove trailing ```
                        
                        content_clean = content_clean.strip()
                        
                        # Parse JSON response from LLM
                        parsed_data = json.loads(content_clean)

                        logger.info(f"✅ CLAUDE SUCCESS: Processed URL: {parsed_data.get('url', 'Unknown')[:50]}...")
                        logger.info(f"✅ CLAUDE SUCCESS: Processed Title: {parsed_data.get('title', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score', 'N/A')})")
                        logger.info(f"✅ CLAUDE SUCCESS: Processed Author: {parsed_data.get('author', 'Unknown')[:50]}... (Complexity Level: {parsed_data.get('complexity_level', 'N/A')})")
                        logger.info(f"✅ CLAUDE SUCCESS: Processed Summary: {parsed_data.get('summary', 'Unknown')[:150]}... (Content: {parsed_data.get('content', 'N/A')})")
                        logger.info(f"✅ CLAUDE SUCCESS: Processed Published Date: {parsed_data.get('date')}... (Content Type Label: {parsed_data.get('content_type_label', 'N/A')})")


                        # Ensure publisher_id is preserved from original scraped_data
                        publisher_id_preserved = scraped_data.get('publisher_id')
                        logger.info(f"🔗 Preserving publisher_id {publisher_id_preserved} in Claude response for {scraped_data.get('url', 'Unknown URL')}")
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=parsed_data.get('date') or scraped_data.get('date'),
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=scraped_data.get('tags', []),
                            llm_processed='claude-3-haiku-20240307',  # Add explicit model name
                            image_url=scraped_data.get('image_url'),
                            image_source=scraped_data.get('image_source')
                        )                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ CLAUDE JSON ERROR: Failed to parse response: {e}")
                        logger.error(f"Claude raw response: {content[:200]}...")
                        logger.error(f"Claude cleaned response: {content_clean[:200]}...")
                        # Try to fix common JSON issues and retry
                        fixed_content = self._fix_json_formatting(content_clean)
                        if fixed_content:
                            try:
                                parsed_data = json.loads(fixed_content)
                                logger.info(f"✅ CLAUDE SUCCESS (after JSON fix): Processed {parsed_data.get('title', 'Unknown')[:50]}...")
                            except json.JSONDecodeError:
                                logger.error("❌ JSON fix attempt failed, skipping article")
                                return None
                        else:
                            return None
                        
        except Exception as e:
            logger.error(f"❌ CLAUDE PROCESSING FAILED: {str(e)}")
            return None
    
    async def process_with_gemini(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data to Google Gemini for processing.
        """
        try:
            model_name = "gemini-2.0-flash-exp"
            logger.info(f"🤖 GEMINI PROCESSING ({model_name}): {scraped_data.get('title', 'Unknown')[:60]}...")

            if not self.google_api_key:
                logger.error("❌ GOOGLE_API_KEY is not set. Cannot process with Gemini.")
                return None

            # Use standardized prompt (same as Claude)
            prompt = self._get_standardized_prompt(scraped_data)

            gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.google_api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1000,
                    "responseMimeType": "application/json",
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(gemini_api_url, json=payload, timeout=60) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"❌ GEMINI API ERROR: HTTP {response.status} - {response_text}")
                        return None

                    result = await response.json()
                    
                    if 'candidates' not in result or not result['candidates']:
                        logger.error(f"❌ GEMINI API ERROR: No candidates in response. {result.get('promptFeedback')}")
                        return None

                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    try:
                        # Clean up JSON response
                        content_clean = content.strip()
                        
                        if content_clean.startswith('```json'):
                            content_clean = content_clean[7:]
                        if content_clean.startswith('```'):
                            content_clean = content_clean[3:]
                        if content_clean.endswith('```'):
                            content_clean = content_clean[:-3]
                        
                        content_clean = content_clean.strip()
                        
                        parsed_data = json.loads(content_clean)
                        
                        logger.info(f"✅ GEMINI SUCCESS: Processed Title: {parsed_data.get('title', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score', 'N/A')})")

                        publisher_id_preserved = scraped_data.get('publisher_id')
                        logger.info(f"🔗 Preserving publisher_id {publisher_id_preserved} in Gemini response for {scraped_data.get('url', 'Unknown URL')}")
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=parsed_data.get('date') or scraped_data.get('date'),
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=scraped_data.get('tags', []),
                            llm_processed=model_name,
                            image_url=scraped_data.get('image_url'),
                            image_source=scraped_data.get('image_source')
                        )

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ GEMINI JSON ERROR: Failed to parse response: {e}")
                        logger.error(f"Gemini raw response: {content[:200]}...")
                        logger.error(f"Gemini cleaned response: {content_clean[:200]}...")
                        fixed_content = self._fix_json_formatting(content_clean)    
                        if fixed_content:
                            try:
                                parsed_data = json.loads(fixed_content)
                                logger.info(f"✅ GEMINI SUCCESS (after JSON fix): Processed {parsed_data.get('title', 'Unknown')[:50]}...")
                                
                                publisher_id_preserved = scraped_data.get('publisher_id')
                                
                                return ScrapedArticle(
                                    title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                                    author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                                    summary=parsed_data.get('summary', 'No summary available'),
                                    content=scraped_data.get('content', '')[:1000],
                                    date=parsed_data.get('date') or scraped_data.get('date'),
                                    url=scraped_data.get('url', ''),
                                    source=self._extract_domain(scraped_data.get('url', '')),
                                    significance_score=float(parsed_data.get('significance_score', 5.0)),
                                    complexity_level=parsed_data.get('complexity_level', 'Medium'),
                                    reading_time=scraped_data.get('reading_time', 1),
                                    publisher_id=publisher_id_preserved,
                                    published_date=parsed_data.get('date') or scraped_data.get('date'),
                                    scraped_date=scraped_data.get('extracted_date'),
                                    content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                                    topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                                    keywords=scraped_data.get('tags', []),
                                    llm_processed=model_name,
                                    image_url=scraped_data.get('image_url'),
                                    image_source=scraped_data.get('image_source')
                                )
                            except json.JSONDecodeError:
                                logger.error("❌ JSON fix failed for Gemini")
                                return None
                        return None

        except Exception as e:
            logger.error(f"❌ GEMINI PROCESSING FAILED: {str(e)}")
            logger.error(traceback.format_exc())  # ✅ NOW traceback IS IMPORTED
            return None

    async def process_with_huggingface(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data to HuggingFace for processing.
        NOTE: Free tier HuggingFace Inference API has limited model availability.
        For production use, consider using Claude or Gemini instead.
        """
        logger.warning("⚠️ HuggingFace Inference API has limited free tier model availability")
        logger.info("💡 Recommendation: Use Claude (claude) or Gemini (gemini) for better results")
        
        # Since most HuggingFace models return 404, fallback to Claude or create basic article
        if not self.huggingface_api_key:
            logger.error("❌ HUGGINGFACE_API_KEY is not set. Cannot process with HuggingFace.")
            return None
        
        # Try one simple model that might work on free tier
        simple_model = "google/flan-t5-base"  # Smaller model more likely to be available
        
        try:
            logger.info(f"🤖 HUGGINGFACE PROCESSING ({simple_model}): {scraped_data.get('title', 'Unknown')[:60]}...")
            
            # Simplified prompt for smaller model
            prompt = f"""Analyze this AI/tech article and extract: title, author (or null), 2-3 sentence summary, date (YYYY-MM-DD or null), content type (Videos/Podcasts/Blogs), importance score 1-10, complexity (Low/Medium/High), and main topic category.

Title: {scraped_data.get('title', '')}
Content: {scraped_data.get('content', '')[:2000]}
URL: {scraped_data.get('url', '')}

Respond with JSON only."""

            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }

            model_url = f"https://api-inference.huggingface.co/models/{simple_model}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(model_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status != 200:
                        logger.error(f"❌ HUGGINGFACE API ERROR: HTTP {response.status}")
                        logger.info("💡 Falling back to basic extraction without LLM processing")
                        
                        # Return basic article without LLM processing
                        return self._create_basic_article(scraped_data, "huggingface-fallback")
                    
                    result = await response.json()
                    
                    # Extract generated text
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                    else:
                        generated_text = result.get('generated_text', '')
                    
                    if not generated_text:
                        logger.warning("⚠️ No generated text, using basic extraction")
                        return self._create_basic_article(scraped_data, "huggingface-fallback")
                    
                    # Try to parse JSON from response
                    try:
                        content_clean = generated_text.strip()
                        
                        if '```json' in content_clean:
                            content_clean = content_clean.split('```json')[1].split('```')[0]
                        elif '```' in content_clean:
                            content_clean = content_clean.split('```')[1].split('```')[0]
                        
                        start_idx = content_clean.find('{')
                        end_idx = content_clean.rfind('}')
                        
                        if start_idx != -1 and end_idx != -1:
                            content_clean = content_clean[start_idx:end_idx + 1]
                        
                        parsed_data = json.loads(content_clean)
                        
                        logger.info(f"✅ HUGGINGFACE SUCCESS: {parsed_data.get('title', 'Unknown')[:50]}...")
                        
                        publisher_id_preserved = scraped_data.get('publisher_id')
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=parsed_data.get('date') or scraped_data.get('date'),
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=scraped_data.get('tags', []),
                            llm_processed=simple_model,
                            image_url=scraped_data.get('image_url'),
                            image_source=scraped_data.get('image_source')
                        )
                        
                    except json.JSONDecodeError:
                        logger.warning("⚠️ Failed to parse HuggingFace JSON, using basic extraction")
                        return self._create_basic_article(scraped_data, simple_model)
                        
        except Exception as e:
            logger.error(f"❌ HUGGINGFACE PROCESSING FAILED: {str(e)}")
            logger.info("💡 Using basic extraction as fallback")
            return self._create_basic_article(scraped_data, "huggingface-error")
    
    async def process_with_ollama(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data to local Ollama model for processing.
        Works with any Ollama model (Llama 3.2, Mistral, Phi, etc.)
        """
        try:
            logger.info(f"🦙 OLLAMA PROCESSING ({self.ollama_model}): {scraped_data.get('title', 'Unknown')[:60]}...")

            # Use standardized prompt (same as Claude/Gemini)
            prompt = self._get_standardized_prompt(scraped_data)

            # Call Ollama API (synchronous, so we use asyncio.to_thread)
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.ollama_model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.2,
                    'num_predict': 1000,  # Max tokens
                    'top_k': 40,
                    'top_p': 0.9,
                }
            )
            
            if not response or 'message' not in response:
                logger.error("❌ OLLAMA API ERROR: No response or invalid format")
                return None

            content = response['message']['content']
            
            try:
                # Clean up JSON response (same as other LLMs)
                content_clean = content.strip()
                
                # Remove markdown code blocks if present
                if content_clean.startswith('```json'):
                    content_clean = content_clean[7:]
                if content_clean.startswith('```'):
                    content_clean = content_clean[3:]
                if content_clean.endswith('```'):
                    content_clean = content_clean[:-3]
                
                content_clean = content_clean.strip()
                
                # Extract JSON from the response
                start_idx = content_clean.find('{')
                end_idx = content_clean.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    content_clean = content_clean[start_idx:end_idx + 1]
                
                parsed_data = json.loads(content_clean)
                
                logger.info(f"✅ OLLAMA SUCCESS: Processed Title: {parsed_data.get('title', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score', 'N/A')})")
                logger.info(f"✅ OLLAMA Model: {response.get('model', self.ollama_model)}")
                logger.info(f"✅ OLLAMA Tokens: {response.get('eval_count', 'N/A')}")

                publisher_id_preserved = scraped_data.get('publisher_id')
                logger.info(f"🔗 Preserving publisher_id {publisher_id_preserved} in Ollama response for {scraped_data.get('url', 'Unknown URL')}")
                
                return ScrapedArticle(
                    title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                    author=parsed_data.get('author') or scraped_data.get('author', 'Unknown'),
                    summary=parsed_data.get('summary', 'No summary available'),
                    content=scraped_data.get('content', '')[:1000],
                    date=parsed_data.get('date') or scraped_data.get('date'),
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=float(parsed_data.get('significance_score', 5.0)),
                    complexity_level=parsed_data.get('complexity_level', 'Medium'),
                    reading_time=scraped_data.get('reading_time', 1),
                    publisher_id=publisher_id_preserved,
                    published_date=parsed_data.get('date') or scraped_data.get('date'),
                    scraped_date=scraped_data.get('extracted_date'),
                    content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                    topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                    keywords=scraped_data.get('tags', []),
                    llm_processed=f"ollama:{self.ollama_model}",  # Track which Ollama model was used
                    image_url=scraped_data.get('image_url'),
                    image_source=scraped_data.get('image_source')
                )

            except json.JSONDecodeError as e:
                logger.error(f"❌ OLLAMA JSON ERROR: Failed to parse response: {e}")
                logger.error(f"Ollama raw response: {content[:200]}...")
                logger.error(f"Ollama cleaned response: {content_clean[:200]}...")
                
                # Try to fix common JSON issues
                fixed_content = self._fix_json_formatting(content_clean)
                if fixed_content:
                    try:
                        parsed_data = json.loads(fixed_content)
                        logger.info(f"✅ OLLAMA SUCCESS (after JSON fix): Processed {parsed_data.get('title', 'Unknown')[:50]}...")
                        
                        publisher_id_preserved = scraped_data.get('publisher_id')
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author', 'Unknown'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=parsed_data.get('date') or scraped_data.get('date'),
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=scraped_data.get('tags', []),
                            llm_processed=f"ollama:{self.ollama_model}",
                            image_url=scraped_data.get('image_url'),
                            image_source=scraped_data.get('image_source')
                        )
                    except json.JSONDecodeError:
                        logger.error("❌ JSON fix failed for Ollama")
                        return None
                return None

        except Exception as e:
            logger.error(f"❌ OLLAMA PROCESSING FAILED: {str(e)}")
            logger.error(traceback.format_exc())
            
            # If Ollama fails, suggest checking if service is running
            logger.info("💡 Make sure Ollama is running: ollama serve")
            logger.info(f"💡 Check model is installed: ollama list | grep {self.ollama_model}")
            
            return None

    def _create_basic_article(self, scraped_data: Dict[str, Any], llm_model: str) -> ScrapedArticle:
        """Create a basic article structure when LLM processing fails"""
        publisher_id_preserved = scraped_data.get('publisher_id')
        
        # Create a basic summary from description and content
        description = scraped_data.get('description', '')
        content = scraped_data.get('content', '')
        summary = description if description else (content[:300] + '...' if content else 'AI and technology news article')
        
        return ScrapedArticle(
            title=scraped_data.get('title', 'AI News Article'),
            author=scraped_data.get('author'),
            summary=summary,
            content=content[:1000],
            date=scraped_data.get('date') or scraped_data.get('extracted_date'),
            url=scraped_data.get('url', ''),
            source=self._extract_domain(scraped_data.get('url', '')),
            significance_score=6.0,
            complexity_level="Medium",
            reading_time=scraped_data.get('reading_time', 1),
            publisher_id=publisher_id_preserved,
            published_date=scraped_data.get('date') or scraped_data.get('extracted_date'),
            scraped_date=scraped_data.get('extracted_date'),
            content_type_label=scraped_data.get('content_type', 'Blogs'),
            topic_category_label="AI News & Updates",
            keywords=scraped_data.get('tags', []),
            llm_processed=llm_model,
            image_url=scraped_data.get('image_url'),
            image_source=scraped_data.get('image_source')
        )
    
    async def scrape_article(self, url: str, llm_model: str = 'claude') -> Optional[ScrapedArticle]:
        """Complete scraping process: Scrape URL with Crawl4AI, process with selected LLM, return structured article data"""
        logger.info(f"🚀 Starting complete scraping process for: {url} using LLM: {llm_model}")
        
        scraped_data = await self.scrape_with_crawl4ai(url)
        if not scraped_data:
            return None
        
        article = None
        if llm_model == 'gemini':
            article = await self.process_with_gemini(scraped_data)
        elif llm_model == 'huggingface':
            article = await self.process_with_ollama(scraped_data)
        elif llm_model == 'claude':
            article = await self.process_with_claude(scraped_data)
        elif llm_model == 'ollama':  # ✅ NEW: Add Ollama option
            article = await self.process_with_ollama(scraped_data)
        else:
            logger.warning(f"⚠️ Unknown LLM model '{llm_model}'. Defaulting to Claude.")
            article = await self.process_with_claude(scraped_data)

        if article:
            logger.info(f"✅ ARTICLE COMPLETE: {article.title[:60]}... (Score: {article.significance_score})")
        return article
    
    async def scrape_multiple_sources(self, source_urls: List[str]) -> List[ScrapedArticle]:
        """Scrape multiple sources concurrently"""
        logger.info(f"🔄 Scraping {len(source_urls)} sources with Claude processing...")
        
        tasks = [self.scrape_article(url, llm_model='claude') for url in source_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = [r for r in results if isinstance(r, ScrapedArticle)]
        logger.info(f"🎉 PROCESSING COMPLETE: {len(articles)} articles processed")
        return articles
    
    async def scrape_multiple_sources_with_publisher_id(self, article_data: List[Dict], llm_model: str = 'claude') -> List[ScrapedArticle]:
        """Scrape multiple sources with publisher_id and source_category mapping"""
        logger.info(f"🔄 Scraping {len(article_data)} sources with LLM: {llm_model}...")
        
        tasks = [self.scrape_article_with_publisher_id(data['url'], data['publisher_id'], data.get('source_category', 'general'), llm_model) for data in article_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = [r for r in results if isinstance(r, ScrapedArticle)]
        logger.info(f"🎉 LLM PROCESSING COMPLETE: {len(articles)} articles processed using {llm_model}")
        return articles
    
    async def scrape_article_with_publisher_id(self, url: str, publisher_id: int, source_category: str = 'general', llm_model: str = 'claude') -> Optional[ScrapedArticle]:
        """Scrape individual article, include publisher_id, and process with the selected LLM"""
        try:
            scraped_data = await self.scrape_with_crawl4ai(url)
            if scraped_data:
                scraped_data['publisher_id'] = publisher_id
                scraped_data['source_category'] = source_category
                
                if llm_model == 'gemini':
                    return await self.process_with_gemini(scraped_data) if self.google_api_key else await self.process_with_claude(scraped_data)
                elif llm_model == 'ollama' or llm_model == 'huggingface':
                    return await self.process_with_ollama(scraped_data) 
                else:
                    return await self.process_with_claude(scraped_data) if self.anthropic_api_key else None
            return None
        except Exception as e:
            logger.error(f"❌ Failed to scrape article {url}: {str(e)}")
            return None

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for source attribution"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith('www.') else domain
        except:
            return "Unknown"
    
    async def parse_rss_feed(self, feed_url: str, max_articles: int = 10) -> List[str]:
        """Parse RSS feed and extract article URLs with deduplication"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/rss+xml, application/xml, text/xml, */*'}
            response = await asyncio.to_thread(requests.get, feed_url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            all_urls = [entry.get('link') or entry.get('id') for entry in feed.entries[:max_articles * 2] if (entry.get('link') or entry.get('id', '')).startswith('http')]
            
            if not all_urls:
                return []
            
            from db_service import get_database_service
            url_existence = get_database_service().check_multiple_urls_exist(all_urls)
            new_urls = [url for url in all_urls if not url_existence.get(url, False)]
            
            logger.info(f"✅ URL Deduplication: {len(new_urls)} new, {len(all_urls) - len(new_urls)} existing")
            return new_urls[:max_articles]
        except Exception as e:
            logger.error(f"❌ Error parsing RSS feed {feed_url}: {str(e)}")
            return []
    
    def _fix_json_formatting(self, json_str: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues"""
        try:
            json_str = json_str.strip()
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx + 1]
            json.loads(json_str)
            return json_str
        except:
            return None
    
    async def scrape_pending_podcasts(self, llm_model: str = 'gemini') -> Dict[str, Any]:
        """
        Scrape pending podcasts (scraped_status = 'pending') and insert into articles table
        Updates scraped_status to 'completed' after successful insertion
        """
        try:
            logger.info(f"🎧 Starting pending podcast scraping with LLM: {llm_model}")
            
            # Get database service
            from db_service import get_database_service
            db_service = get_database_service()
            
            # Get pending podcasts
            podcasts = db_service.get_pending_podcasts()
            
            if not podcasts:
                logger.info("ℹ️ No pending podcasts to scrape")
                return {
                    "success": True,
                    "message": "No pending podcasts found",
                    "podcasts_processed": 0
                }
            
            logger.info(f"📋 Found {len(podcasts)} pending podcasts")
            
            articles_inserted = 0
            podcasts_processed = []
            
            for podcast in podcasts:
                try:
                    podcast_url = podcast['url']
                    podcast_id = podcast['id']
                    
                    logger.info(f"🎧 Scraping podcast: {podcast['name'][:50]}... - {podcast_url}")
                    
                    # Scrape the podcast URL
                    article = await self.scrape_article_with_publisher_id(
                        url=podcast_url,
                        publisher_id=podcast['publisher_id'],
                        source_category=podcast.get('category_name', 'Podcasts'),
                        llm_model=llm_model
                    )
                    
                    if article:
                        # Force content type to podcasts
                        # Fix null string values
                        published_date = article.date if article.date and article.date.lower() != 'null' else None
                        author = article.author if article.author and str(article.author).lower() != 'null' else "Unknown"
                        
                        article_data = {
                            'title': str(article.title or "Unknown Podcast"),
                            'author': author,
                            'summary': str(article.summary or ""),
                            'content': str(article.content or ""),
                            'url': str(article.url),
                            'source': str(article.source),
                            'significance_score': float(article.significance_score or 5.0),
                            'complexity_level': str(article.complexity_level or "Medium"),
                            'published_date': published_date,
                            'reading_time': int(article.reading_time) if article.reading_time else 30,
                            'content_type_label': 'podcasts',  # Force podcasts
                            'topic_category_label': str(article.topic_category_label or "AI News & Updates"),
                            'scraped_date': datetime.now(timezone.utc).isoformat(),
                            'created_date': datetime.now(timezone.utc).isoformat(),
                            'updated_date': datetime.now(timezone.utc).isoformat(),
                            'llm_processed': article.llm_processed or llm_model,
                            'publisher_id': article.publisher_id,
                            'content_type_id': 3,  # Force podcast content type ID
                            'keywords': ', '.join(str(k) for k in article.keywords if k) if article.keywords else None,
                            'image_url': article.image_url,
                            'image_source': article.image_source
                        }
                        
                        # Insert into articles table
                        success = db_service.insert_article(article_data)
                        
                        if success:
                            # Update podcast status to completed
                            db_service.update_podcast_scrape_status(podcast_id, 'completed')
                            articles_inserted += 1
                            podcasts_processed.append(podcast['name'])
                            logger.info(f"✅ Podcast inserted: {article.title[:60]}...")
                        else:
                            # Mark as completed even if duplicate (already exists)
                            db_service.update_podcast_scrape_status(podcast_id, 'completed')
                            logger.debug(f"⏭️ Podcast skipped (duplicate): {article.title[:60]}...")
                    else:
                        # Mark as failed if scraping didn't work
                        db_service.update_podcast_scrape_status(podcast_id, 'failed')
                        logger.warning(f"⚠️ Failed to scrape podcast: {podcast['name']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing podcast {podcast.get('name', 'Unknown')}: {str(e)}")
                    # Mark as failed
                    try:
                        db_service.update_podcast_scrape_status(podcast['id'], 'failed')
                    except:
                        pass
            
            logger.info(f"🎉 PODCAST SCRAPING COMPLETE: {articles_inserted}/{len(podcasts)} inserted")
            
            return {
                "success": True,
                "message": f"Podcast scraping completed",
                "podcasts_total": len(podcasts),
                "podcasts_processed": len(podcasts_processed),
                "articles_inserted": articles_inserted,
                "llm_model": llm_model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Podcast scraping failed: {str(e)}")
            return {
                "success": False,
                "message": f"Podcast scraping failed: {str(e)}",
                "podcasts_processed": 0
            }
    
    async def scrape_pending_videos(self, llm_model: str = 'gemini') -> Dict[str, Any]:
        """
        Scrape pending videos (scraped_status = 'pending') and insert into articles table
        Updates scraped_status to 'completed' after successful insertion
        """
        try:
            logger.info(f"🎥 Starting pending video scraping with LLM: {llm_model}")
            
            # Get database service
            from db_service import get_database_service
            db_service = get_database_service()
            
            # Get pending videos
            videos = db_service.get_pending_videos()
            
            if not videos:
                logger.info("ℹ️ No pending videos to scrape")
                return {
                    "success": True,
                    "message": "No pending videos found",
                    "videos_processed": 0
                }
            
            logger.info(f"📋 Found {len(videos)} pending videos")
            
            articles_inserted = 0
            videos_processed = []
            
            for video in videos:
                try:
                    video_url = video['url']
                    video_id = video['id']
                    
                    logger.info(f"🎥 Scraping video: {video['name'][:50]}... - {video_url}")
                    
                    # Scrape the video URL
                    article = await self.scrape_article_with_publisher_id(
                        url=video_url,
                        publisher_id=video['publisher_id'],
                        source_category=video.get('category_name', 'Videos'),
                        llm_model=llm_model
                    )
                    
                    if article:
                        # Force content type to videos
                        # Fix null string values
                        published_date = article.date if article.date and article.date.lower() != 'null' else None
                        author = article.author if article.author and str(article.author).lower() != 'null' else "Unknown"
                        
                        article_data = {
                            'title': str(article.title or "Unknown Video"),
                            'author': author,
                            'summary': str(article.summary or ""),
                            'content': str(article.content or ""),
                            'url': str(article.url),
                            'source': str(article.source),
                            'significance_score': float(article.significance_score or 5.0),
                            'complexity_level': str(article.complexity_level or "Medium"),
                            'published_date': published_date,
                            'reading_time': int(article.reading_time) if article.reading_time else 5,
                            'content_type_label': 'videos',  # Force videos
                            'topic_category_label': str(article.topic_category_label or "AI News & Updates"),
                            'scraped_date': datetime.now(timezone.utc).isoformat(),
                            'created_date': datetime.now(timezone.utc).isoformat(),
                            'updated_date': datetime.now(timezone.utc).isoformat(),
                            'llm_processed': article.llm_processed or llm_model,
                            'publisher_id': article.publisher_id,
                            'content_type_id': 2,  # Force video content type ID
                            'keywords': ', '.join(str(k) for k in article.keywords if k) if article.keywords else None,
                            'image_url': article.image_url,
                            'image_source': article.image_source
                        }
                        
                        # Insert into articles table
                        success = db_service.insert_article(article_data)
                        
                        if success:
                            # Update video status to completed
                            db_service.update_video_scrape_status(video_id, 'completed')
                            articles_inserted += 1
                            videos_processed.append(video['name'])
                            logger.info(f"✅ Video inserted: {article.title[:60]}...")
                        else:
                            # Mark as completed even if duplicate (already exists)
                            db_service.update_video_scrape_status(video_id, 'completed')
                            logger.debug(f"⏭️ Video skipped (duplicate): {article.title[:60]}...")
                    else:
                        # Mark as failed if scraping didn't work
                        db_service.update_video_scrape_status(video_id, 'failed')
                        logger.warning(f"⚠️ Failed to scrape video: {video['name']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing video {video.get('name', 'Unknown')}: {str(e)}")
                    # Mark as failed
                    try:
                        db_service.update_video_scrape_status(video['id'], 'failed')
                    except:
                        pass
            
            logger.info(f"🎉 VIDEO SCRAPING COMPLETE: {articles_inserted}/{len(videos)} inserted")
            
            return {
                "success": True,
                "message": f"Video scraping completed",
                "videos_total": len(videos),
                "videos_processed": len(videos_processed),
                "articles_inserted": articles_inserted,
                "llm_model": llm_model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Video scraping failed: {str(e)}")
            return {
                "success": False,
                "message": f"Video scraping failed: {str(e)}",
                "videos_processed": 0
            }



# Admin interface
class AdminScrapingInterface:
    """Admin interface for initiating scraping process"""
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.scraper = Crawl4AIScraper()
        
    def map_content_type_to_id(self, content_type_str, url=None):
        """Map content type string to database ID (1=blogs, 2=videos, 3=podcasts)"""
        if url:
            url_lower = url.lower()
            if 'youtube.com' in url_lower or 'youtu.be' in url_lower or 'vimeo.com' in url_lower:
                return 2
            if 'spotify.com/episode' in url_lower or 'soundcloud.com' in url_lower or 'podcasts.apple.com' in url_lower:
                return 3
        
        if not content_type_str:
            return 1
        
        content_lower = content_type_str.lower().strip()
        if any(term in content_lower for term in ['video', 'youtube', 'vimeo']):
            return 2
        elif any(term in content_lower for term in ['podcast', 'audio']):
            return 3
        return 1
    
    def _get_publisher_id_for_article(self, article_url, domain_mapping):
        """Get publisher_id for an article URL using domain mapping"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(article_url).netloc.lower()
            
            if domain in domain_mapping:
                return domain_mapping[domain]
            if domain.startswith('www.'):
                if domain[4:] in domain_mapping:
                    return domain_mapping[domain[4:]]
            if f"www.{domain}" in domain_mapping:
                return domain_mapping[f"www.{domain}"]
            
            return None
        except:
            return None
    
    def map_ai_topic_to_id(self, content_type_str, content_text=""):
        """Map content to AI topic ID based on content analysis"""
        content_lower = content_text.lower()
        if any(word in content_lower for word in ['robot', 'automation']):
            return 1
        elif any(word in content_lower for word in ['nlp', 'language', 'gpt']):
            return 2
        return 21  # Default: AI News & Updates
        
    async def initiate_scraping(self, admin_email: str = "admin@vidyagam.com", llm_model: str = 'claude', scrape_frequency: int = 1) -> Dict[str, Any]:
        """Admin-initiated scraping process with frequency filtering and URL deduplication"""
        logger.info(f"🔧 Admin initiated scraping: LLM={llm_model}, Frequency={scrape_frequency} days")
        
        # Validate API keys
        if llm_model == 'gemini' and not self.scraper.google_api_key:
            return {"success": False, "message": "Gemini API key not configured", "articles_processed": 0}
        
        elif llm_model == 'ollama':
            # Test Ollama connection
            if not self.scraper._test_ollama_connection():
                return {
                    "success": False, 
                    "message": f"Ollama not available. Make sure it's running: 'ollama serve' and model is installed: 'ollama pull {self.scraper.ollama_model}'", 
                    "articles_processed": 0
                }
        elif llm_model == 'huggingface' and not self.scraper.huggingface_api_key:
            return {"success": False, "message": "HuggingFace API key not configured", "articles_processed": 0}
        elif llm_model == 'claude' and not self.scraper.anthropic_api_key:
            return {"success": False, "message": "Claude API key not configured", "articles_processed": 0}

        try:
            sources = self.db_service.get_ai_sources_by_frequency(scrape_frequency)
            if not sources:
                return {"success": False, "message": f"No sources for {scrape_frequency}-day frequency", "articles_processed": 0}
            
            all_article_data = []
            domain_mapping = {}
            
            for source in sources:
                rss_url = source.get('rss_url')
                publisher_id = source.get('publisher_id')
                
                if rss_url:
                    article_urls = await self.scraper.parse_rss_feed(rss_url, max_articles=3)
                    for url in article_urls:
                        all_article_data.append({'url': url, 'publisher_id': publisher_id, 'source_category': source.get('category', 'general')})
                        try:
                            from urllib.parse import urlparse
                            domain_mapping[urlparse(url).netloc.lower()] = publisher_id
                        except:
                            pass
            
            articles = await self.scraper.scrape_multiple_sources_with_publisher_id(all_article_data, llm_model)
            
            articles_inserted = 0
            for article in articles:
                try:
                    # Fix null string values
                    published_date = article.date if article.date and str(article.date).lower() != 'null' else None
                    author = article.author if article.author and str(article.author).lower() != 'null' else None
                    
                    article_data = {
                        'content_hash': hashlib.md5(article.url.encode()).hexdigest(),
                        'title': str(article.title or "No title")[:500],
                        'summary': str(article.summary or "")[:1000],
                        'content': str(article.content or "")[:10000],
                        'url': str(article.url or ""),
                        'source': str(article.source or "Unknown"),
                        'author': author,
                        'significance_score': int(article.significance_score) if article.significance_score else 6,
                        'complexity_level': str(article.complexity_level or "Medium"),
                        'published_date': published_date,
                        'reading_time': int(article.reading_time) if article.reading_time else 1,
                        'content_type_label': str(article.content_type_label or "Blogs"),
                        'topic_category_label': str(article.topic_category_label or "AI News & Updates"),
                        'scraped_date': datetime.now(timezone.utc).isoformat(),
                        'created_date': datetime.now(timezone.utc).isoformat(),
                        'updated_date': datetime.now(timezone.utc).isoformat(),
                        'llm_processed': article.llm_processed or llm_model,
                        'publisher_id': self._get_publisher_id_for_article(article.url, domain_mapping),
                        'content_type_id': self.map_content_type_to_id(article.content_type_label, article.url),
                        'ai_topic_id': self.map_ai_topic_to_id(article.content_type_label, article.summary),
                        'keywords': ', '.join(str(k) for k in article.keywords if k) if article.keywords else None,
                        'image_url': article.image_url,
                        'image_source': article.image_source
                    }
                    
                    success = await asyncio.to_thread(self.db_service.insert_article, article_data)
                    
                    if success:
                        articles_inserted += 1
                        logger.info(f"✅ Inserted article: {article.title[:60]}... (LLM: {llm_model})")
                    else:
                        logger.debug(f"⏭️ Skipped (duplicate): {article.title[:60]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to insert article: {e}")
            
            logger.info(f"🎉 SCRAPING COMPLETE: {articles_inserted} articles processed")
            
            return {
                "success": True,
                "message": f"Scraping completed for {scrape_frequency}-day frequency sources",
                "sources_scraped": len(sources),
                "articles_found": len(articles),
                "articles_processed": articles_inserted,
                "initiated_by": admin_email,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Admin scraping failed: {str(e)}")
            return {"success": False, "message": f"Scraping failed: {str(e)}", "articles_processed": 0}
    
