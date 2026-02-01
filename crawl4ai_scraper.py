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

# Import resource monitor for system resource management
try:
    from resource_monitor import ResourceMonitor, ResourceThresholds
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False
    logging.warning("⚠️ Resource monitor not available - no resource throttling")

# ✅ OPTIONAL: Ollama import (only for local development)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("⚠️ Ollama not available - local LLM features disabled")

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
    # Use system Playwright browsers instead of temp directory
    playwright_path = os.path.expanduser('~/Library/Caches/ms-playwright')
    if os.path.exists(playwright_path):
        os.environ['PLAYWRIGHT_BROWSERS_PATH'] = playwright_path
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
    keywords: Optional[str] = None  # Changed from List[str] to str for comma-separated format
    publisher_id: Optional[int] = None
    llm_processed: Optional[str] = None
    image_url: Optional[str] = None          # ✅ ADD THIS
    image_source: Optional[str] = None       # ✅ ADD THIS

class Crawl4AIScraper:
    """AI News Scraper using Crawl4AI and Claude LLM with resource management"""
    
    def __init__(self):
        self.extraction_warnings = []  # Track extraction warnings for reporting
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.claude_api_url = "https://api.anthropic.com/v1/messages"

        # Tavily API configuration
        self.tavily_api_key = os.getenv('TAVILY_API_KEY', '')
        self.tavily_api_url = "https://api.tavily.com/search"

        # Log Tavily API key status
        logger.info(f"   - TAVILY_API_KEY: {'✅ Set' if self.tavily_api_key else '❌ Not set'}")
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not set - Tavily search will not work")

        # ✅ NEW: Resource Monitor for system health
        if RESOURCE_MONITOR_AVAILABLE:
            # Configure conservative thresholds for Mac to prevent crashes
            thresholds = ResourceThresholds(
                max_cpu_percent=70.0,  # Throttle threshold
                max_memory_percent=65.0,  # Throttle threshold
                max_memory_mb=5000,  # Throttle at 5GB process memory
                warning_memory_percent=60.0,  # Less noisy - 60% is normal with apps running
                critical_memory_percent=70.0,
                abort_cpu_percent=85.0,  # STOP scraping if CPU > 85%
                abort_memory_percent=80.0,  # STOP scraping if memory > 80%
                abort_memory_mb=6500  # STOP scraping if process > 6.5GB
            )
            self.resource_monitor = ResourceMonitor(thresholds)
            logger.info("✅ Resource monitoring enabled with ABORT protection")
            logger.info(f"   Throttle: CPU<{thresholds.max_cpu_percent}%, Memory<{thresholds.max_memory_percent}%")
            logger.info(f"   ABORT: CPU>{thresholds.abort_cpu_percent}%, Memory>{thresholds.abort_memory_percent}%")
        else:
            self.resource_monitor = None
            logger.warning("⚠️ Resource monitoring disabled")

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
            
            # Configure Crawl4AI with LIGHTWEIGHT browser options for reduced resource usage
            # NOTE: These settings affect ONLY the browser (Chromium), not Ollama/system GPU
            # Ollama can still use GPU for LLM processing independently
            crawler_strategy = AsyncPlaywrightCrawlerStrategy(
                headless=True,
                browser_type="chromium",  # Chromium is lighter than Chrome
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                # Lightweight browser arguments to reduce memory and CPU usage
                # These only affect browser rendering, not Ollama or system GPU
                browser_args=[
                    "--disable-dev-shm-usage",  # Overcome limited resource problems
                    "--disable-setuid-sandbox",
                    "--no-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",  # Don't load images to save bandwidth and memory
                    "--blink-settings=imagesEnabled=false",
                    "--single-process",  # Use single process mode
                    "--no-zygote",  # Don't use zygote process
                    "--disable-accelerated-2d-canvas",  # Disable canvas acceleration
                    "--disable-accelerated-jpeg-decoding",
                    "--disable-accelerated-mjpeg-decode",
                    "--disable-accelerated-video-decode",
                    "--memory-pressure-off",  # Disable memory pressure signals
                    "--max-old-space-size=512",  # Limit V8 heap size to 512MB
                    # GPU flags - disable for browser rendering only (Ollama GPU unaffected)
                    "--disable-gpu",  # Browser doesn't need GPU for text scraping
                    "--disable-software-rasterizer",
                ],
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Cache-Control": "max-age=0"
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
                result = None
                try:
                    # Use longer delays for sites that require browser mode
                    wait_time = 3.0 if self._requires_browser_mode(url) else 2.0
                    
                    result = await crawler.arun(
                        url=url,
                        extraction_strategy=extraction_strategy,
                        bypass_cache=True,
                        process_iframes=True,
                        remove_overlay_elements=True,
                        simulate_user=True,
                        override_navigator=True,
                        wait_for="body",
                        delay_before_return_html=wait_time,
                        css_selector="body",
                        screenshot=False,
                        magic=True,
                        page_timeout=60000  # 60 second timeout for heavy pages
                    )
                except Exception as crawl_error:
                    logger.error(f"❌ Crawl4AI exception for {url}: {str(crawl_error)}")
                    logger.debug(f"📋 Crawl error traceback: {traceback.format_exc()}")
                    # Don't fallback to HTTP for sites that require browser rendering
                    if self._requires_browser_mode(url):
                        logger.warning(f"⚠️ Skipping HTTP fallback for {url} - browser mode required")
                        return None
                    return await self._fallback_scrape(url)
                
                # ✅ ADD NULL CHECK FOR RESULT
                if result is None:
                    logger.error(f"❌ Crawl result is None for {url} - crawler.arun() returned None")
                    # For browser-required sites, try direct Playwright as fallback
                    if self._requires_browser_mode(url):
                        logger.info(f"🔄 Attempting direct Playwright fallback for {url}")
                        return await self._direct_playwright_fallback(url)
                    return await self._fallback_scrape(url)
                
                # ✅ NOW SAFE TO ACCESS result.success
                if not result.success:
                    logger.error(f"❌ Crawl4AI failed for {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                    # For browser-required sites, try direct Playwright as fallback
                    if self._requires_browser_mode(url):
                        logger.info(f"🔄 Attempting direct Playwright fallback for {url}")
                        return await self._direct_playwright_fallback(url)
                    return await self._fallback_scrape(url)
                    # Don't fallback to HTTP for sites that require browser rendering
                    if self._requires_browser_mode(url):
                        logger.warning(f"⚠️ Skipping HTTP fallback for {url} - browser mode required")
                        return None
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9'
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9'
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
    
    def _normalize_date(self, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize various date formats to valid PostgreSQL timestamps.
        Handles malformed dates from LLMs like:
        - "2024-07-00" -> "2024-07-01"
        - "2024-12" -> "2024-12-01"
        - "July 2024" -> "2024-07-01"
        - Partial dates with missing day -> add day 01
        
        Returns None if date cannot be normalized.
        """
        if not date_str or date_str.lower() in ['null', 'none', 'n/a', 'unknown']:
            return None
        
        try:
            from datetime import datetime
            import re
            
            # Clean the input
            date_str = str(date_str).strip()
            
            # Case 1: "2024-07-00" - invalid day (00)
            if re.match(r'^\d{4}-\d{2}-00$', date_str):
                date_str = date_str[:-2] + '01'  # Replace 00 with 01
                logger.debug(f"📅 Normalized date with day=00: {date_str}")
            
            # Case 2: "2024-12" or "2024-07" - missing day
            if re.match(r'^\d{4}-\d{1,2}$', date_str):
                date_str = date_str + '-01'  # Add day 01
                logger.debug(f"📅 Normalized date missing day: {date_str}")
            
            # Case 3: "2024" - year only
            if re.match(r'^\d{4}$', date_str):
                date_str = date_str + '-01-01'  # Add Jan 1st
                logger.debug(f"📅 Normalized year-only date: {date_str}")
            
            # Case 4: Try to parse various formats
            try:
                # Try ISO format first
                parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return parsed_date.isoformat()
            except ValueError:
                # Try other common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%B %Y', '%b %Y', '%Y-%m', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
                
                # If nothing works, return None
                logger.warning(f"⚠️ Could not normalize date: {date_str}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Date normalization error for '{date_str}': {e}")
            return None
    
    def _is_category_or_root_url(self, url: str) -> bool:
        """
        Detect if a URL is a category/root page rather than an individual article.
        Category pages typically end with just a category name (e.g., /news/product-releases/)
        while article pages have additional path segments (e.g., /index/article-title/).
        Also filters out RSS/XML feeds.
        
        Returns True if URL is a category/root page that should be excluded.
        """
        # ✅ CRITICAL: Filter out RSS/XML feed URLs
        rss_feed_patterns = [
            r'\.xml$',           # Ends with .xml
            r'\.rss$',           # Ends with .rss
            r'/rss\.xml$',       # /rss.xml
            r'/feed\.xml$',      # /feed.xml
            r'/atom\.xml$',      # /atom.xml
            r'/feed/?$',         # /feed or /feed/
            r'/feeds/?$',        # /feeds or /feeds/
            r'/rss/?$',          # /rss or /rss/
        ]
        
        for pattern in rss_feed_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.debug(f"🚫 Detected RSS/XML feed URL (pattern: {pattern}): {url}")
                return True
        
        # Common category path patterns that should be excluded
        # NOTE: Patterns should only match ROOT/CATEGORY pages, not article URLs that contain these paths
        category_patterns = [
            # OpenAI specific patterns (exact matches only)
            r'/news/product-releases/?$',
            r'/news/safety-alignment/?$', 
            r'/news/security/?$',
            r'/news/engineering/?$',
            r'/news/research/?$',
            r'/news/?$',
            
            # Root category pages - only if they're truly at the end of the URL
            # Use negative lookahead to ensure nothing follows after the trailing slash
            r'/index/?$(?!.)',
            r'/blog/?$(?!.)',
            
            # Google DeepMind specific patterns
            r'/blog/page/\d+/?$',              # Pagination: /blog/page/2/
            r'/research/evals/?$',             # Research evals category
            r'/research/publications/?$',      # Publications category
            r'/research/projects/?$',          # Projects category
            r'/research/datasets/?$',          # Datasets category
            r'/discover/blog/?$',              # Blog root
            r'/discover/?$',                   # Discover root
            
            # Google Research blog label patterns (category pages with articles)
            r'/blog/label/[^/]+/?$',           # Blog label categories: /blog/label/algorithms-theory/
            r'research\.google/blog/label/',  # Full research.google domain pattern
            r'/blog/\d{4}/?$',                 # Year-based archives: /blog/2026/, /blog/2025/
            r'research\.google/blog/\d{4}',   # Full year archive pattern
            
            # General pagination patterns
            r'/page/\d+/?$',                   # Generic pagination: /page/2/
            
            # General category patterns (exact matches)
            r'/category/[^/]+/?$',
            r'/tag/[^/]+/?$',
            r'/topic/[^/]+/?$',
            r'/archive/?$',
            r'/archives/?$',
            
            # Only match /posts/, /articles/, etc if they're truly root pages
            r'/posts/?$(?!.)',
            r'/articles/?$(?!.)',
        ]
        
        # Check if URL matches any category pattern
        for pattern in category_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.debug(f"🚫 Detected category URL (pattern: {pattern}): {url}")
                return True
        
        # Additional heuristic: If URL ends with /news/something/ where something is a single word,
        # it's likely a category page
        news_category_match = re.search(r'/news/([a-z-]+)/?$', url, re.IGNORECASE)
        if news_category_match:
            category_name = news_category_match.group(1)
            # If it's a short category name (single word or hyphenated), likely a category page
            if len(category_name.split('-')) <= 3:  # e.g., "product-releases", "engineering"
                logger.debug(f"🚫 Detected news category URL: {url}")
                return True
        
        # If URL contains /index/ but doesn't have a path after it (beyond trailing slash), 
        # it's the index page itself
        if re.search(r'/index/?$', url, re.IGNORECASE):
            logger.debug(f"🚫 Detected index root URL: {url}")
            return True
            
        return False
    
    def _requires_browser_mode(self, url: str) -> bool:
        """Check if URL requires browser rendering (no fallback to basic HTTP)"""
        browser_required_domains = [
            'openai.com',
            'anthropic.com',
            'cohere.com',  # Added for Cohere research pages
            'deepmind.com',
            'google.com/ai',
            'microsoft.com/ai',
            'research.google'  # Added for Google Research blog
        ]
        return any(domain in url.lower() for domain in browser_required_domains)
    
    async def _direct_playwright_fallback(self, url: str) -> Optional[Dict[str, Any]]:
        """Direct Playwright fallback when Crawl4AI fails for browser-required sites"""
        try:
            from playwright.async_api import async_playwright
            import subprocess
            import sys
            
            logger.info(f"🎭 Starting direct Playwright scraping for {url}")
            
            # Get the Playwright browser path from the venv
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                logger.info(f"📦 Playwright install result: {result.returncode}")
            except Exception as install_error:
                logger.warning(f"⚠️ Playwright install attempt failed: {install_error}")
            
            async with async_playwright() as p:
                # Launch browser with stealth settings - let Playwright find its own installation
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )
                
                context = await browser.new_context(
                    user_agent=self._get_random_user_agent(),
                    viewport={'width': 1920, 'height': 1080},
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://www.google.com/'
                    }
                )
                
                page = await context.new_page()
                
                # Navigate to the page with a more reliable wait strategy
                logger.info(f"🌐 Navigating to {url}")
                try:
                    # Try with domcontentloaded first (faster, more reliable)
                    await page.goto(url, wait_until='domcontentloaded', timeout=120000)
                    logger.info(f"✅ Page loaded successfully: {url}")
                except Exception as goto_error:
                    logger.warning(f"⚠️ Failed with domcontentloaded, trying with load: {str(goto_error)}")
                    await page.goto(url, wait_until='load', timeout=120000)
                
                # Wait for content to fully render (but don't wait for networkidle)
                await page.wait_for_timeout(5000)  # Give JavaScript time to execute
                
                # Get the HTML content
                html_content = await page.content()
                
                # Extract text content
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                
                # Get title
                title = soup.find('h1')
                title_text = title.get_text(strip=True) if title else soup.title.string if soup.title else "No title"
                
                # Get main content
                content_element = soup.find('article') or soup.find('main') or soup.find('body')
                content_text = content_element.get_text(separator=' ', strip=True) if content_element else ""
                
                await browser.close()
                
                logger.info(f"✅ Direct Playwright scraping successful for {url}")
                
                return {
                    'title': title_text,
                    'content': content_text[:5000],  # Limit content length
                    'html': html_content,
                    'url': url
                }
                
        except Exception as e:
            logger.error(f"❌ Direct Playwright fallback failed for {url}: {str(e)}")
            logger.debug(f"📋 Playwright error traceback: {traceback.format_exc()}")
            return None
    
    async def _retry_request_with_different_agent(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Retry HTTP request with different user agents"""
        import random
        import time
        
        for attempt in range(max_retries):
            try:
                user_agent = self._get_random_user_agent()
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0',
                    'Referer': 'https://www.google.com/'
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
                                wait_time = random.uniform(2, 5) * (1.5 ** attempt)
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
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.google.com/',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'cross-site',
                    'Sec-Fetch-User': '?1'
                }
                
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, timeout=30, allow_redirects=True) as response:
                        if response.status == 403 and attempt < 2:
                            # HTTP 403 - try again with different user agent and longer delay
                            wait_time = random.uniform(2, 5) * (1.5 ** attempt)  # Exponential backoff
                            logger.warning(f"⚠️ HTTP 403 for {url}, retrying with different user agent (attempt {attempt + 1}/3) after {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status != 200:
                            logger.error(f"❌ Failed to fetch {url}: HTTP {response.status}")
                            if attempt < 2:
                                wait_time = random.uniform(1, 3) * (1.5 ** attempt)
                                await asyncio.sleep(wait_time)
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
        
        # 1. PRIORITY: Extract keywords from title first (most important)
        if title:
            # Extract capitalized words and proper nouns from title
            title_caps = re.findall(r'\b[A-Z][a-zA-Z0-9]{2,}\b', title)
            # Filter out common stop words
            stop_words = {'The', 'This', 'That', 'When', 'Where', 'What', 'How', 'Why', 'And', 'But', 'For', 'With', 'Are', 'Our', 'New', 'All'}
            title_keywords = [word for word in title_caps if word not in stop_words]
            fallback_keywords.extend(title_keywords[:5])  # Take up to 5 keywords from title
            
            # Extract quoted terms from title (often important)
            quoted_terms = re.findall(r'"([^"]+)"|\'([^\']+)\'', title)
            for quoted in quoted_terms:
                term = quoted[0] or quoted[1]
                if term:
                    fallback_keywords.append(term.strip())
        
        # 2. Extract domain-based keywords
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
            
        # 3. Extract from title and content
        text_to_analyze = f"{title} {content[:500]}"  # First 500 chars of content
        
        # Find common AI/tech terms
        tech_pattern = r'\b(?:AI|ML|GPU|CPU|API|IoT|5G|quantum|neural|machine learning|deep learning|algorithm|blockchain|cryptocurrency|robot|automation|cloud|edge computing|cybersecurity|artificial intelligence|natural language|computer vision|data science|big data|LLM|GPT|transformer)\b'
        tech_terms = re.findall(tech_pattern, text_to_analyze, re.IGNORECASE)
        fallback_keywords.extend([term.title() for term in tech_terms])
        
        # 4. Find capitalized words from content (potential proper nouns/companies)
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{2,15}\b', content[:500])
        # Filter out common words
        meaningful_caps = [word for word in capitalized_words if word not in stop_words]
        fallback_keywords.extend(meaningful_caps[:3])
        
        # 5. Add generic AI keywords if nothing found
        if not fallback_keywords:
            fallback_keywords = ['Artificial Intelligence', 'Technology', 'Innovation']
            
        # Remove duplicates and limit
        unique_keywords = []
        seen = set()
        for kw in fallback_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords[:8]
    
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
                
                # Normalize dates
                raw_date = scraped_data.get('date') or scraped_data.get('extracted_date')
                normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                
                return ScrapedArticle(
                    title=scraped_data.get('title', 'AI News Article'),
                    author=scraped_data.get('author'),
                    summary=scraped_data.get('description', 'AI news and developments'),
                    content=scraped_data.get('content', '')[:1000],
                    date=normalized_date,
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=6.0,
                    complexity_level="Medium",
                    reading_time=scraped_data.get('reading_time', 1),
                    publisher_id=publisher_id_fallback,  # Ensure publisher_id is preserved from scraped_data
                    published_date=normalized_date,
                    scraped_date=scraped_data.get('extracted_date'),
                    content_type_label=scraped_data.get('content_type', 'Blogs'),
                    topic_category_label="AI News & Updates",
                    keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
                        
                        # Normalize dates from LLM response
                        raw_date = parsed_data.get('date') or scraped_data.get('date')
                        normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=normalized_date,
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=normalized_date,
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
                        
                        # Normalize dates from LLM response
                        raw_date = parsed_data.get('date') or scraped_data.get('date')
                        normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=normalized_date,
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=normalized_date,
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
                                
                                # Normalize dates from LLM response
                                raw_date = parsed_data.get('date') or scraped_data.get('date')
                                normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                                
                                return ScrapedArticle(
                                    title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                                    author=parsed_data.get('author') or scraped_data.get('author','Unknown'),
                                    summary=parsed_data.get('summary', 'No summary available'),
                                    content=scraped_data.get('content', '')[:1000],
                                    date=normalized_date,
                                    url=scraped_data.get('url', ''),
                                    source=self._extract_domain(scraped_data.get('url', '')),
                                    significance_score=float(parsed_data.get('significance_score', 5.0)),
                                    complexity_level=parsed_data.get('complexity_level', 'Medium'),
                                    reading_time=scraped_data.get('reading_time', 1),
                                    publisher_id=publisher_id_preserved,
                                    published_date=normalized_date,
                                    scraped_date=scraped_data.get('extracted_date'),
                                    content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                                    topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                                    keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
                        
                        # Normalize dates from LLM response
                        raw_date = parsed_data.get('date') or scraped_data.get('date')
                        normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                        
                        return ScrapedArticle(
                            title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=normalized_date,
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            complexity_level=parsed_data.get('complexity_level', 'Medium'),
                            reading_time=scraped_data.get('reading_time', 1),
                            publisher_id=publisher_id_preserved,
                            published_date=normalized_date,
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
        # ✅ GUARD: Check if Ollama is available
        if not OLLAMA_AVAILABLE:
            logger.warning("⚠️ Ollama not available - skipping local LLM processing")
            return None

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
                
                # Normalize dates from LLM response
                raw_date = parsed_data.get('date') or scraped_data.get('date')
                normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                
                return ScrapedArticle(
                    title=parsed_data.get('title', scraped_data.get('title', 'Unknown')),
                    author=parsed_data.get('author') or scraped_data.get('author', 'Unknown'),
                    summary=parsed_data.get('summary', 'No summary available'),
                    content=scraped_data.get('content', '')[:1000],
                    date=normalized_date,
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=float(parsed_data.get('significance_score', 5.0)),
                    complexity_level=parsed_data.get('complexity_level', 'Medium'),
                    reading_time=scraped_data.get('reading_time', 1),
                    publisher_id=publisher_id_preserved,
                    published_date=normalized_date,
                    scraped_date=scraped_data.get('extracted_date'),
                    content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                    topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                    keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
                            keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
        
        # Normalize dates
        raw_date = scraped_data.get('date') or scraped_data.get('extracted_date')
        normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
        
        return ScrapedArticle(
            title=scraped_data.get('title', 'AI News Article'),
            author=scraped_data.get('author'),
            summary=summary,
            content=content[:1000],
            date=normalized_date,
            url=scraped_data.get('url', ''),
            source=self._extract_domain(scraped_data.get('url', '')),
            significance_score=6.0,
            complexity_level="Medium",
            reading_time=scraped_data.get('reading_time', 1),
            publisher_id=publisher_id_preserved,
            published_date=normalized_date,
            scraped_date=scraped_data.get('extracted_date'),
            content_type_label=scraped_data.get('content_type', 'Blogs'),
            topic_category_label="AI News & Updates",
            keywords=', '.join(scraped_data.get('tags', ['AI', 'Technology'])) if isinstance(scraped_data.get('tags'), list) else scraped_data.get('tags', 'AI, Technology'),
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
        """Scrape multiple sources sequentially with resource monitoring"""
        logger.info(f"🔄 Scraping {len(source_urls)} sources SEQUENTIALLY with Claude processing...")
        
        articles = []
        for i, url in enumerate(source_urls, 1):
            try:
                # Check for ABORT condition first
                if self.resource_monitor and self.resource_monitor.should_abort():
                    logger.error(f"🛑 ABORTING scraping at article {i}/{len(source_urls)} due to resource constraints")
                    logger.error(f"🛑 Successfully processed {len(articles)} articles before abort")
                    break
                
                # Check resources before each scrape
                if self.resource_monitor:
                    if self.resource_monitor.should_throttle():
                        logger.warning(f"⏳ Resources constrained, waiting before article {i}/{len(source_urls)}...")
                        await self.resource_monitor.wait_for_resources(timeout=120)
                    
                    if i % 5 == 0:  # Log every 5 articles
                        self.resource_monitor.log_status()
                
                logger.info(f"📰 Processing article {i}/{len(source_urls)}: {url}")
                article = await self.scrape_article(url, llm_model='claude')
                if article:
                    articles.append(article)
                    logger.info(f"✅ Article {i}/{len(source_urls)} complete: {article.title[:50]}...")
                else:
                    logger.warning(f"⚠️ Article {i}/{len(source_urls)} returned None")
                
                # Small delay between scrapes to prevent overwhelming the system
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to process article {i}/{len(source_urls)} ({url}): {str(e)}")
                continue
        
        logger.info(f"🎉 PROCESSING COMPLETE: {len(articles)}/{len(source_urls)} articles processed")
        return articles
    
    async def scrape_multiple_sources_with_publisher_id(self, article_data: List[Dict], llm_model: str = 'claude', batch_size: int = 3) -> List[ScrapedArticle]:
        """Scrape multiple sources with publisher_id using small batches for controlled resource usage"""
        total = len(article_data)
        logger.info(f"🔄 Scraping {total} sources SEQUENTIALLY with LLM: {llm_model} (batch_size={batch_size})...")
        
        if self.resource_monitor:
            self.resource_monitor.log_status()
            logger.info(f"💡 Using batch size of {batch_size} to control resource usage")
        
        articles = []
        
        # Process in small batches to control resource usage
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = article_data[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            # Check for ABORT condition before starting batch
            if self.resource_monitor and self.resource_monitor.should_abort():
                logger.error(f"🛑 ABORTING scraping at batch {batch_num}/{total_batches} due to resource constraints")
                logger.error(f"🛑 Successfully processed {len(articles)} articles before abort")
                break
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} (articles {batch_start+1}-{batch_end}/{total})")
            
            # Check resources before each batch
            if self.resource_monitor:
                if self.resource_monitor.should_throttle():
                    logger.warning(f"⏳ Resources constrained before batch {batch_num}, waiting...")
                    await self.resource_monitor.wait_for_resources(timeout=180)
                self.resource_monitor.log_status()
            
            # Process batch sequentially (not in parallel)
            for i, data in enumerate(batch, 1):
                try:
                    # Check for ABORT condition during batch processing
                    if self.resource_monitor and self.resource_monitor.should_abort():
                        logger.error(f"🛑 ABORTING during batch {batch_num} due to resource constraints")
                        logger.error(f"🛑 Successfully processed {len(articles)} articles before abort")
                        return articles  # Exit immediately
                    
                    article_num = batch_start + i
                    url = data['url']
                    publisher_id = data['publisher_id']
                    source_category = data.get('source_category', 'general')
                    
                    logger.info(f"📰 Processing article {article_num}/{total}: {url}")
                    
                    article = await self.scrape_article_with_publisher_id(
                        url, publisher_id, source_category, llm_model
                    )
                    
                    if article:
                        articles.append(article)
                        logger.info(f"✅ Article {article_num}/{total} complete: {article.title[:50]}...")
                    else:
                        logger.warning(f"⚠️ Article {article_num}/{total} returned None")
                    
                    # Small delay between articles
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process article {article_num}/{total}: {str(e)}")
                    continue
            
            # Longer delay between batches
            if batch_end < total:
                logger.info(f"⏸️ Batch {batch_num} complete, pausing before next batch...")
                await asyncio.sleep(2)
        
        logger.info(f"🎉 LLM PROCESSING COMPLETE: {len(articles)}/{total} articles processed using {llm_model}")
        
        if self.resource_monitor:
            self.resource_monitor.log_status()
        
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
        """Parse RSS feed and extract article URLs with deduplication, with retry logic"""
        import random
        
        # Try up to 3 times with different user agents
        for attempt in range(3):
            try:
                headers = {
                    'User-Agent': self._get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.google.com/',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                logger.info(f"🔄 Fetching RSS feed: {feed_url} (attempt {attempt + 1}/3)")
                response = await asyncio.to_thread(requests.get, feed_url, headers=headers, timeout=20, allow_redirects=True)
                
                # Check if we got a 403 and should retry
                if response.status_code == 403 and attempt < 2:
                    wait_time = random.uniform(2, 4) * (1.5 ** attempt)
                    logger.warning(f"⚠️ HTTP 403 for RSS feed {feed_url}, retrying after {wait_time:.1f}s (attempt {attempt + 1}/3)")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                logger.info(f"✅ RSS feed fetched successfully: {feed_url} (status: {response.status_code})")
                
                # Check if content type suggests HTML instead of RSS/XML
                content_type = response.headers.get('content-type', '').lower()
                is_likely_html = 'html' in content_type or (
                    'xml' not in content_type and 
                    'rss' not in content_type and
                    response.text.strip().startswith('<!doctype html') or 
                    response.text.strip().startswith('<html')
                )
                
                feed = feedparser.parse(response.content)
                
                # Log feed parsing details
                if hasattr(feed, 'bozo') and feed.bozo:
                    logger.warning(f"⚠️ RSS feed has parsing issues: {feed_url}")
                    if hasattr(feed, 'bozo_exception'):
                        logger.warning(f"⚠️ Feed exception: {feed.bozo_exception}")
                
                logger.info(f"📰 Found {len(feed.entries)} entries in RSS feed: {feed_url}")
                
                # If RSS parsing failed and it looks like HTML, try scraping the page for article links
                if len(feed.entries) == 0 and (is_likely_html or (hasattr(feed, 'bozo') and feed.bozo)):
                    logger.info(f"🔄 RSS parsing failed, attempting to scrape HTML page for article links: {feed_url}")
                    try:
                        from bs4 import BeautifulSoup
                        from urllib.parse import urljoin
                        
                        # Check if this URL requires browser rendering
                        html_content = response.text
                        if self._requires_browser_mode(feed_url):
                            logger.info(f"🌐 Browser mode required for list page, using Playwright: {feed_url}")
                            try:
                                from playwright.async_api import async_playwright
                                
                                async with async_playwright() as p:
                                    browser = await p.chromium.launch(headless=True)
                                    context = await browser.new_context(
                                        viewport={'width': 1920, 'height': 1080},
                                        user_agent=self._get_random_user_agent()
                                    )
                                    page = await context.new_page()
                                    
                                    logger.info(f"🔄 Loading list page with Playwright: {feed_url}")
                                    await page.goto(feed_url, wait_until='domcontentloaded', timeout=120000)
                                    await asyncio.sleep(5)  # Wait for dynamic content
                                    
                                    html_content = await page.content()
                                    await browser.close()
                                    logger.info(f"✅ List page rendered with Playwright: {feed_url}")
                            except Exception as pw_error:
                                logger.warning(f"⚠️ Playwright rendering failed for list page, using basic HTML: {str(pw_error)}")
                        
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Try to find article links - common patterns
                        article_links = []
                        
                        # Exclusion patterns for non-article pages
                        exclude_patterns = [
                            '/team/', '/about/', '/careers/', '/contact/', '/privacy/', '/terms/',
                            '/category/', '/tag/', '/author/', '/subscribe/', '/login/', '/signup/',
                            '#', 'mailto:', 'tel:', 'javascript:',
                            # ✅ Exclude RSS/XML feed files
                            '.xml', '.rss', '/feed', '/feeds', '/rss'
                        ]
                        
                        # Inclusion patterns for article-like URLs (these get priority)
                        include_patterns = ['/index/', '/blog/', '/news/', '/article/', '/post/', '/research/']
                        
                        # Look for article links in common HTML structures
                        for link in soup.find_all('a', href=True):
                            href = link.get('href', '')
                            # Make absolute URLs
                            if href.startswith('/'):
                                href = urljoin(feed_url, href)
                            
                            # Skip if URL doesn't start with http
                            if not href.startswith('http'):
                                continue
                            
                            # Skip if same as feed URL
                            if href.rstrip('/') == feed_url.rstrip('/'):
                                continue
                            
                            # Skip if URL contains excluded patterns
                            if any(pattern in href.lower() for pattern in exclude_patterns):
                                continue
                            
                            # ✅ CRITICAL: Skip category/root URLs first (before checking inclusion patterns)
                            if self._is_category_or_root_url(href):
                                continue
                            
                            # ✅ IMPROVED: Include if URL matches inclusion patterns OR if it doesn't match exclusions
                            # This allows article URLs that don't follow standard patterns (like /research/paper-name)
                            if any(pattern in href for pattern in include_patterns):
                                article_links.append(href)
                                continue
                            
                            # ✅ NEW: Also include URLs from the same domain that pass all filters
                            # This catches article URLs that don't match standard patterns
                            from urllib.parse import urlparse
                            feed_domain = urlparse(feed_url).netloc
                            href_domain = urlparse(href).netloc
                            if feed_domain == href_domain:
                                article_links.append(href)
                                continue
                            
                            # For research pages, look for article indicators
                            if '/research/' in href.lower() or '/paper/' in href.lower():
                                # Check link text or parent elements for article indicators
                                link_text = link.get_text(strip=True).lower()
                                parent_classes = ' '.join(link.parent.get('class', [])).lower() if link.parent else ''
                                
                                # Look for article-like indicators in text or structure
                                article_indicators = [
                                    'read more', 'learn more', 'full paper', 'article', 'research paper',
                                    'publication', 'paper', 'study', 'report', 'announcement'
                                ]
                                
                                # If link text suggests it's an article, or parent suggests it's in article list
                                if (any(indicator in link_text for indicator in article_indicators) or
                                    'article' in parent_classes or 'post' in parent_classes or 
                                    'card' in parent_classes or 'item' in parent_classes):
                                    article_links.append(href)
                        
                        # Remove duplicates while preserving order
                        article_links = list(dict.fromkeys(article_links))
                        
                        # ✅ CRITICAL: For browser-mode domains (especially openai.com), filter out category URLs
                        if self._requires_browser_mode(feed_url):
                            initial_count = len(article_links)
                            article_links = [url for url in article_links if not self._is_category_or_root_url(url)]
                            filtered_count = initial_count - len(article_links)
                            if filtered_count > 0:
                                logger.info(f"🚫 Filtered {filtered_count} category/root URLs from browser-mode domain: {feed_url}")
                        
                        logger.info(f"🔗 Extracted {len(article_links)} article links from HTML page: {feed_url}")
                        
                        if article_links:
                            # Check all extracted links against database (no artificial limit)
                            from db_service import get_database_service
                            url_existence = get_database_service().check_multiple_urls_exist(article_links)
                            new_urls = [url for url in article_links if not url_existence.get(url, False)]
                            
                            logger.info(f"✅ URL Deduplication (from HTML): {len(new_urls)} new, {len(article_links) - len(new_urls)} existing")
                            return new_urls  # Return all new URLs, no limit
                    except Exception as scrape_error:
                        logger.error(f"❌ Failed to scrape HTML page for article links: {str(scrape_error)}")
                
                # Extract all RSS entries (no limit)
                all_urls = [entry.get('link') or entry.get('id') for entry in feed.entries if (entry.get('link') or entry.get('id', '')).startswith('http')]
                
                # ✅ NEW: For browser-mode domains, scrape pagination/category pages for article links
                if self._requires_browser_mode(feed_url) and all_urls:
                    pagination_urls = [url for url in all_urls if self._is_category_or_root_url(url)]
                    if pagination_urls:
                        logger.info(f"🔍 Found {len(pagination_urls)} pagination/category pages to scrape for article links")
                        extracted_article_links = []
                        
                        for pagination_url in pagination_urls[:5]:  # Limit to first 5 pagination pages
                            try:
                                logger.info(f"📄 Scraping pagination page for articles: {pagination_url}")
                                from playwright.async_api import async_playwright
                                
                                async with async_playwright() as p:
                                    browser = await p.chromium.launch(headless=True)
                                    context = await browser.new_context(
                                        viewport={'width': 1920, 'height': 1080},
                                        user_agent=self._get_random_user_agent()
                                    )
                                    page = await context.new_page()
                                    
                                    await page.goto(pagination_url, wait_until='domcontentloaded', timeout=60000)
                                    await asyncio.sleep(3)  # Wait for dynamic content
                                    
                                    html_content = await page.content()
                                    await browser.close()
                                    
                                    # Extract article links from pagination page
                                    from bs4 import BeautifulSoup
                                    from urllib.parse import urljoin
                                    soup = BeautifulSoup(html_content, 'html.parser')
                                    
                                    for link in soup.find_all('a', href=True):
                                        href = link.get('href', '')
                                        if href.startswith('/'):
                                            href = urljoin(pagination_url, href)
                                        
                                        if not href.startswith('http'):
                                            continue
                                        
                                        # Only include article-like URLs, not more category/pagination URLs
                                        if not self._is_category_or_root_url(href):
                                            if any(pattern in href for pattern in ['/index/', '/blog/', '/news/', '/article/', '/post/', '/research/']):
                                                extracted_article_links.append(href)
                                    
                                    logger.info(f"✅ Extracted {len(extracted_article_links)} article links from {pagination_url}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to scrape pagination page {pagination_url}: {str(e)}")
                        
                        # Add extracted article links to all_urls
                        if extracted_article_links:
                            extracted_article_links = list(dict.fromkeys(extracted_article_links))  # Remove duplicates
                            all_urls.extend(extracted_article_links)
                            logger.info(f"📝 Added {len(extracted_article_links)} article links from pagination pages")
                
                # ✅ CRITICAL: Filter out category/root URLs (especially important for browser-mode domains)
                initial_count = len(all_urls)
                all_urls = [url for url in all_urls if not self._is_category_or_root_url(url)]
                filtered_count = initial_count - len(all_urls)
                
                if filtered_count > 0:
                    logger.info(f"🚫 Filtered {filtered_count} category/root URLs from RSS feed: {feed_url}")
                    if self._requires_browser_mode(feed_url):
                        logger.info(f"   ↳ Browser-mode domain detected - category filtering is critical")
                
                if not all_urls:
                    logger.warning(f"⚠️ No valid article URLs found in RSS feed: {feed_url}")
                    return []
                
                logger.info(f"🔗 Extracted {len(all_urls)} URLs from RSS feed (after category filter): {feed_url}")
                
                from db_service import get_database_service
                url_existence = get_database_service().check_multiple_urls_exist(all_urls)
                new_urls = [url for url in all_urls if not url_existence.get(url, False)]
                
                logger.info(f"✅ URL Deduplication: {len(new_urls)} new, {len(all_urls) - len(new_urls)} existing")
                return new_urls  # Return all new URLs, no limit
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Request error parsing RSS feed {feed_url} (attempt {attempt + 1}/3): {str(e)}")
                if attempt < 2:
                    wait_time = random.uniform(2, 4) * (1.5 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                return []
            except Exception as e:
                logger.error(f"❌ Error parsing RSS feed {feed_url}: {str(e)}")
                logger.debug(f"📋 RSS parse error traceback: {traceback.format_exc()}")
                return []
        
        # If all attempts failed
        logger.error(f"❌ All {3} attempts failed for RSS feed: {feed_url}")
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

    async def search_and_insert_tavily_articles(
        self, 
        query: str, 
        max_results: int = 10,
        enrich_with_llm: bool = False,
        llm_model: str = 'gemini',
        include_domains: List[str] = None,
        exclude_domains: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search Tavily API for articles and insert them into the database.
        Returns statistics about the operation.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to fetch (1-50)
            enrich_with_llm: Whether to enhance Tavily results with LLM processing
            llm_model: Which LLM to use for enrichment ('claude', 'gemini', 'ollama')
            include_domains: Optional list of domains to include in search
            exclude_domains: Optional list of domains to exclude from search
        
        Returns:
            Dict with success status, counts, and search metadata
        """
        try:
            logger.info(f"🔍 TAVILY SEARCH: '{query}' (max_results={max_results}, enrich={enrich_with_llm})")
            
            if not self.tavily_api_key:
                logger.error("❌ TAVILY_API_KEY not set")
                return {
                    "success": False,
                    "message": "Tavily API key not configured",
                    "articles_inserted": 0
                }
            
            # Get database service
            from db_service import get_database_service
            db_service = get_database_service()
            
            # Create search record in tavily_searches table
            search_params = {
                "include_domains": include_domains or [],
                "exclude_domains": exclude_domains or []
            }
            
            try:
                search_id = db_service.execute_query(
                    """
                    INSERT INTO tavily_searches 
                    (query, max_results, enrich_with_llm, llm_model, search_params, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (query, max_results, enrich_with_llm, llm_model, json.dumps(search_params))
                )['id']
                logger.info(f"📝 Created Tavily search record: ID {search_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create search record: {e}")
                search_id = None
            
            # Call Tavily API
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": min(max_results, 50),  # Tavily max is 50
                "search_depth": "advanced",  # Use advanced search for better results
                "include_answer": False,  # We don't need the AI-generated answer
                "include_raw_content": False,  # We'll scrape full content ourselves if needed
            }
            
            # Add domain filters if provided
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains
            
            logger.info(f"🌐 Calling Tavily API: {self.tavily_api_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.tavily_api_url,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ Tavily API error {response.status}: {error_text}")
                        
                        # Update search record with error
                        if search_id:
                            try:
                                db_service.execute_query(
                                    """
                                    UPDATE tavily_searches 
                                    SET completed_at = NOW(), error_message = %s
                                    WHERE id = %s
                                    """,
                                    (f"API error {response.status}: {error_text[:500]}", search_id)
                                )
                            except:
                                pass
                        
                        return {
                            "success": False,
                            "message": f"Tavily API error: {response.status}",
                            "articles_inserted": 0
                        }
                    
                    tavily_results = await response.json()
            
            # Extract results array
            results = tavily_results.get('results', [])
            logger.info(f"✅ Tavily returned {len(results)} results")
            
            if not results:
                logger.warning("⚠️ No results from Tavily")
                
                # Update search record
                if search_id:
                    try:
                        db_service.execute_query(
                            """
                            UPDATE tavily_searches 
                            SET articles_found = 0, articles_inserted = 0, articles_skipped = 0,
                                completed_at = NOW()
                            WHERE id = %s
                            """,
                            (search_id,)
                        )
                    except:
                        pass
                
                return {
                    "success": True,
                    "message": "No results found",
                    "query": query,
                    "articles_found": 0,
                    "articles_inserted": 0,
                    "articles_skipped": 0,
                    "tavily_search_id": search_id
                }
            
            # Check which URLs already exist
            article_urls = [r.get('url') for r in results if r.get('url')]
            url_existence = db_service.check_multiple_urls_exist(article_urls)
            
            articles_inserted = 0
            articles_skipped = 0
            
            # Process each result
            for i, result in enumerate(results, 1):
                try:
                    url = result.get('url')
                    if not url:
                        logger.warning(f"⚠️ Result {i}/{len(results)} has no URL, skipping")
                        continue
                    
                    # Skip if URL already exists (deduplication)
                    if url_existence.get(url, False):
                        logger.debug(f"⏭️ Skipping existing URL: {url}")
                        articles_skipped += 1
                        continue
                    
                    logger.info(f"📰 Processing Tavily result {i}/{len(results)}: {url}")
                    
                    # Extract domain for source/publisher
                    domain = self._extract_domain(url)
                    
                    # Detect content type from URL
                    content_type = self._detect_content_type(
                        url=url,
                        html="",
                        title=result.get('title', ''),
                        content=result.get('content', ''),
                        media=[]
                    )
                    
                    # Prepare base article data from Tavily
                    tavily_data = {
                        'url': url,
                        'title': result.get('title', 'Unknown Article'),
                        'description': result.get('content', '')[:500],  # Use as summary
                        'content': result.get('content', ''),
                        'author': 'Tavily Search',  # Default author
                        'date': result.get('published_date'),  # May be None
                        'tags': [],
                        'image_url': None,
                        'image_source': None,
                        'reading_time': max(1, len(result.get('content', '').split()) // 200),  # Estimate
                        'content_type': content_type,
                        'extraction_method': 'tavily-api'
                    }
                    
                    # Optional: Enrich with LLM
                    if enrich_with_llm:
                        logger.info(f"🤖 Enriching with {llm_model.upper()}...")
                        
                        if llm_model == 'claude':
                            enriched = await self.process_with_claude(tavily_data)
                        elif llm_model == 'gemini':
                            enriched = await self.process_with_gemini(tavily_data)
                        elif llm_model == 'ollama':
                            enriched = await self.process_with_ollama(tavily_data)
                        else:
                            logger.warning(f"⚠️ Unknown LLM {llm_model}, using Claude")
                            enriched = await self.process_with_claude(tavily_data)
                        
                        if enriched:
                            # Use enriched data
                            article_data = {
                                'title': enriched.title,
                                'author': enriched.author or 'Tavily Search',
                                'summary': enriched.summary,
                                'content': enriched.content,
                                'url': enriched.url,
                                'source': enriched.source,
                                'significance_score': enriched.significance_score,
                                'complexity_level': enriched.complexity_level,
                                'published_date': enriched.published_date,
                                'reading_time': enriched.reading_time,
                                'content_type_label': enriched.content_type_label,
                                'topic_category_label': enriched.topic_category_label,
                                'scraped_date': datetime.now(timezone.utc).isoformat(),
                                'created_date': datetime.now(timezone.utc).isoformat(),
                                'updated_date': datetime.now(timezone.utc).isoformat(),
                                'llm_processed': enriched.llm_processed or f'tavily+{llm_model}',
                                'keywords': enriched.keywords,
                                'image_url': enriched.image_url,
                                'image_source': enriched.image_source
                            }
                            logger.info(f"✅ LLM enrichment successful")
                            logger.debug(f"   ↳ Enriched title: {enriched.title[:60]}...")
                        else:
                            logger.warning(f"⚠️ LLM enrichment failed, using basic Tavily data")
                            enrich_with_llm = False  # Disable for remaining articles
                    else:
                        # Use basic Tavily data
                        # Map Tavily score (0.0-1.0) to significance_score (1-10)
                        tavily_score = result.get('score', 0.5)
                        significance_score = max(1.0, min(10.0, tavily_score * 10))
                        
                        # Normalize date
                        raw_date = result.get('published_date')
                        normalized_date = self._normalize_date(raw_date) or datetime.now(timezone.utc).isoformat()
                        
                        article_data = {
                            'title': result.get('title', 'Unknown Article'),
                            'author': 'Tavily Search',
                            'summary': result.get('content', '')[:500] or 'No summary available',
                            'content': result.get('content', ''),
                            'url': url,
                            'source': domain,
                            'significance_score': significance_score,
                            'complexity_level': 'Medium',
                            'published_date': normalized_date,
                            'reading_time': max(1, len(result.get('content', '').split()) // 200),
                            'content_type_label': content_type,
                            'topic_category_label': 'AI News & Updates',  # Default category
                            'scraped_date': datetime.now(timezone.utc).isoformat(),
                            'created_date': datetime.now(timezone.utc).isoformat(),
                            'updated_date': datetime.now(timezone.utc).isoformat(),
                            'llm_processed': 'tavily-only',
                            'keywords': 'AI, Technology, Tavily Search',
                            'image_url': None,
                            'image_source': None
                        }
                    logger.info(f"📝 Prepared tavily article data for insertion: {article_data['title'][:60]}...")
                    # Insert article (publisher will be auto-created from URL/source)
                    success = db_service.insert_article(article_data)
                    
                    if success:
                        articles_inserted += 1
                        logger.info(f"✅ Inserted: {article_data['title'][:60]}...")
                    else:
                        articles_skipped += 1
                        logger.debug(f"⏭️ Skipped (duplicate or error): {url}")
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing Tavily result {i}: {str(e)}")
                    articles_skipped += 1
                    continue
            
            # Update search record with final counts
            if search_id:
                try:
                    db_service.execute_query(
                        """
                        UPDATE tavily_searches 
                        SET articles_found = %s, articles_inserted = %s, articles_skipped = %s,
                            completed_at = NOW()
                        WHERE id = %s
                        """,
                        (len(results), articles_inserted, articles_skipped, search_id)
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to update search record: {e}")
            
            logger.info(f"🎉 TAVILY SEARCH COMPLETE: {articles_inserted}/{len(results)} inserted, {articles_skipped} skipped")
            
            return {
                "success": True,
                "message": "Search completed successfully",
                "query": query,
                "articles_found": len(results),
                "articles_inserted": articles_inserted,
                "articles_skipped": articles_skipped,
                "enrich_with_llm": enrich_with_llm,
                "llm_model": llm_model if enrich_with_llm else None,
                "tavily_search_id": search_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Tavily search failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update search record with error
            if search_id:
                try:
                    db_service.execute_query(
                        """
                        UPDATE tavily_searches 
                        SET completed_at = NOW(), error_message = %s
                        WHERE id = %s
                        """,
                        (str(e)[:500], search_id)
                    )
                except:
                    pass
            
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "articles_inserted": 0,
                "tavily_search_id": search_id
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
            
            # ✅ Limit to 100 URLs per run to prevent resource exhaustion
            total_urls = len(all_article_data)
            if total_urls > 100:
                logger.warning(f"⚠️ Found {total_urls} URLs, limiting to 100 per run")
                logger.info(f"💡 Run the scraper again to process remaining {total_urls - 100} URLs")
                all_article_data = all_article_data[:100]
            else:
                logger.info(f"📊 Processing all {total_urls} URLs (within 100 limit)")
            
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
    
