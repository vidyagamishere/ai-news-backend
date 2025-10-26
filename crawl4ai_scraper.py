#!/usr/bin/env python3
"""
AI News Scraper using Crawl4AI and Claude as specified in functional requirements.
Implements the exact scraping process: Crawl4AI -> Claude -> Structured Output
"""

import os
import tempfile
import logging
import feedparser
import re
import json

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

class Crawl4AIScraper:
    """AI News Scraper using Crawl4AI and Claude LLM"""
    
    def __init__(self):
        self.extraction_warnings = []  # Track extraction warnings for reporting
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')  # Add HuggingFace API key
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.huggingface_api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"  # Updated model
        self.headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        if not self.anthropic_api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not set - Claude processing will not work")
        
        if not self.google_api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set - Gemini processing will not be available")
        
        if not self.huggingface_api_key:
            logger.warning("⚠️ HUGGINGFACE_API_KEY not set - HuggingFace processing will not be available")

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
                user_data_dir=os.path.join(session_dir, 'user_data'),  # Use session directory for user data
                downloads_path=os.path.join(session_dir, 'downloads')   # Use session directory for downloads
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
                crawler = AsyncWebCrawler(
                    crawler_strategy=crawler_strategy,
                    always_by_pass_cache=True,
                    verbose=False,  # Reduce logging for Railway
                    cache_mode="disabled",  # Disable caching to avoid permission issues
                    timeout=30  # Shorter timeout for Railway
                )
                
                # Modern Crawl4AI versions handle lifecycle automatically
                # await crawler.astart()  # Not needed in newer versions
                
                # Perform the crawl with advanced options
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
                    magic=True  # Enable AI-powered content extraction
                )
                
                if not result.success:
                    logger.error(f"❌ Crawl4AI failed for {url}: {result.error_message}")
                    return await self._fallback_scrape(url)
                
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
                "content_type": content_type  # Add detected content type
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
            
            # Extract host/creator name
            host_patterns = [
                r'"creator[^"]*":"([^"]+)"',
                r'"author[^"]*":"([^"]+)"',
                r'"artist[^"]*":"([^"]+)"',
                r'<meta name="author" content="([^"]*)"',
                r'"podcast[^"]*name":"([^"]+)"',
                r'"show[^"]*name":"([^"]+)"',
            ]
            
            for pattern in host_patterns:
                host_match = re.search(pattern, html, re.IGNORECASE)
                if host_match:
                    metadata['host'] = host_match.group(1)[:100]  # Limit host name length
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
                            "content_type": content_type  # Add detected content type
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
        
        # Check for media elements (if available)
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

    async def process_with_claude(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data from Crawl4AI to Claude for processing.
        Use specific prompt to get structured output with key details.
        """
        try:
            logger.info(f"🤖 CLAUDE PROCESSING: {scraped_data.get('title', 'Unknown')[:60]}...")
            
            # Create specific prompt for structured output with enhanced data
            extraction_method = scraped_data.get('extraction_method', 'unknown')
            author_info = scraped_data.get('author', 'Not specified')
            tags_info = scraped_data.get('tags', [])
            logger.info(f"🤖  URL : {scraped_data.get('url', '')}, Extraction Method: {extraction_method}, Author: {author_info}, Tags : {tags_info}...")
            logger.info(f"🤖  Content Txt: {scraped_data.get('content', '')[:100]}...")


            prompt = f"""You are an expert Artificial Intelligence technology expert analyst and content classifier. Analyze the following scraped content and extract key information in JSON format.

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
                "topic_category_label": "Classify the core subject matter into ONLY ONE of the following 10 categories: Generative AI,AI Applications,AI Start Ups,AI Infrastructure,Cloud Computing,Machine Learning,AI Safety and Governance,Robotics,Internet Of Things (IoT),Quantum AI, Future Technology"
            }}
            Focus on Artificial Intelligence, machine learning, technology, and innovation content. Be accurate and provide meaningful analysis.
            If this is not AI/tech related content, set significance_score to 1-3."""

            # Call Claude API - using Claude 3 Haiku (only available model with current API key)
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
                    keywords=scraped_data.get('tags', [])
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
                            publisher_id=publisher_id_preserved,  # Ensure publisher_id is preserved from scraped_data
                            published_date=parsed_data.get('date') or scraped_data.get('date'),
                            scraped_date=scraped_data.get('extracted_date'),
                            content_type_label=parsed_data.get('content_type_label', scraped_data.get('content_type', 'Blogs')),
                            topic_category_label=parsed_data.get('topic_category_label', 'Generative AI'),
                            keywords=scraped_data.get('tags', [])
                        )

                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ CLAUDE JSON ERROR: Failed to parse response: {e}")
                        logger.error(f"Claude raw response: {content[:200]}...")
                        
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
            # Updated to Gemini 2.0 Flash (better performance)
            model_name = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash-latest"
            logger.info(f"🤖 GEMINI PROCESSING ({model_name}): {scraped_data.get('title', 'Unknown')[:60]}...")

            if not self.google_api_key:
                logger.error("❌ GOOGLE_API_KEY is not set. Cannot process with Gemini.")
                return None

            prompt = f"""You are an expert Artificial Intelligence technology expert analyst and content classifier. Analyze the following scraped content and extract key information in JSON format.

            Content Title: {scraped_data.get('title', '')}
            Content Description: {scraped_data.get('description', '')}
            Author: {scraped_data.get('author', 'Not specified')}
            Existing Tags: {scraped_data.get('tags', [])}
            Content Text: {scraped_data.get('content', '')[:4000]}
            Source URL: {scraped_data.get('url', '')}
            Extraction Method: {scraped_data.get('extraction_method', 'unknown')}

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
                "topic_category_label": "Classify the core subject matter into ONLY ONE of the following 10 categories: Generative AI,AI Applications,AI Start Ups,AI Infrastructure,Cloud Computing,Machine Learning,AI Safety and Governance,Robotics,Internet Of Things (IoT),Quantum AI, Future Technology"
            }}
            Focus on Artificial Intelligence, machine learning, technology, and innovation content. Be accurate and provide meaningful analysis.
            If this is not AI/tech related content, set significance_score to 1-3."""

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
                        parsed_data = json.loads(content)
                        logger.info(f"✅ GEMINI SUCCESS: Processed Title: {parsed_data.get('title', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score', 'N/A')})")

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
                            llm_processed=model_name
                        )

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ GEMINI JSON ERROR: Failed to parse response: {e}")
                        logger.error(f"Gemini raw response: {content[:200]}...")
                        fixed_content = self._fix_json_formatting(content)
                        if fixed_content:
                            try:
                                parsed_data = json.loads(fixed_content)
                                logger.info(f"✅ GEMINI SUCCESS (after JSON fix): Processed {parsed_data.get('title', 'Unknown')[:50]}...")
                                # You would reconstruct the ScrapedArticle here as above
                            except json.JSONDecodeError:
                                logger.error("❌ JSON fix attempt failed for Gemini, skipping article")
                                return None
                        else:
                            return None

        except Exception as e:
            logger.error(f"❌ GEMINI PROCESSING FAILED: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def process_with_huggingface(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data to HuggingFace Llama 3.1 8B Instruct for processing.
        This model is excellent for content classification, metadata extraction, and analysis.
        """
        try:
            logger.info(f"🤖 HUGGINGFACE PROCESSING (Llama-3.1-8B-Instruct): {scraped_data.get('title', 'Unknown')[:60]}...")

            if not self.huggingface_api_key:
                logger.error("❌ HUGGINGFACE_API_KEY is not set. Cannot process with HuggingFace.")
                return None

            extraction_method = scraped_data.get('extraction_method', 'unknown')
            author_info = scraped_data.get('author', 'Not specified')
            tags_info = scraped_data.get('tags', [])
            
            prompt = f"""You are an expert AI content analyst. Analyze this content and return ONLY a valid JSON object.

Content Title: {scraped_data.get('title', '')}
Content Description: {scraped_data.get('description', '')}
Author: {author_info}
Tags: {tags_info}
Content Text: {scraped_data.get('content', '')[:4000]}
Source URL: {scraped_data.get('url', '')}

CRITICAL INSTRUCTIONS:
1. Detect content type from URL patterns:
   - YouTube/Vimeo/video platforms → "Videos"
   - Spotify/podcast platforms → "Podcasts"
   - News/blog sites → "Blogs"
2. Look for publication dates in text (e.g., "Published October 10, 2025", "Oct 10, 2025")
3. Return ONLY valid JSON, no markdown, no explanation

JSON Schema:
{{
    "title": "clear article headline",
    "author": "author name or null",
    "summary": "4-5 sentence summary of key points",
    "date": "publication date in YYYY-MM-DD format or null",
    "content_type_label": "Videos|Podcasts|Blogs",
    "significance_score": 1-10,
    "complexity_level": "Low|Medium|High",
    "key_topics": ["topic1", "topic2", "topic3"],
    "topic_category_label": "Generative AI|AI Applications|AI Start Ups|AI Infrastructure|Cloud Computing|Machine Learning|AI Safety and Governance|Robotics|Internet Of Things (IoT)|Quantum AI|Future Technology"
}}

Return ONLY the JSON object, nothing else."""

            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }

            async with aiohttp.ClientSession() as session:
                # HuggingFace Inference API may need retry for model loading
                for attempt in range(3):
                    async with session.post(
                        self.huggingface_api_url,
                        headers=headers,
                        json=payload,
                        timeout=90
                    ) as response:
                        if response.status == 503:
                            # Model is loading, wait and retry
                            logger.warning(f"⏳ HuggingFace model loading, attempt {attempt + 1}/3...")
                            await asyncio.sleep(20)
                            continue
                        elif response.status != 200:
                            response_text = await response.text()
                            logger.error(f"❌ HUGGINGFACE API ERROR: HTTP {response.status} - {response_text}")
                            return None

                        result = await response.json()
                        
                        # Extract generated text
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')
                        else:
                            generated_text = result.get('generated_text', '')
                        
                        if not generated_text:
                            logger.error(f"❌ HUGGINGFACE ERROR: No generated text in response")
                            return None
                        
                        try:
                            # Clean up response to extract JSON
                            content_clean = generated_text.strip()
                            
                            # Remove markdown code blocks if present
                            if '```json' in content_clean:
                                content_clean = content_clean.split('```json')[1].split('```')[0]
                            elif '```' in content_clean:
                                content_clean = content_clean.split('```')[1].split('```')[0]
                            
                            # Find JSON object boundaries
                            start_idx = content_clean.find('{')
                            end_idx = content_clean.rfind('}')
                            
                            if start_idx != -1 and end_idx != -1:
                                content_clean = content_clean[start_idx:end_idx + 1]
                            
                            parsed_data = json.loads(content_clean)
                            
                            logger.info(f"✅ HUGGINGFACE SUCCESS: {parsed_data.get('title', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score')})")

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
                                llm_processed='huggingface-llama-3.1-8b-instruct'
                            )

                        except json.JSONDecodeError as e:
                            logger.error(f"❌ HUGGINGFACE JSON ERROR: {e}")
                            logger.error(f"Raw response: {generated_text[:300]}...")
                            
                            # Try to fix JSON formatting
                            fixed_content = self._fix_json_formatting(content_clean)
                            if fixed_content:
                                try:
                                    parsed_data = json.loads(fixed_content)
                                    logger.info(f"✅ HUGGINGFACE SUCCESS (after JSON fix): {parsed_data.get('title', 'Unknown')[:50]}...")
                                    
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
                                        llm_processed='huggingface-llama-3.1-8b-instruct'
                                    )
                                except json.JSONDecodeError:
                                    logger.error("❌ JSON fix failed for HuggingFace")
                                    return None
                            return None
                        
                        break  # Success, exit retry loop
                
                # All retries exhausted
                logger.error("❌ HUGGINGFACE: All retry attempts exhausted")
                return None

        except Exception as e:
            logger.error(f"❌ HUGGINGFACE PROCESSING FAILED: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def scrape_article(self, url: str, llm_model: str = 'claude') -> Optional[ScrapedArticle]:
        """
        Complete scraping process:
        1. Scrape URL with Crawl4AI
        2. Process with selected LLM (claude/gemini/huggingface)
        3. Return structured article data
        """
        logger.info(f"🚀 Starting complete scraping process for: {url} using LLM: {llm_model}")
        
        # Step 1: Scrape with Crawl4AI
        scraped_data = await self.scrape_with_crawl4ai(url)
        if not scraped_data:
            return None
        
        # Step 2: Process with the selected LLM
        article = None
        if llm_model == 'gemini':
            article = await self.process_with_gemini(scraped_data)
        elif llm_model == 'huggingface':
            article = await self.process_with_huggingface(scraped_data)
        elif llm_model == 'claude':
            article = await self.process_with_claude(scraped_data)
        else:
            logger.warning(f"⚠️ Unknown LLM model '{llm_model}'. Defaulting to HuggingFace.")
            article = await self.process_with_huggingface(scraped_data)

        if not article:
            return None
            
        logger.info(f"✅ ARTICLE COMPLETE: {article.title[:60]}... (Score: {article.significance_score})")
        return article
    
    async def scrape_multiple_sources(self, source_urls: List[str]) -> List[ScrapedArticle]:
        """
        Scrape multiple sources concurrently
        """
        logger.info(f"🔄 Scraping {len(source_urls)} sources with Claude processing...")
        
        tasks = [self.scrape_article(url, llm_model='claude') for url in source_urls] # Assuming default for this function
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for result in results:
            if isinstance(result, ScrapedArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Scraping error: {result}")
        
        logger.info(f"🎉 CLAUDE PROCESSING COMPLETE: {len(articles)} articles processed from {len(source_urls)} sources")
        return articles
    
    async def scrape_multiple_sources_with_publisher_id(self, article_data: List[Dict], llm_model: str = 'claude') -> List[ScrapedArticle]:
        """Scrape multiple sources with publisher_id and source_category mapping"""
        logger.info(f"🔄 Scraping {len(article_data)} sources with publisher_id and category mapping, using LLM: {llm_model}...")
        
        tasks = []
        for data in article_data:
            url = data['url']
            publisher_id = data['publisher_id']
            source_category = data.get('source_category', 'general')  # Get source category for fallback
            tasks.append(self.scrape_article_with_publisher_id(url, publisher_id, source_category, llm_model))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for result in results:
            if isinstance(result, ScrapedArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Scraping error: {result}")
        
        logger.info(f"🎉 LLM PROCESSING COMPLETE: {len(articles)} articles processed with publisher_id mapping using {llm_model}")
        return articles
    
    async def scrape_article_with_publisher_id(self, url: str, publisher_id: int, source_category: str = 'general', llm_model: str = 'claude') -> Optional[ScrapedArticle]:
        """Scrape individual article, include publisher_id, and process with the selected LLM."""
        logger.info(f"🕸️ Scraping article with publisher_id {publisher_id}, source_category '{source_category}': {url}")
        
        try:
            scraped_data = await self.scrape_with_crawl4ai(url)
            if scraped_data:
                # Add publisher_id and source_category to scraped_data before processing
                scraped_data['publisher_id'] = publisher_id
                scraped_data['source_category'] = source_category
                logger.info(f"📄 Added publisher_id {publisher_id} and source_category '{source_category}' to scraped data for {url}")
                
                if llm_model == 'gemini':
                    return await self.process_with_gemini(scraped_data)
                elif llm_model == 'huggingface':
                    return await self.process_with_huggingface(scraped_data)
                elif llm_model == 'claude':
                    return await self.process_with_claude(scraped_data)
                else:
                    logger.warning(f"⚠️ Unknown LLM model '{llm_model}'. Defaulting to HuggingFace for {url}.")
                    return await self.process_with_huggingface(scraped_data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to scrape article {url}: {str(e)}")
            return None

# Admin interface and example usage remain unchanged
class AdminScrapingInterface:
    """Admin interface for initiating scraping process as specified in requirements"""
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.scraper = Crawl4AIScraper()
        
    def map_content_type_to_id(self, content_type_str, url=None):
        """Map content type string to database ID (1=blogs, 2=videos, 3=podcasts) with enhanced video detection"""
        
        # First, check URL patterns for video platforms  
        if url:
            url_lower = url.lower()
            
            # Enhanced YouTube detection
            if 'youtube.com/watch' in url_lower or 'youtu.be/' in url_lower:
                logger.info(f"🎥 YouTube video detected: {url}")
                return 2  # Videos
            
            # Video platform URL detection
            video_platforms = [
                'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com', 
                'twitch.tv', 'facebook.com/watch', 'instagram.com/p', 
                'instagram.com/reel', 'tiktok.com', 'linkedin.com/posts',
                '/video/', '/watch/', '/tv/', 'video.', 'videos.',
                'stream.', 'live.', 'player.', 'embed/', 'iframe'
            ]
            
            if any(platform in url_lower for platform in video_platforms):
                logger.info(f"🎥 Video platform detected in URL: {url}")
                return 2  # Videos
            
            # Podcast platform URL detection  
            podcast_platforms = [
                'spotify.com/episode', 'soundcloud.com', 'anchor.fm',
                'podcasts.apple.com', 'podcasts.google.com', 'stitcher.com',
                'overcast.fm', 'pocketcasts.com', 'castbox.fm',
                '/podcast/', '/audio/', '/episode/', 'podcast.',
                '.mp3', '.wav', '.ogg', '.m4a'
            ]
            
            if any(platform in url_lower for platform in podcast_platforms):
                logger.info(f"🎧 Podcast platform detected in URL: {url}")
                return 3  # Podcasts
        
        # Then check content type string if provided
        if not content_type_str:
            return 1  # Default to blogs
        
        content_type_lower = content_type_str.lower().strip()
        
        # Enhanced mapping with multiple variations
        if any(term in content_type_lower for term in ['video', 'youtube', 'vimeo', 'mp4', 'visual', 'tv', 'stream', 'watch', 'film', 'movie', 'webinar']):
            logger.info(f"🎥 Video content type detected: {content_type_str}")
            return 2  # Videos  
        elif any(term in content_type_lower for term in ['podcast', 'audio', 'radio', 'soundcloud', 'spotify', 'mp3', 'sound', 'listen', 'episode']):
            logger.info(f"🎧 Podcast content type detected: {content_type_str}")
            return 3  # Podcasts
        elif any(term in content_type_lower for term in ['blog', 'article', 'post', 'text', 'news', 'story', 'report', 'read']):
            return 1  # Blogs
        else:
            # Check exact matches (case insensitive)
            exact_mapping = {
                'blogs': 1,
                'videos': 2, 
                'podcasts': 3,
                'articles': 1,
                'posts': 1,
                'news': 1
            }
            detected_type = exact_mapping.get(content_type_lower, 1)
            if detected_type != 1:
                logger.info(f"📝 Content type mapped: {content_type_str} → {detected_type}")
            return detected_type
    
    def _get_publisher_id_for_article(self, article_url, domain_to_publisher_mapping):
        """Get publisher_id for an article URL using domain mapping"""
        try:
            from urllib.parse import urlparse
            article_domain = urlparse(article_url).netloc.lower()
            
            # Try exact domain match first
            if article_domain in domain_to_publisher_mapping:
                return domain_to_publisher_mapping[article_domain]
            
            # Try without 'www.' prefix
            if article_domain.startswith('www.'):
                base_domain = article_domain[4:]
                if base_domain in domain_to_publisher_mapping:
                    return domain_to_publisher_mapping[base_domain]
            
            # Try with 'www.' prefix  
            www_domain = f"www.{article_domain}"
            if www_domain in domain_to_publisher_mapping:
                return domain_to_publisher_mapping[www_domain]
                
            logger.warning(f"⚠️ No domain mapping found for: {article_domain}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting publisher_id for {article_url}: {e}")
            return None
    
    def map_ai_topic_to_id(self, content_type_str, content_text=""):
        """Map content to AI topic ID based on content analysis"""
        # Simple keyword-based mapping - can be enhanced later
        content_lower = content_text.lower()
        
        if any(word in content_lower for word in ['robot', 'automation', 'robotic']):
            return 1  # Robotics & Automation
        elif any(word in content_lower for word in ['nlp', 'language', 'text', 'gpt', 'transformer']):
            return 2  # Natural Language Processing  
        elif any(word in content_lower for word in ['vision', 'image', 'computer vision', 'cv']):
            return 3  # Computer Vision
        elif any(word in content_lower for word in ['deep learning', 'neural network', 'cnn', 'rnn']):
            return 4  # Deep Learning
        elif any(word in content_lower for word in ['healthcare', 'medical', 'health', 'diagnosis']):
            return 5  # AI in Healthcare
        elif any(word in content_lower for word in ['finance', 'financial', 'trading', 'fintech']):
            return 6  # AI in Finance
        elif any(word in content_lower for word in ['startup', 'funding', 'investment', 'venture']):
            return 7  # AI Startups & Funding
        elif any(word in content_lower for word in ['policy', 'regulation', 'governance', 'ethics']):
            return 8  # AI Policy & Regulation
        elif any(word in content_lower for word in ['hardware', 'chip', 'gpu', 'computing']):
            return 9  # AI Hardware & Computing
        elif any(word in content_lower for word in ['tool', 'platform', 'framework', 'api']):
            return 10  # AI Tools & Platforms
        else:
            return 21  # Default: AI News & Updates
        
    async def initiate_scraping(self, admin_email: str = "admin@vidyagam.com", llm_model: str = 'claude') -> Dict[str, Any]:
        """
        Admin-initiated scraping process as specified:
        1. Select AI sources from ai_sources table
        2. Use Crawl4AI to scrape and extract content
        3. Process with selected LLM (Claude or Gemini)
        4. Store structured output in articles table
        5. Repeat for all sources
        """
        logger.info(f"🔧 Admin {admin_email} initiated scraping process with LLM: {llm_model}")
        logger.info("📰 RSS Feed Parsing Mode: Extracting individual article URLs from RSS feeds")
        
        try:
            # Step 1: Select AI sources from ai_sources table
            sources = self.db_service.get_ai_sources()
            if not sources:
                return {
                    "success": False,
                    "message": "No AI sources found in database",
                    "articles_processed": 0
                }
            
            logger.info(f"📡 Found {len(sources)} AI sources to scrape")
            
            # Step 2: Parse RSS feeds and extract article URLs with publisher_id mapping
            all_article_data = []  # Changed to store URL with publisher_id
            domain_to_publisher_mapping = {}  # Map domains to publisher_id for robust lookup
            rss_source_to_publisher_mapping = {}  # Map RSS sources to publisher_id
            
            for source in sources:
                rss_url = source.get('rss_url')
                website = source.get('website')
                source_name = source.get('name', 'Unknown')
                publisher_id = source.get('publisher_id')  # Get publisher_id from source
                source_category = source.get('category', 'general')  # Get category from source for fallback
                
                if rss_url:
                    # Parse RSS feed to get individual article URLs
                    logger.info(f"📰 RSS FEED MODE for {source_name}: {rss_url} (publisher_id: {publisher_id}, category: {source_category})")
                    article_urls = await self.scraper.parse_rss_feed(rss_url, max_articles=3)
                    logger.info(f"✅ Extracted {len(article_urls)} articles from RSS feed for {source_name}")
                    
                    # Map the RSS source domain to publisher_id
                    try:
                        from urllib.parse import urlparse
                        rss_domain = urlparse(rss_url).netloc.lower()
                        domain_to_publisher_mapping[rss_domain] = publisher_id
                        rss_source_to_publisher_mapping[rss_url] = publisher_id
                        logger.info(f"🗺️ Mapped RSS domain {rss_domain} → publisher_id: {publisher_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse RSS domain: {e}")
                    
                    # Associate each article URL with its publisher_id and category
                    for url in article_urls:
                        all_article_data.append({
                            'url': url, 
                            'publisher_id': publisher_id,
                            'source_category': source_category,  # RSS source category for fallback
                            'source_rss_url': rss_url,  # Keep track of source RSS
                            'source_name': source_name
                        })
                        
                        # Also map article domain to same publisher_id
                        try:
                            article_domain = urlparse(url).netloc.lower()
                            domain_to_publisher_mapping[article_domain] = publisher_id
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to parse article domain: {e}")
                            
                elif website:
                    # Fallback to website homepage if no RSS feed
                    logger.info(f"🌐 WEBSITE FALLBACK for {source_name}: {website} (publisher_id: {publisher_id}, category: {source_category})")
                    all_article_data.append({
                        'url': website, 
                        'publisher_id': publisher_id,
                        'source_category': source_category,  # RSS source category for fallback
                        'source_name': source_name
                    })
                    
                    # Map website domain to publisher_id
                    try:
                        website_domain = urlparse(website).netloc.lower()
                        domain_to_publisher_mapping[website_domain] = publisher_id
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse website domain: {e}")
                else:
                    logger.warning(f"⚠️ No RSS or website URL for {source_name}")
                
            logger.info(f"📡 Total article URLs collected: {len(all_article_data)}")
            logger.info(f"🗺️ Domain to publisher_id mapping created with {len(domain_to_publisher_mapping)} entries")
            logger.info(f"🔗 RSS source mappings: {len(rss_source_to_publisher_mapping)} RSS feeds mapped")
            
            # Step 3-4: Scrape individual articles with Crawl4AI + selected LLM, passing publisher_id
            logger.info(f"🤖 Starting LLM processing ({llm_model}) for {len(all_article_data)} articles...")
            articles = await self.scraper.scrape_multiple_sources_with_publisher_id(all_article_data, llm_model)
            
            # Step 5: Insert results into articles table
            articles_inserted = 0
            for article in articles:
                try:
                    # Map content type and AI topic dynamically with exception handling
                    content_type_id = 1  # Default to blogs
                    ai_topic_id = 21    # Default to AI News & Updates
                    
                    try:
                        # Enhanced content type detection with URL analysis
                        logger.debug(f"🔍 Content type detection for: {article.url}")
                        logger.debug(f"🔍 Original content_type_label: {article.content_type_label}")
                        content_type_id = self.map_content_type_to_id(article.content_type_label, article.url)
                        logger.info(f"📋 Content type assigned: {article.url} → content_type_id={content_type_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Content type mapping failed for {article.title[:50]}...: {str(e)}")
                        content_type_id = 1
                    
                    try:
                        ai_topic_id = self.map_ai_topic_to_id(article.content_type_label, article.summary + " " + article.content)
                    except Exception as e:
                        logger.warning(f"⚠️ AI topic mapping failed for {article.title[:50]}...: {str(e)}")
                        ai_topic_id = 21
                    
                    # Construct article data with exception handling for each field
                    article_data = {}
                    
                    try:
                        article_data['content_hash'] = hashlib.md5(article.url.encode()).hexdigest()
                    except Exception as e:
                        logger.warning(f"⚠️ Content hash generation failed for {article.url}: {str(e)}")
                        article_data['content_hash'] = hashlib.md5(f"fallback_{article.url}".encode()).hexdigest()
                    
                    # Add fields with safe fallbacks
                    article_data.update({
                        'title': str(article.title or "No title found")[:500],  # Limit length
                        'summary': str(article.summary or "")[:1000],
                        'content': str(article.content or "")[:10000],
                        'url': str(article.url or ""),
                        'source': str(article.source or "Unknown"),
                        'author': str(article.author or "") if article.author else None,
                        'significance_score': int(article.significance_score) if article.significance_score else 6,
                        'complexity_level': str(article.complexity_level or "Medium"),
                        'published_date': article.date,
                        'reading_time': int(article.reading_time) if article.reading_time else 1,
                        'content_type_label': str(article.content_type_label or "Blogs"),
                        'topic_category_label': str(article.topic_category_label or "AI News & Updates"),
                        'scraped_date': datetime.now(timezone.utc).isoformat(),
                        'created_date': datetime.now(timezone.utc).isoformat(),
                        'updated_date': datetime.now(timezone.utc).isoformat(),
                        'llm_processed': article.llm_processed or llm_model,
                        'publisher_id': self._get_publisher_id_for_article(article.url, domain_to_publisher_mapping),  # Direct mapping from ai_sources
                        'content_type_id': content_type_id,
                        'ai_topic_id': ai_topic_id
                    })
                    
                    # Debug logging for publisher_id mapping
                    mapped_publisher_id = self._get_publisher_id_for_article(article.url, domain_to_publisher_mapping)
                    if mapped_publisher_id:
                        logger.info(f"🔗 Domain mapping: {article.url} → publisher_id: {mapped_publisher_id}")
                    else:
                        logger.warning(f"⚠️ No publisher_id mapping found for URL: {article.url}")
                    
                    # Handle keywords safely
                    try:
                        if article.keywords and isinstance(article.keywords, list):
                            article_data['keywords'] = ', '.join(str(k) for k in article.keywords if k)
                        else:
                            article_data['keywords'] = None
                    except Exception as e:
                        logger.warning(f"⚠️ Keywords processing failed for {article.title[:50]}...: {str(e)}")
                        article_data['keywords'] = None
                    
                    logger.info(f"📋 MAPPING: {article.title[:50]}... → Type:{content_type_id} ({article.content_type_label}) Topic:{ai_topic_id}")
                    
                    # Insert into database with exception handling
                    try:
                        self.db_service.insert_article(article_data)
                        articles_inserted += 1
                    except Exception as db_e:
                        logger.error(f"❌ Database insertion failed for {article.title[:50]}...: {str(db_e)}")
                        continue
                    logger.info(f"💾 DATABASE INSERT: {article.title[:50]}... (#{articles_inserted})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to insert article {article.title}: {e}")
            
            logger.info(f"🎉 SCRAPING COMPLETE: {articles_inserted} articles processed by {llm_model} and stored in database")
            
            return {
                "success": True,
                "message": f"Scraping completed successfully",
                "sources_scraped": len(sources),
                "articles_found": len(articles),
                "articles_processed": articles_inserted,
                "initiated_by": admin_email,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Admin scraping process failed: {str(e)}")
            return {
                "success": False,
                "message": f"Scraping failed: {str(e)}",
                "articles_processed": 0
            }

# Example usage
async def main():
    """Example of how to use the scraper"""
    scraper = Crawl4AIScraper()
    
    # Test scraping a single article
    test_url = "https://openai.com/blog"
    article = await scraper.scrape_article(test_url)
    
    if article:
        print(f"✅ Scraped: {article.title}")
        print(f"📝 Summary: {article.summary}")
        print(f"⭐ Score: {article.significance_score}")
    else:
        print("❌ Scraping failed")

# Cleanup function for module-level temp directory
def cleanup_crawl4ai_temp():
    """Cleanup the module-level temp directory"""
    try:
        import shutil
        global temp_base_dir
        if os.path.exists(temp_base_dir):
            shutil.rmtree(temp_base_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up Crawl4AI temp directory: {temp_base_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")

# Register cleanup on module exit
import atexit
atexit.register(cleanup_crawl4ai_temp)

if __name__ == "__main__":
    asyncio.run(main())