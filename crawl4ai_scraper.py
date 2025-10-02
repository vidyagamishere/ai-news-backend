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
    headline: str
    author: Optional[str]
    summary: str
    content: str
    date: Optional[str]
    url: str
    source: str
    content_type: str = "article"
    significance_score: float = 5.0
    reading_time: int = 1

class Crawl4AIScraper:
    """AI News Scraper using Crawl4AI and Claude LLM"""
    
    def __init__(self):
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        if not self.anthropic_api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not set - scraper will not work properly")
            
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
            async with AsyncWebCrawler(
                crawler_strategy=crawler_strategy,
                always_by_pass_cache=True,
                verbose=False,  # Reduce logging for Railway
                cache_mode="disabled",  # Disable caching to avoid permission issues
                timeout=30  # Shorter timeout for Railway
            ) as crawler:
                
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
                
                # Extract title from multiple sources
                title = (
                    extracted_data.get("title") or 
                    result.metadata.get("title") or
                    "No title found"
                )
                if isinstance(title, list):
                    title = title[0] if title else "No title found"
                
                # Extract description
                description = (
                    extracted_data.get("description") or
                    result.metadata.get("description") or
                    ""
                )
                if isinstance(description, list):
                    description = description[0] if description else ""
                
                # Extract author
                author = extracted_data.get("author")
                if isinstance(author, list):
                    author = author[0] if author else None
                
                # Extract publication date
                pub_date = extracted_data.get("date")
                if isinstance(pub_date, list):
                    pub_date = pub_date[0] if pub_date else None
                
                # Extract main content (prefer structured extraction, fallback to markdown)
                content = ""
                if extracted_data.get("content"):
                    content = extracted_data["content"]
                    if isinstance(content, list):
                        content = " ".join(content)
                else:
                    content = markdown_content
                
                  # --- NEW STEP 1: Full Cleaning ---
                # Clean the entire extracted content before calculating word count.
                full_clean_content = self._clean_content(content)

                # --- NEW STEP 2: Calculate Reading Time (on FULL content) ---
                # A standard average reading speed is 225 words per minute.
                # Use a regex to count words robustly.
                word_count = len(re.findall(r'\b\w+\b', full_clean_content))
                
                # Calculate reading time in minutes, rounded up to the nearest whole minute
                # Adding 0.5 before converting to int simulates ceiling
                reading_time_minutes = int((word_count / 225) + 0.5) if word_count > 0 else 1
                
                # Ensure a minimum reading time of 1 minute
                if reading_time_minutes == 0:
                    reading_time_minutes = 1
                
                # --- NEW STEP 3: Limit Content (for LLM prompt) ---
                # Now, truncate the content to 4000 characters for the LLM processing step.
                content = full_clean_content[:4000] 
 
                # Extract tags/topics
                tags = extracted_data.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                
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
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "extraction_method": "crawl4ai_full",
                    "links": result.links[:10] if result.links else [],  # Extract some links
                    "media": result.media[:5] if result.media else []  # Extract some media
                }
                
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
    
    async def _fallback_scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Enhanced fallback scraping for Railway deployment"""
        try:
            # logger.info(f"🔄 Using enhanced fallback scraping for: {url}")
            
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Referer': 'https://google.com/',
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=30, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.error(f"❌ Failed to fetch {url}: HTTP {response.status}")
                        return None
                    
                    html_content = await response.text()
                    
                    # Extract basic content using BeautifulSoup
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript", "form"]):
                        element.decompose()
                    
                    # Extract title with multiple fallbacks
                    title = self._extract_title(soup)
                    
                    # Extract meta description
                    description = self._extract_description(soup)
                    
                    # Extract author with multiple methods
                    author = self._extract_author(soup)
                    
                    # Extract publication date
                    pub_date = self._extract_date(soup)
                    
                     # --- ADDED: Reading Time Calculation ---
                     # Extract main content with priority selectors
                    content = self._extract_content(soup)
                    # 1. Clean the full content
                    full_clean_content = self._clean_content(content)
                    
                    # 2. Calculate word count and reading time (225 WPM)
                    word_count = len(re.findall(r'\b\w+\b', full_clean_content))
                    reading_time_minutes = int((word_count / 225) + 0.5) if word_count > 0 else 1
                    
                    if reading_time_minutes == 0:
                        reading_time_minutes = 1
                    # --------------------------------------

                    
                    # Extract tags/keywords
                    tags = self._extract_tags(soup)
                    
                    return {
                        "title": title.strip(),
                        "description": description.strip(),
                        "content": full_clean_content.strip()[:4000],
                        "author": author,
                        "date": pub_date,
                        "tags": tags,
                        "url": url,
                        "reading_time": reading_time_minutes,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                        "extraction_method": "enhanced_fallback"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Enhanced fallback scraping failed for {url}: {str(e)}")
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
        """Extract tags and keywords"""
        tags = []
        
        # Try meta keywords
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            tags.extend([tag.strip() for tag in keywords_meta['content'].split(',')])
        
        # Try tag/category elements
        for selector in ['.tags', '.categories', '.keywords', '.topics', '.tag']:
            elements = soup.select(f'{selector} a, {selector} span')
            for elem in elements:
                tag_text = elem.get_text().strip()
                if tag_text and len(tag_text) < 50:
                    tags.append(tag_text)
        
        return list(set(tags))[:10]  # Remove duplicates and limit
    
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
            
            prompt = f"""You are an expert AI news analyst and content classifier. Analyze the following scraped content and extract key information in JSON format.

Content Title: {scraped_data.get('title', '')}
Content Description: {scraped_data.get('description', '')}
Author: {author_info}
Existing Tags: {tags_info}
Content Text: {scraped_data.get('content', '')[:4000]}
Source URL: {scraped_data.get('url', '')}
Extraction Method: {extraction_method}

Please analyze this content and return ONLY a valid JSON object with the following structure:
{{
    "headline": "Clear, concise headline for the article (use original title if good)",
    "author": "Author name if found, or null",
    "summary": "4-5 sentence summary of the key points and significance",
    "date": "Publication date if found, or null",
    "content_type_label": "Classify the content into one of the content types: Blog Posts & Articles,Videos,Podcasts & Audio, Research Papers, Events & Conferences, Demos & Tools, Newsletters & Email Updates",
    "significance_score": "Number from 1-10 indicating importance of this AI/tech news",
    "key_topics": ["list of key AI tech topics covered"],
    "topic_category_label": "Classify the core subject matter into ONLY ONE of the following 22 categories: AI Tools & Platforms, AI Startups & Funding, Robotics & Automation, AI Research Papers, AI Policy & Regulation, Natural Language Processing, AI News & Updates, Machine Learning, AI Learning & Education, AI International, AI in Healthcare, AI Hardware & Computing, AI in Gaming, AI in Finance, AI Events & Conferences, AI Ethics & Safety, Deep Learning, AI in Creative Arts, Computer Vision, AI Cloud Services, AI in Automotive, AI in Business"
}}

Focus on AI, machine learning, technology, and innovation content. Be accurate and provide meaningful analysis.
If this is not AI/tech related content, set significance_score to 1-3."""

            # Call Claude API
            payload = {
                "model": "claude-3-5-sonnet-20240620",
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
                logger.warning("⚠️ Using fallback processing (no Anthropic API key)")
                return ScrapedArticle(
                    headline=scraped_data.get('title', 'AI News Article'),
                    author=scraped_data.get('author'),
                    summary=scraped_data.get('description', 'AI news and developments'),
                    content=scraped_data.get('content', '')[:1000],
                    date=scraped_data.get('date') or scraped_data.get('extracted_at'),
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=6.0,
                    content_type="Articles & Blog Posts",
                    reading_time=scraped_data.get('reading_time', 1)
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
                        logger.info(f"✅ CLAUDE SUCCESS: Processed {parsed_data.get('headline', 'Unknown')[:50]}... (Score: {parsed_data.get('significance_score', 'N/A')})")
                        
                        return ScrapedArticle(
                            headline=parsed_data.get('headline', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', '')[:1000],
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0)),
                            content_type=parsed_data.get('content_type_label', 'Blog Posts & Articles'),
                            reading_time=scraped_data.get('reading_time', 1)
                        )
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ CLAUDE JSON ERROR: Failed to parse response: {e}")
                        logger.error(f"Claude raw response: {content[:200]}...")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ CLAUDE PROCESSING FAILED: {str(e)}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "Unknown Source"
    
    async def parse_rss_feed(self, rss_url: str, max_articles: int = 5) -> List[str]:
        """Parse RSS feed and extract article URLs"""
        try:
            logger.info(f"📰 Parsing RSS feed: {rss_url}")
            
            # Parse the RSS feed
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                logger.warning(f"⚠️ No entries found in RSS feed: {rss_url}")
                return []
            
            # Extract article URLs from feed entries
            article_urls = []
            for entry in feed.entries[:max_articles]:  # Limit number of articles per feed
                if hasattr(entry, 'link') and entry.link:
                    article_urls.append(entry.link)
                    logger.debug(f"📄 Found article: {entry.get('title', 'No title')} - {entry.link}")
            
            logger.info(f"✅ Extracted {len(article_urls)} article URLs from RSS feed: {rss_url}")
            return article_urls
            
        except Exception as e:
            logger.error(f"❌ Failed to parse RSS feed {rss_url}: {str(e)}")
            return []
    
    async def scrape_article(self, url: str) -> Optional[ScrapedArticle]:
        """
        Complete scraping process:
        1. Scrape URL with Crawl4AI
        2. Process with Claude
        3. Return structured article data
        """
        # logger.info(f"🚀 Starting complete scraping process for: {url}")
        
        # Step 1: Scrape with Crawl4AI
        scraped_data = await self.scrape_with_crawl4ai(url)
        if not scraped_data:
            return None
        
        # Step 2: Process with Claude
        article = await self.process_with_claude(scraped_data)
        if not article:
            return None
            
        logger.info(f"✅ ARTICLE COMPLETE: {article.headline[:60]}... (Score: {article.significance_score})")
        return article
    
    async def scrape_multiple_sources(self, source_urls: List[str]) -> List[ScrapedArticle]:
        """
        Scrape multiple sources concurrently
        """
        logger.info(f"🔄 Scraping {len(source_urls)} sources with Claude processing...")
        
        tasks = [self.scrape_article(url) for url in source_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for result in results:
            if isinstance(result, ScrapedArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Scraping error: {result}")
        
        logger.info(f"🎉 CLAUDE PROCESSING COMPLETE: {len(articles)} articles processed from {len(source_urls)} sources")
        return articles

class AdminScrapingInterface:
    """Admin interface for initiating scraping process as specified in requirements"""
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.scraper = Crawl4AIScraper()
        
    async def initiate_scraping(self, admin_email: str = "admin@vidyagam.com") -> Dict[str, Any]:
        """
        Admin-initiated scraping process as specified:
        1. Select AI sources from ai_sources table
        2. Use Crawl4AI to scrape and extract content
        3. Process with Claude LLM
        4. Store structured output in articles table
        5. Repeat for all sources
        """
        logger.info(f"🔧 Admin {admin_email} initiated scraping process with RSS feed parsing")
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
            
            # Step 2: Parse RSS feeds and extract article URLs
            all_article_urls = []
            
            for source in sources:
                rss_url = source.get('rss_url')
                website = source.get('website')
                source_name = source.get('name', 'Unknown')
                
                if rss_url:
                    # Parse RSS feed to get individual article URLs
                    logger.info(f"📰 RSS FEED MODE for {source_name}: {rss_url}")
                    article_urls = await self.scraper.parse_rss_feed(rss_url, max_articles=3)
                    logger.info(f"✅ Extracted {len(article_urls)} articles from RSS feed for {source_name}")
                    all_article_urls.extend(article_urls)
                elif website:
                    # Fallback to website homepage if no RSS feed
                    logger.info(f"🌐 WEBSITE FALLBACK for {source_name}: {website}")
                    all_article_urls.append(website)
                else:
                    logger.warning(f"⚠️ No RSS or website URL for {source_name}")
                
            logger.info(f"📡 Total article URLs collected: {len(all_article_urls)}")
            
            # Step 3-4: Scrape individual articles with Crawl4AI + Claude
            logger.info(f"🤖 Starting Claude AI processing for {len(all_article_urls)} articles...")
            articles = await self.scraper.scrape_multiple_sources(all_article_urls)
            
            # Step 5: Insert results into articles table
            articles_inserted = 0
            for article in articles:
                try:
                    article_data = {
                        'content_hash': hashlib.md5(article.url.encode()).hexdigest(),
                        'title': article.headline,
                        'description': article.summary,
                        'content': article.content,
                        'url': article.url,
                        'source': article.source,
                        'content_type': article.content_type,
                        'significance_score': article.significance_score or 6,
                        'published_date': article.date,
                        'reading_time': article.reading_time,
                        'content_type_id': 1,  # Default to blogs/articles
                        'ai_topic_id': 21,     # Default AI topic ID
                        'scraped_date': datetime.now(timezone.utc).isoformat(),
                        'created_date': datetime.now(timezone.utc).isoformat(),
                        'updated_date': datetime.now(timezone.utc).isoformat(),
                        'llm_processed': True
                    }
                    
                    # Insert into database
                    self.db_service.insert_article(article_data)
                    articles_inserted += 1
                    logger.info(f"💾 DATABASE INSERT: {article.headline[:50]}... (#{articles_inserted})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to insert article {article.headline}: {e}")
            
            logger.info(f"🎉 SCRAPING COMPLETE: {articles_inserted} articles processed by Claude and stored in database")
            
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
        print(f"✅ Scraped: {article.headline}")
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