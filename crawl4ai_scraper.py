#!/usr/bin/env python3
"""
AI News Scraper using Crawl4AI and Mistral-Small-3 as specified in functional requirements.
Implements the exact scraping process: Crawl4AI -> Mistral-Small-3 -> Structured Output
"""

import os
import tempfile

# Set up writable directories for Railway deployment BEFORE any Crawl4AI imports
temp_base_dir = tempfile.mkdtemp(prefix='crawl4ai_')
os.environ['CRAWL4AI_CACHE_DIR'] = temp_base_dir
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = os.path.join(temp_base_dir, 'browsers')
os.environ['CRAWL4AI_BASE_DIRECTORY'] = temp_base_dir

import json
import logging
import asyncio
import hashlib
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp

# Configure logging
logger = logging.getLogger(__name__)

# Crawl4AI imports - after environment setup
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, CosineStrategy
    from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
    CRAWL4AI_AVAILABLE = True
    logger.info(f"✅ Crawl4AI available - using cache dir: {temp_base_dir}")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    logger.warning(f"⚠️ Crawl4AI not installed - falling back to basic scraping: {e}")

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

class Crawl4AIScraper:
    """AI News Scraper using Crawl4AI and Mistral-Small-3 LLM"""
    
    def __init__(self):
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY', '')
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        if not self.mistral_api_key:
            logger.warning("⚠️ MISTRAL_API_KEY not set - scraper will not work properly")
            
    async def scrape_with_crawl4ai(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Use Crawl4AI to scrape article URLs and extract core content,
        automatically converting it into clean Markdown or JSON.
        """
        try:
            logger.info(f"🕷️ Scraping URL with Crawl4AI: {url}")
            
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
                
                # Clean and limit content for LLM processing
                content = self._clean_content(content)[:8000]
                
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
        """Fallback basic scraping when Crawl4AI is not available"""
        try:
            logger.info(f"🔄 Using fallback scraping for: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"❌ Failed to fetch {url}: HTTP {response.status}")
                        return None
                    
                    html_content = await response.text()
                    
                    # Extract basic content using BeautifulSoup
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "aside"]):
                        script.decompose()
                    
                    # Extract title
                    title_tag = soup.find('title')
                    title = title_tag.get_text() if title_tag else "No title found"
                    
                    # Try to find better title in h1 tags
                    h1_tag = soup.find('h1')
                    if h1_tag and len(h1_tag.get_text().strip()) > 10:
                        title = h1_tag.get_text().strip()
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc.get('content') if meta_desc else ""
                    
                    # Extract author from meta tags or common selectors
                    author = None
                    author_meta = soup.find('meta', attrs={'name': 'author'})
                    if author_meta:
                        author = author_meta.get('content')
                    else:
                        author_elem = soup.find(class_=lambda x: x and 'author' in x.lower())
                        if author_elem:
                            author = author_elem.get_text().strip()
                    
                    # Extract main content
                    content = ""
                    
                    # Try to find article content using common selectors
                    content_selectors = [
                        'article', '.content', '.post-content', '.entry-content', 
                        '.article-body', 'main', '.main-content', '.story-body'
                    ]
                    
                    for selector in content_selectors:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            content = content_elem.get_text()
                            break
                    
                    # If no specific content found, extract all text
                    if not content:
                        content = soup.get_text()
                    
                    # Clean up content
                    content = self._clean_content(content)[:5000]
                    
                    return {
                        "title": title.strip(),
                        "description": description.strip(),
                        "content": content.strip(),
                        "author": author,
                        "date": None,
                        "tags": [],
                        "url": url,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                        "extraction_method": "fallback_basic"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Fallback scraping failed for {url}: {str(e)}")
            return None
    
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

    async def process_with_mistral(self, scraped_data: Dict[str, Any]) -> Optional[ScrapedArticle]:
        """
        Feed the structured data from Crawl4AI to Mistral-Small-3 for processing.
        Use specific prompt to get structured output with key details.
        """
        try:
            logger.info(f"🧠 Processing with Mistral-Small-3: {scraped_data.get('title', 'Unknown')}")
            
            # Create specific prompt for structured output with enhanced data
            extraction_method = scraped_data.get('extraction_method', 'unknown')
            author_info = scraped_data.get('author', 'Not specified')
            tags_info = scraped_data.get('tags', [])
            
            prompt = f"""
You are an AI news analyst. Analyze the following scraped content and extract key information in JSON format.

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
    "summary": "2-3 sentence summary of the key points and significance",
    "date": "Publication date if found, or null",
    "content_type": "article",
    "significance_score": "Number from 1-10 indicating importance of this AI/tech news",
    "key_topics": ["list", "of", "key", "AI", "tech", "topics", "covered"],
    "category": "One of: AI/ML, Robotics, Startups, Research, Industry, Policy, Hardware, Software"
}}

Focus on AI, machine learning, technology, and innovation content. Be accurate and provide meaningful analysis.
If this is not AI/tech related content, set significance_score to 1-3.
"""

            # Call Mistral-Small-3 API
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            if not self.mistral_api_key:
                # Enhanced fallback for demo purposes
                logger.warning("⚠️ Using fallback processing (no Mistral API key)")
                return ScrapedArticle(
                    headline=scraped_data.get('title', 'AI News Article'),
                    author=scraped_data.get('author'),
                    summary=scraped_data.get('description', 'AI news and developments'),
                    content=scraped_data.get('content', '')[:1000],
                    date=scraped_data.get('date') or scraped_data.get('extracted_at'),
                    url=scraped_data.get('url', ''),
                    source=self._extract_domain(scraped_data.get('url', '')),
                    significance_score=6.0
                )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.mistral_api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status != 200:
                        logger.error(f"❌ Mistral API error: HTTP {response.status}")
                        return None
                    
                    result = await response.json()
                    
                    # Extract and parse the LLM response
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                    
                    try:
                        # Parse JSON response from LLM
                        parsed_data = json.loads(content)
                        
                        return ScrapedArticle(
                            headline=parsed_data.get('headline', scraped_data.get('title', 'Unknown')),
                            author=parsed_data.get('author') or scraped_data.get('author'),
                            summary=parsed_data.get('summary', 'No summary available'),
                            content=scraped_data.get('content', ''),
                            date=parsed_data.get('date') or scraped_data.get('date'),
                            url=scraped_data.get('url', ''),
                            source=self._extract_domain(scraped_data.get('url', '')),
                            significance_score=float(parsed_data.get('significance_score', 5.0))
                        )
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse Mistral response as JSON: {e}")
                        logger.error(f"Raw response: {content}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Mistral processing failed: {str(e)}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "Unknown Source"
    
    async def scrape_article(self, url: str) -> Optional[ScrapedArticle]:
        """
        Complete scraping process:
        1. Scrape URL with Crawl4AI
        2. Process with Mistral-Small-3
        3. Return structured article data
        """
        logger.info(f"🚀 Starting complete scraping process for: {url}")
        
        # Step 1: Scrape with Crawl4AI
        scraped_data = await self.scrape_with_crawl4ai(url)
        if not scraped_data:
            return None
        
        # Step 2: Process with Mistral-Small-3
        article = await self.process_with_mistral(scraped_data)
        if not article:
            return None
            
        logger.info(f"✅ Successfully processed article: {article.headline}")
        return article
    
    async def scrape_multiple_sources(self, source_urls: List[str]) -> List[ScrapedArticle]:
        """
        Scrape multiple sources concurrently
        """
        logger.info(f"🔄 Scraping {len(source_urls)} sources...")
        
        tasks = [self.scrape_article(url) for url in source_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for result in results:
            if isinstance(result, ScrapedArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Scraping error: {result}")
        
        logger.info(f"✅ Successfully scraped {len(articles)} articles from {len(source_urls)} sources")
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
        3. Process with Mistral-Small-3 LLM
        4. Store structured output in articles table
        5. Repeat for all sources
        """
        logger.info(f"🔧 Admin {admin_email} initiated scraping process")
        
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
            
            # Step 2-4: Scrape all sources with Crawl4AI + Mistral-Small-3
            source_urls = [source.get('rss_url') or source.get('website') for source in sources if source.get('rss_url') or source.get('website')]
            articles = await self.scraper.scrape_multiple_sources(source_urls)
            
            # Step 5: Insert results into articles table
            articles_inserted = 0
            for article in articles:
                try:
                    article_data = {
                        'id': hashlib.md5(article.url.encode()).hexdigest(),
                        'title': article.headline,
                        'description': article.summary,
                        'content': article.summary,
                        'url': article.url,
                        'source': article.source,
                        'significance_score': article.significance_score,
                        'published_date': article.date,
                        'scraped_date': datetime.now(timezone.utc).isoformat(),
                        'llm_processed': True
                    }
                    
                    # Insert into database
                    self.db_service.insert_article(article_data)
                    articles_inserted += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to insert article {article.headline}: {e}")
            
            logger.info(f"✅ Scraping completed: {articles_inserted} articles processed and stored")
            
            return {
                "success": True,
                "message": f"Scraping completed successfully",
                "sources_scraped": len(source_urls),
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