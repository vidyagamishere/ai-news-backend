# multimedia_scraper.py
import os
import sqlite3
import json
import requests
import hashlib
import time
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import feedparser

logger = logging.getLogger(__name__)

# Multimedia Sources Configuration
MULTIMEDIA_SOURCES = {
    "audio": [
        {
            "name": "Lex Fridman Podcast",
            "type": "podcast_rss",
            "url": "https://lexfridman.com/feed/podcast/",
            "website": "https://lexfridman.com/podcast/",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "AI Podcast by NVIDIA",
            "type": "podcast_rss",
            "url": "https://feeds.soundcloud.com/users/soundcloud:users:264034133/sounds.rss",
            "website": "https://blogs.nvidia.com/ai-podcast/",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "Machine Learning Street Talk",
            "type": "podcast_rss",
            "url": "https://anchor.fm/s/3e1b5938/podcast/rss",
            "website": "https://anchor.fm/machinelearningstreettalk",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "Latent Space Podcast",
            "type": "podcast_rss",
            "url": "https://feeds.transistor.fm/latent-space",
            "website": "https://latent.space/podcast",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "No Priors AI",
            "type": "podcast_rss",
            "url": "https://feeds.megaphone.fm/no-priors",
            "website": "https://nopriors.ai",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "Practical AI",
            "type": "podcast_rss",
            "url": "https://changelog.com/practicalai/feed",
            "website": "https://changelog.com/practicalai",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "TWIML AI Podcast",
            "type": "podcast_rss",
            "url": "https://twimlai.com/feed/podcast/",
            "website": "https://twimlai.com/podcast",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "The AI Breakdown",
            "type": "podcast_rss",
            "url": "https://feeds.buzzsprout.com/2034686.rss",
            "website": "https://www.theaibreakdown.com/",
            "priority": 3,
            "enabled": True
        },
        {
            "name": "Eye on AI",
            "type": "podcast_rss",
            "url": "https://feeds.megaphone.fm/eyeonai",
            "website": "https://www.eyeonai.io/",
            "priority": 3,
            "enabled": True
        },
        {
            "name": "AI Today Podcast",
            "type": "podcast_rss",
            "url": "https://feeds.feedburner.com/aitodaypodcast",
            "website": "https://www.aitoday.com/",
            "priority": 3,
            "enabled": True
        }
    ],
    "video": [
        {
            "name": "Two Minute Papers",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg",
            "website": "https://www.youtube.com/@TwoMinutePapers",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "DeepLearning.AI",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCcIXc5mJsHVYTZR1maL5l9w",
            "website": "https://www.youtube.com/@DeepLearningAI",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "Yannic Kilcher",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCZHmQk67mSJgfCCTn7xBfew",
            "website": "https://www.youtube.com/@YannicKilcher",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "3Blue1Brown",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw",
            "website": "https://www.youtube.com/@3blue1brown",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "Lex Fridman",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCSHZKyawb77ixDdsGog4iWA",
            "website": "https://www.youtube.com/@lexfridman",
            "priority": 1,
            "enabled": True
        },
        {
            "name": "AI Explained",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw",
            "website": "https://www.youtube.com/@aiexplained-official",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "Machine Learning with Phil",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC58v9cLitc8VaCjrcKyAbrw",
            "website": "https://www.youtube.com/@MachineLearningwithPhil",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "Sentdex",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCfzlCWGWYyIQ0aLC5w48gBQ",
            "website": "https://www.youtube.com/@sentdex",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "CodeEmporium",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC5_6ZD6s8klmMu9TXEB_1IA",
            "website": "https://www.youtube.com/@CodeEmporium",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "StatQuest with Josh Starmer",
            "type": "youtube_rss",
            "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCtYLUTtgS3k1Fg4y5tAhLbw",
            "website": "https://www.youtube.com/@statquest",
            "channel_url": "@statquest",
            "priority": 2,
            "enabled": True
        },
        {
            "name": "Sentdex",
            "type": "youtube_channel",
            "channel_id": "UCfzlCWGWYyIQ0aLC5w48gBQ",
            "channel_url": "@sentdex",
            "priority": 3,
            "enabled": True
        },
        {
            "name": "Krish Naik",
            "type": "youtube_channel",
            "channel_id": "UCNU_lfiiWBdtULKOw6X0Dig",
            "channel_url": "@krishnaik06",
            "priority": 3,
            "enabled": True
        }
    ]
}

class MultimediaDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_multimedia_tables()
    
    def init_multimedia_tables(self):
        """Initialize multimedia tables in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Audio content table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    summary TEXT,
                    url TEXT UNIQUE,
                    audio_url TEXT,
                    duration INTEGER,
                    url_hash TEXT UNIQUE,
                    published_date TEXT,
                    significance_score REAL DEFAULT 0.0,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Video content table  
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    summary TEXT,
                    url TEXT UNIQUE,
                    video_url TEXT,
                    duration INTEGER,
                    thumbnail_url TEXT,
                    view_count INTEGER DEFAULT 0,
                    url_hash TEXT UNIQUE,
                    published_date TEXT,
                    significance_score REAL DEFAULT 0.0,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_source ON audio_content(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_score ON audio_content(significance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_created ON audio_content(created_at)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_source ON video_content(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_score ON video_content(significance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_created ON video_content(created_at)')
            
            conn.commit()
            conn.close()
            logger.info("Multimedia database tables initialized successfully")
        except Exception as e:
            logger.error(f"Multimedia database initialization error: {e}")
            raise
    
    def save_audio_content(self, content: Dict) -> bool:
        """Save audio content with deduplication"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            url_hash = hashlib.md5(content['url'].encode()).hexdigest()
            
            published_date = content.get('published_date')
            if isinstance(published_date, datetime):
                published_date = published_date.isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO audio_content 
                (source, title, description, url, audio_url, duration, url_hash, published_date, 
                 significance_score, processed, summary, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content['source'],
                content['title'], 
                content.get('description', ''),
                content['url'],
                content.get('audio_url', ''),
                content.get('duration', 0),
                url_hash,
                published_date,
                content.get('significance_score', 0.0),
                content.get('processed', False),
                content.get('summary', ''),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving audio content: {e}")
            return False
    
    def save_video_content(self, content: Dict) -> bool:
        """Save video content with deduplication"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            url_hash = hashlib.md5(content['url'].encode()).hexdigest()
            
            published_date = content.get('published_date')
            if isinstance(published_date, datetime):
                published_date = published_date.isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO video_content 
                (source, title, description, url, video_url, duration, thumbnail_url, view_count,
                 url_hash, published_date, significance_score, processed, summary, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content['source'],
                content['title'], 
                content.get('description', ''),
                content['url'],
                content.get('video_url', ''),
                content.get('duration', 0),
                content.get('thumbnail_url', ''),
                content.get('view_count', 0),
                url_hash,
                published_date,
                content.get('significance_score', 0.0),
                content.get('processed', False),
                content.get('summary', ''),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving video content: {e}")
            return False
    
    def get_recent_audio_content(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent audio content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM audio_content 
                WHERE created_at > ? 
                ORDER BY significance_score DESC, published_date DESC
                LIMIT ?
            ''', (since_date, limit))
            
            columns = [desc[0] for desc in cursor.description]
            content = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return content
        except Exception as e:
            logger.error(f"Error getting recent audio content: {e}")
            return []
    
    def get_recent_video_content(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent video content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM video_content 
                WHERE created_at > ? 
                ORDER BY significance_score DESC, published_date DESC
                LIMIT ?
            ''', (since_date, limit))
            
            columns = [desc[0] for desc in cursor.description]
            content = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return content
        except Exception as e:
            logger.error(f"Error getting recent video content: {e}")
            return []

class AudioContentScraper:
    def __init__(self, cache_manager, rate_limiter):
        self.cache = cache_manager
        self.rate_limiter = rate_limiter
    
    async def fetch_podcast_feed(self, source: Dict, max_episodes: int = 10) -> List[Dict]:
        """Fetch podcast episodes from RSS feed"""
        cache_key = f"podcast_feed_{source['name']}"
        
        cached_data = self.cache.get(cache_key)
        if cached_data:
            try:
                logger.info(f"Using cached podcast data for {source['name']}")
                return json.loads(cached_data)
            except json.JSONDecodeError:
                pass
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        try:
            logger.info(f"Fetching podcast feed for {source['name']}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(source['url'], headers=headers, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            
            episodes = []
            entries = getattr(feed, 'entries', [])
            
            for entry in entries[:max_episodes]:
                try:
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published_date = datetime(*entry.published_parsed[:6])
                        except (TypeError, ValueError):
                            pass
                    
                    # Extract audio URL from enclosures
                    audio_url = ""
                    duration = 0
                    if hasattr(entry, 'enclosures') and entry.enclosures:
                        for enclosure in entry.enclosures:
                            if 'audio' in enclosure.get('type', ''):
                                audio_url = enclosure.get('href', '')
                                break
                    
                    # Extract duration if available
                    if hasattr(entry, 'itunes_duration'):
                        duration_str = entry.itunes_duration
                        try:
                            if ':' in duration_str:
                                parts = duration_str.split(':')
                                if len(parts) == 2:
                                    duration = int(parts[0]) * 60 + int(parts[1])
                                elif len(parts) == 3:
                                    duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            else:
                                duration = int(duration_str)
                        except ValueError:
                            pass
                    
                    description = ""
                    if hasattr(entry, 'summary'):
                        description = entry.summary
                        soup = BeautifulSoup(description, 'html.parser')
                        description = soup.get_text().strip()
                    
                    if not description or len(description) < 50:
                        continue
                    
                    episode = {
                        'source': source['name'],
                        'title': getattr(entry, 'title', 'No title'),
                        'description': description[:2000],
                        'url': getattr(entry, 'link', ''),
                        'audio_url': audio_url,
                        'duration': duration,
                        'published_date': published_date,
                        'processed': False,
                        'priority': source.get('priority', 3)
                    }
                    
                    episodes.append(episode)
                    
                except Exception as e:
                    logger.warning(f"Error processing podcast episode: {e}")
                    continue
            
            self.cache.set(cache_key, json.dumps(episodes, default=str), 3600)
            logger.info(f"Fetched {len(episodes)} episodes from {source['name']}")
            return episodes
            
        except Exception as e:
            logger.error(f"Error fetching podcast for {source['name']}: {e}")
            return []
    
    def _extract_audio_url(self, entry) -> str:
        """Extract direct audio URL from RSS entry"""
        # Priority 1: Check for enclosure (most common for podcasts)
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enclosure in entry.enclosures:
                enclosure_type = enclosure.get('type', '').lower()
                if any(audio_type in enclosure_type for audio_type in ['audio', 'mp3', 'wav', 'ogg', 'm4a']):
                    return enclosure.get('href', '')
        
        # Priority 2: Check for links with audio file extensions
        if hasattr(entry, 'links'):
            for link in entry.links:
                href = link.get('href', '')
                if any(ext in href.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac']):
                    return href
        
        # Priority 3: Check for media content
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                media_type = media.get('type', '').lower()
                if any(audio_type in media_type for audio_type in ['audio', 'mp3', 'wav', 'ogg', 'm4a']):
                    return media.get('url', '')
        
        # Fallback: return the entry link
        return entry.get('link', '')
    
    def _extract_duration(self, entry) -> str:
        """Extract episode duration from RSS entry"""
        # Check iTunes duration tag
        if hasattr(entry, 'itunes_duration'):
            return entry.itunes_duration
        
        # Check for duration in description
        description = entry.get('summary', '') or entry.get('description', '')
        import re
        duration_pattern = r'(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})'
        match = re.search(duration_pattern, description)
        if match:
            return match.group(1)
        
        return "Unknown"
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML tags from text"""
        if not text:
            return ""
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text().strip()
        except:
            # Fallback: simple regex HTML removal
            import re
            clean = re.sub(r'<[^>]+>', '', text)
            return clean.strip()
    
    def _parse_date(self, date_string: str):
        """Parse date string to datetime object"""
        if not date_string:
            return datetime.now()
        
        try:
            import email.utils
            parsed_time = email.utils.parsedate_tz(date_string)
            if parsed_time:
                timestamp = email.utils.mktime_tz(parsed_time)
                return datetime.fromtimestamp(timestamp)
        except:
            pass
        
        return datetime.now()

class VideoContentScraper:
    def __init__(self, cache_manager, rate_limiter):
        self.cache = cache_manager
        self.rate_limiter = rate_limiter
    
    async def fetch_youtube_channel_rss(self, source: Dict, max_videos: int = 10) -> List[Dict]:
        """Fetch YouTube channel videos via RSS feed"""
        cache_key = f"youtube_feed_{source['name']}"
        
        cached_data = self.cache.get(cache_key)
        if cached_data:
            try:
                logger.info(f"Using cached YouTube data for {source['name']}")
                return json.loads(cached_data)
            except json.JSONDecodeError:
                pass
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        try:
            # Use RSS URL from source configuration
            rss_url = source.get('url', f"https://www.youtube.com/feeds/videos.xml?channel_id={source.get('channel_id', '')}")
            
            logger.info(f"Fetching YouTube feed for {source['name']}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(rss_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse XML directly for YouTube RSS
            root = ET.fromstring(response.content)
            
            videos = []
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for entry in entries[:max_videos]:
                try:
                    title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                    link_elem = entry.find('.//{http://www.w3.org/2005/Atom}link')
                    published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                    
                    media_group = entry.find('.//{http://search.yahoo.com/mrss/}group')
                    description_elem = media_group.find('.//{http://search.yahoo.com/mrss/}description') if media_group else None
                    thumbnail_elem = media_group.find('.//{http://search.yahoo.com/mrss/}thumbnail') if media_group else None
                    
                    title = title_elem.text if title_elem is not None else "No title"
                    video_url = link_elem.get('href') if link_elem is not None else ""
                    description = description_elem.text if description_elem is not None else ""
                    
                    published_date = None
                    if published_elem is not None:
                        try:
                            published_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                        except ValueError:
                            pass
                    
                    thumbnail_url = ""
                    if thumbnail_elem is not None:
                        thumbnail_url = thumbnail_elem.get('url', '')
                    
                    # Extract video ID for additional info
                    video_id = ""
                    if video_url:
                        parsed_url = urlparse(video_url)
                        if 'youtube.com' in parsed_url.netloc:
                            query_params = parse_qs(parsed_url.query)
                            video_id = query_params.get('v', [''])[0]
                    
                    if not description or len(description) < 30:
                        continue
                    
                    video = {
                        'source': source['name'],
                        'title': title,
                        'description': description[:2000],
                        'url': video_url,
                        'video_url': video_url,
                        'thumbnail_url': thumbnail_url,
                        'published_date': published_date,
                        'processed': False,
                        'priority': source.get('priority', 3),
                        'video_id': video_id
                    }
                    
                    videos.append(video)
                    
                except Exception as e:
                    logger.warning(f"Error processing YouTube video: {e}")
                    continue
            
            self.cache.set(cache_key, json.dumps(videos, default=str), 3600)
            logger.info(f"Fetched {len(videos)} videos from {source['name']}")
            return videos
            
        except Exception as e:
            logger.error(f"Error fetching YouTube channel for {source['name']}: {e}")
            return []

class MultimediaContentProcessor:
    def __init__(self, claude_client, cache_manager):
        self.claude = claude_client
        self.cache = cache_manager
        self.has_claude = claude_client is not None
        
    async def process_audio_content(self, content: Dict) -> Dict:
        """Process and summarize audio content"""
        cache_key = f"audio_summary_{hashlib.md5(content['url'].encode()).hexdigest()}"
        
        cached_summary = self.cache.get(cache_key)
        if cached_summary:
            try:
                cached_data = json.loads(cached_summary)
                content.update(cached_data)
                return content
            except json.JSONDecodeError:
                pass
        
        if not self.has_claude:
            return self._fallback_audio_processing(content)
        
        try:
            prompt = f"""
            Analyze this AI podcast episode:
            
            Title: {content['title']}
            Description: {content['description'][:1000]}
            Duration: {content.get('duration', 'Unknown')} seconds
            
            Provide:
            1. A concise 5-sentence summary focusing on AI/ML topics
            2. A significance score from 1-10 for AI community
            
            Format:
            SUMMARY: [summary]
            SCORE: [number]
            """
            
            message = self.claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text
            
            summary = ""
            score = 5.0
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                elif line.startswith('SCORE:'):
                    try:
                        score = float(line.replace('SCORE:', '').strip())
                        score = max(1.0, min(10.0, score))
                    except ValueError:
                        score = 5.0
            
            if not summary:
                summary = content['description'][:200] + "..."
            
            processed_data = {
                'summary': summary,
                'significance_score': score,
                'processed': True
            }
            
            self.cache.set(cache_key, json.dumps(processed_data), 3600 * 24)
            content.update(processed_data)
            
            logger.info(f"Processed audio: {content['title'][:50]}... (Score: {score})")
            return content
            
        except Exception as e:
            logger.error(f"Error processing audio with Claude: {e}")
            return self._fallback_audio_processing(content)
    
    async def process_video_content(self, content: Dict) -> Dict:
        """Process and summarize video content"""
        cache_key = f"video_summary_{hashlib.md5(content['url'].encode()).hexdigest()}"
        
        cached_summary = self.cache.get(cache_key)
        if cached_summary:
            try:
                cached_data = json.loads(cached_summary)
                content.update(cached_data)
                return content
            except json.JSONDecodeError:
                pass
        
        if not self.has_claude:
            return self._fallback_video_processing(content)
        
        try:
            prompt = f"""
            Analyze this AI/ML YouTube video:
            
            Title: {content['title']}
            Description: {content['description'][:1000]}
            Channel: {content['source']}
            
            Provide:
            1. A concise 5-sentence summary focusing on AI/ML educational value
            2. A significance score from 1-10 for AI learning
            
            Format:
            SUMMARY: [summary]
            SCORE: [number]
            """
            
            message = self.claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text
            
            summary = ""
            score = 5.0
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                elif line.startswith('SCORE:'):
                    try:
                        score = float(line.replace('SCORE:', '').strip())
                        score = max(1.0, min(10.0, score))
                    except ValueError:
                        score = 5.0
            
            if not summary:
                summary = content['description'][:200] + "..."
            
            processed_data = {
                'summary': summary,
                'significance_score': score,
                'processed': True
            }
            
            self.cache.set(cache_key, json.dumps(processed_data), 3600 * 24)
            content.update(processed_data)
            
            logger.info(f"Processed video: {content['title'][:50]}... (Score: {score})")
            return content
            
        except Exception as e:
            logger.error(f"Error processing video with Claude: {e}")
            return self._fallback_video_processing(content)
    
    def _fallback_audio_processing(self, content: Dict) -> Dict:
        """Fallback audio processing without Claude"""
        title_lower = content['title'].lower()
        description_lower = content['description'].lower()
        
        high_impact_terms = ['openai', 'anthropic', 'gpt', 'claude', 'breakthrough', 'launch']
        medium_impact_terms = ['ai', 'machine learning', 'deep learning', 'neural network', 'llm']
        
        score = 5.0
        
        for term in high_impact_terms:
            if term in title_lower or term in description_lower:
                score += 1.0
        
        for term in medium_impact_terms:
            if term in title_lower:
                score += 0.5
        
        score = max(1.0, min(10.0, score))
        
        sentences = [s.strip() for s in content['description'].split('.') if len(s.strip()) > 20]
        summary = '. '.join(sentences[:2])
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        content['summary'] = summary
        content['significance_score'] = score
        content['processed'] = True
        
        return content
    
    def _fallback_video_processing(self, content: Dict) -> Dict:
        """Fallback video processing without Claude"""
        title_lower = content['title'].lower()
        description_lower = content['description'].lower()
        
        high_impact_terms = ['tutorial', 'explained', 'deep dive', 'research paper', 'breakthrough']
        medium_impact_terms = ['ai', 'machine learning', 'deep learning', 'python', 'tensorflow']
        
        score = 5.0
        
        for term in high_impact_terms:
            if term in title_lower or term in description_lower:
                score += 1.0
        
        for term in medium_impact_terms:
            if term in title_lower:
                score += 0.5
        
        score = max(1.0, min(10.0, score))
        
        sentences = [s.strip() for s in content['description'].split('.') if len(s.strip()) > 20]
        summary = '. '.join(sentences[:2])
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        content['summary'] = summary
        content['significance_score'] = score
        content['processed'] = True
        
        return content

class MultimediaScraper:
    def __init__(self, db_manager: MultimediaDatabaseManager, cache_manager, rate_limiter):
        self.db = db_manager
        self.cache = cache_manager
        self.rate_limiter = rate_limiter
        self.audio_scraper = AudioContentScraper(cache_manager, rate_limiter)
        self.video_scraper = VideoContentScraper(cache_manager, rate_limiter)
    
    async def scrape_all_audio_sources(self) -> List[Dict]:
        """Scrape all audio sources"""
        all_audio = []
        
        sorted_sources = sorted(
            [s for s in MULTIMEDIA_SOURCES["audio"] if s.get('enabled', True)],
            key=lambda x: x.get('priority', 3)
        )
        
        for source in sorted_sources:
            episodes = await self.audio_scraper.fetch_podcast_feed(source)
            for episode in episodes:
                if self.db.save_audio_content(episode):
                    all_audio.append(episode)
        
        logger.info(f"Scraped {len(all_audio)} total audio episodes")
        return all_audio
    
    async def scrape_all_video_sources(self) -> List[Dict]:
        """Scrape all video sources"""
        all_videos = []
        
        sorted_sources = sorted(
            [s for s in MULTIMEDIA_SOURCES["video"] if s.get('enabled', True)],
            key=lambda x: x.get('priority', 3)
        )
        
        for source in sorted_sources:
            videos = await self.video_scraper.fetch_youtube_channel_rss(source)
            for video in videos:
                if self.db.save_video_content(video):
                    all_videos.append(video)
        
        logger.info(f"Scraped {len(all_videos)} total videos")
        return all_videos
    
    async def scrape_all_multimedia(self) -> Dict:
        """Scrape all multimedia sources"""
        audio_content = await self.scrape_all_audio_sources()
        video_content = await self.scrape_all_video_sources()
        
        return {
            "audio": audio_content,
            "video": video_content,
            "total_audio": len(audio_content),
            "total_video": len(video_content)
        }