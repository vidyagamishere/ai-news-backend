#!/usr/bin/env python3
"""
RSS Feed Validator for Free AI Sources
Tests all the free AI resource RSS feeds to ensure they're working
"""

import requests
import feedparser
import time
from urllib.parse import urlparse
import sys
from ai_sources_config_updated import AI_SOURCES

def validate_rss_feed(url, timeout=10):
    """Validate an RSS feed URL"""
    try:
        # First check if URL is accessible
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AINewsBot/1.0; +http://example.com/bot)'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse the feed
        feed = feedparser.parse(response.content)
        
        if feed.bozo:
            return {
                'status': 'warning',
                'message': f'Feed has issues: {feed.bozo_exception}',
                'entries': len(feed.entries) if hasattr(feed, 'entries') else 0
            }
        
        return {
            'status': 'success',
            'message': f'Valid RSS feed with {len(feed.entries)} entries',
            'entries': len(feed.entries),
            'title': getattr(feed.feed, 'title', 'No title'),
            'description': getattr(feed.feed, 'description', 'No description')[:100]
        }
        
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': 'Request timeout'}
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': f'Request error: {str(e)}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Parse error: {str(e)}'}

def test_youtube_rss_feeds():
    """Test YouTube RSS feeds specifically"""
    youtube_feeds = [
        source for source in AI_SOURCES 
        if 'youtube.com' in source.get('rss_url', '')
    ]
    
    print(f"\n🎥 Testing {len(youtube_feeds)} YouTube RSS feeds...")
    
    for source in youtube_feeds:
        result = validate_rss_feed(source['rss_url'])
        status_emoji = '✅' if result['status'] == 'success' else '⚠️' if result['status'] == 'warning' else '❌'
        print(f"{status_emoji} {source['name']}: {result['message']}")
        time.sleep(1)  # Be respectful
    
    return youtube_feeds

def test_newsletter_feeds():
    """Test newsletter RSS feeds"""
    newsletter_feeds = [
        source for source in AI_SOURCES 
        if source.get('content_type') == 'newsletters'
    ]
    
    print(f"\n📬 Testing {len(newsletter_feeds)} Newsletter RSS feeds...")
    
    working_feeds = []
    for source in newsletter_feeds:
        result = validate_rss_feed(source['rss_url'])
        status_emoji = '✅' if result['status'] == 'success' else '⚠️' if result['status'] == 'warning' else '❌'
        print(f"{status_emoji} {source['name']}: {result['message']}")
        
        if result['status'] in ['success', 'warning']:
            working_feeds.append(source)
            
        time.sleep(1)  # Be respectful
    
    return working_feeds

def test_all_free_sources():
    """Test all free sources (priority 1-2)"""
    free_sources = [source for source in AI_SOURCES if source.get('priority', 5) <= 2]
    
    print(f"\n🆓 Testing {len(free_sources)} Free AI sources...")
    print("=" * 60)
    
    results = {
        'success': [],
        'warning': [],
        'error': []
    }
    
    for i, source in enumerate(free_sources, 1):
        print(f"\n[{i}/{len(free_sources)}] Testing: {source['name']}")
        result = validate_rss_feed(source['rss_url'])
        
        status_emoji = '✅' if result['status'] == 'success' else '⚠️' if result['status'] == 'warning' else '❌'
        print(f"  {status_emoji} Status: {result['message']}")
        print(f"  📍 URL: {source['rss_url']}")
        print(f"  🏷️  Category: {source.get('content_type', 'N/A')}")
        
        results[result['status']].append(source)
        time.sleep(1)  # Be respectful to free services
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Working feeds: {len(results['success'])}")
    print(f"⚠️  Feeds with warnings: {len(results['warning'])}")
    print(f"❌ Broken feeds: {len(results['error'])}")
    print(f"📈 Success rate: {(len(results['success']) + len(results['warning'])) / len(free_sources) * 100:.1f}%")
    
    if results['error']:
        print(f"\n❌ BROKEN FEEDS TO FIX:")
        for source in results['error']:
            print(f"  - {source['name']}: {source['rss_url']}")
    
    return results

def generate_fixed_config(results):
    """Generate a config file with only working sources"""
    working_sources = results['success'] + results['warning']
    
    print(f"\n📝 Generating config with {len(working_sources)} working sources...")
    
    config_content = '''# ai_sources_config_validated.py - Only Working Free Sources
# Generated automatically after validation

from ai_sources_config_updated import CONTENT_TYPES, FALLBACK_SCRAPING, CATEGORIES

# Only working sources after validation
AI_SOURCES_VALIDATED = [
'''
    
    for source in working_sources:
        config_content += f'''    {{
        "name": "{source['name']}",
        "rss_url": "{source['rss_url']}",
        "website": "{source['website']}",
        "enabled": True,
        "priority": {source['priority']},
        "category": "{source['category']}",
        "content_type": "{source.get('content_type', 'blogs')}"
    }},
'''
    
    config_content += ''']

# Use validated sources
AI_SOURCES = AI_SOURCES_VALIDATED
'''
    
    with open('/Users/vijayansubramaniyan/Desktop/AI-ML/Projects/ai-news-scraper/ai_sources_config_validated.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Generated: ai_sources_config_validated.py")

def main():
    """Main validation function"""
    print("🔍 AI NEWS SCRAPER - FREE SOURCES VALIDATION")
    print("=" * 60)
    print("Testing RSS feeds for free AI resources...")
    
    # Test all free sources
    results = test_all_free_sources()
    
    # Test specific categories
    test_newsletter_feeds()
    test_youtube_rss_feeds()
    
    # Generate validated config
    generate_fixed_config(results)
    
    print(f"\n✅ Validation complete!")
    print(f"💡 Next steps:")
    print(f"   1. Review broken feeds and find alternatives")  
    print(f"   2. Replace ai_sources_config.py with validated version")
    print(f"   3. Test the scraper with new configuration")
    print(f"   4. Deploy updated backend to Vercel")

if __name__ == "__main__":
    main()