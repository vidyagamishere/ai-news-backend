# ai_sources_config_updated.py - Enhanced with Free AI Resources
# Focuses on completely free and accessible AI content sources

# Content type definitions
CONTENT_TYPES = {
    "all_sources": {
        "name": "All Sources",
        "description": "Comprehensive AI content from all our curated sources",
        "icon": "🌐"
    },
    "blogs": {
        "name": "Blogs",
        "description": "Expert insights, analysis, and thought leadership articles",
        "icon": "✍️"
    },
    "podcasts": {
        "name": "Podcasts", 
        "description": "Audio content, interviews, and discussions from AI leaders",
        "icon": "🎧"
    },
    "videos": {
        "name": "Videos",
        "description": "Visual content, presentations, and educational videos",
        "icon": "📹"
    },
    "events": {
        "name": "Events", 
        "description": "AI conferences, webinars, workshops, and networking events",
        "icon": "📅"
    },
    "learn": {
        "name": "Learn",
        "description": "Courses, tutorials, educational content, and skill development",
        "icon": "🎓"
    },
    "newsletters": {
        "name": "Newsletters",
        "description": "Daily and weekly AI newsletters and digests", 
        "icon": "📬"
    }
}

AI_SOURCES = [
    # ============================================================================
    # FREE AI NEWSLETTERS & AGGREGATORS - Priority 1 (Highest)
    # ============================================================================
    
    # DeepLearning.AI - The Batch (Completely Free)
    {
        "name": "The Batch by DeepLearning.AI",
        "rss_url": "https://www.deeplearning.ai/the-batch/rss.xml",
        "website": "https://www.deeplearning.ai/the-batch/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters",
        "content_type": "newsletters",
        "description": "Weekly AI insights from Andrew Ng and team - completely free"
    },
    
    # AI Breakfast - Free daily digest  
    {
        "name": "AI Breakfast",
        "rss_url": "https://aibreakfast.substack.com/feed",
        "website": "https://aibreakfast.substack.com/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters", 
        "content_type": "newsletters",
        "description": "Free daily AI news digest"
    },
    
    # The Rundown AI - Free version
    {
        "name": "The Rundown AI",
        "rss_url": "https://www.therundown.ai/rss/",
        "website": "https://www.therundown.ai/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters",
        "content_type": "newsletters", 
        "description": "Daily AI newsletter with free tier"
    },
    
    # Towards AI Newsletter - Free
    {
        "name": "Towards AI Newsletter",
        "rss_url": "https://towardsai.net/feed",
        "website": "https://towardsai.net/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters",
        "content_type": "newsletters",
        "description": "Free AI newsletter and publication"
    },
    
    # Import AI - Free
    {
        "name": "Import AI",
        "rss_url": "https://jack-clark.net/feed/",
        "website": "https://jack-clark.net/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters",
        "content_type": "newsletters",
        "description": "Jack Clark's weekly AI newsletter - completely free"
    },
    
    # Gradient Flow - Free content
    {
        "name": "Gradient Flow",
        "rss_url": "https://gradientflow.com/feed/",
        "website": "https://gradientflow.com/",
        "enabled": True,
        "priority": 1,
        "category": "newsletters",
        "content_type": "newsletters",
        "description": "AI insights with substantial free content"
    },

    # ============================================================================
    # FREE BLOG SOURCES - Priority 1
    # ============================================================================
    
    # Towards Data Science (Medium - free articles)
    {
        "name": "Towards Data Science",
        "rss_url": "https://towardsdatascience.com/feed",
        "website": "https://towardsdatascience.com",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs",
        "description": "Free AI/ML articles on Medium"
    },
    
    # Distill.pub - Completely free
    {
        "name": "Distill.pub",
        "rss_url": "https://distill.pub/rss.xml", 
        "website": "https://distill.pub",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs",
        "description": "Visual explanations of ML concepts - completely free"
    },
    
    # OpenAI Blog - Free
    {
        "name": "OpenAI Blog",
        "rss_url": "https://openai.com/blog/rss.xml",
        "website": "https://openai.com/blog",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs",
        "description": "Official OpenAI announcements and research - free"
    },
    
    # Anthropic Blog - Free
    {
        "name": "Anthropic Blog", 
        "rss_url": "https://www.anthropic.com/news/rss.xml",
        "website": "https://www.anthropic.com/news",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs", 
        "description": "Anthropic research and announcements - free"
    },
    
    # Google AI Blog - Free
    {
        "name": "Google AI Blog",
        "rss_url": "https://ai.googleblog.com/feeds/posts/default",
        "website": "https://ai.googleblog.com",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs",
        "description": "Google's AI research blog - completely free"
    },
    
    # Papers With Code - Free
    {
        "name": "Papers With Code",
        "rss_url": "https://paperswithcode.com/feed.xml",
        "website": "https://paperswithcode.com",
        "enabled": True,
        "priority": 1,
        "category": "blogs",
        "content_type": "blogs",
        "description": "Latest ML papers with code implementations - free"
    },

    # ============================================================================
    # FREE PODCAST SOURCES - Priority 1
    # ============================================================================
    
    # Lex Fridman - Free
    {
        "name": "Lex Fridman Podcast",
        "rss_url": "https://lexfridman.com/feed/podcast/",
        "website": "https://lexfridman.com/podcast/",
        "enabled": True,
        "priority": 1,
        "category": "podcasts",
        "content_type": "podcasts",
        "description": "Long-form AI conversations - completely free"
    },
    
    # Machine Learning Street Talk - Free
    {
        "name": "Machine Learning Street Talk",
        "rss_url": "https://anchor.fm/s/26e1b2b8/podcast/rss",
        "website": "https://www.youtube.com/c/MachineLearningStreetTalk",
        "enabled": True,
        "priority": 1,
        "category": "podcasts", 
        "content_type": "podcasts",
        "description": "Technical ML discussions - completely free"
    },
    
    # AI Today - Free (Note: Need to find correct RSS)
    {
        "name": "AI Today Podcast",
        "rss_url": "https://feeds.feedburner.com/oreilly/radar",
        "website": "https://www.cognilytica.com/aitoday/",
        "enabled": True,
        "priority": 1,
        "category": "podcasts",
        "content_type": "podcasts", 
        "description": "AI industry insights - free"
    },
    
    # The AI Podcast (NVIDIA) - Free
    {
        "name": "The AI Podcast",
        "rss_url": "https://blogs.nvidia.com/ai-podcast/feed/",
        "website": "https://blogs.nvidia.com/ai-podcast/",
        "enabled": True,
        "priority": 1,
        "category": "podcasts",
        "content_type": "podcasts",
        "description": "NVIDIA's AI podcast - completely free"
    },
    
    # TWIML AI Podcast - Free
    {
        "name": "The TWIML AI Podcast", 
        "rss_url": "https://twimlai.com/feed/podcast/",
        "website": "https://twimlai.com/podcast/",
        "enabled": True,
        "priority": 1,
        "category": "podcasts",
        "content_type": "podcasts",
        "description": "This Week in Machine Learning - free"
    },

    # ============================================================================
    # FREE VIDEO SOURCES (YouTube) - Priority 1
    # ============================================================================
    
    # Two Minute Papers - Free
    {
        "name": "Two Minute Papers",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg",
        "website": "https://www.youtube.com/c/TwoMinutePapers",
        "enabled": True,
        "priority": 1,
        "category": "videos",
        "content_type": "videos",
        "description": "AI research explained in 2 minutes - completely free"
    },
    
    # AI Explained - Free
    {
        "name": "AI Explained",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw",
        "website": "https://www.youtube.com/c/AIExplained-Official",
        "enabled": True,
        "priority": 1,
        "category": "videos",
        "content_type": "videos",
        "description": "Clear AI explanations - completely free"
    },
    
    # 3Blue1Brown - Free
    {
        "name": "3Blue1Brown",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw",
        "website": "https://www.youtube.com/c/3blue1brown",
        "enabled": True,
        "priority": 1,
        "category": "videos",
        "content_type": "videos",
        "description": "Mathematical concepts visually explained - free"
    },
    
    # Yannic Kilcher - Free
    {
        "name": "Yannic Kilcher",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCZHmQk67mSJgfCCTn7xBfew",
        "website": "https://www.youtube.com/c/YannicKilcher",
        "enabled": True,
        "priority": 1,
        "category": "videos",
        "content_type": "videos",
        "description": "Paper reviews and ML discussions - free"
    },
    
    # DeepLearning.AI YouTube - Free
    {
        "name": "DeepLearning.AI YouTube",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCcIXc5mJsHVYTZR1maL5l9w",
        "website": "https://www.youtube.com/c/Deeplearningai",
        "enabled": True,
        "priority": 1,
        "category": "videos",
        "content_type": "videos",
        "description": "Andrew Ng's ML content - completely free"
    },

    # ============================================================================
    # FREE LEARNING RESOURCES - Priority 1
    # ============================================================================
    
    # Fast.ai - Completely free courses
    {
        "name": "Fast.ai",
        "rss_url": "https://www.fast.ai/feed.xml",
        "website": "https://www.fast.ai",
        "enabled": True,
        "priority": 1,
        "category": "learning",
        "content_type": "learn",
        "description": "Practical deep learning course - completely free"
    },
    
    # Papers With Code - Free
    {
        "name": "Papers With Code Learning",
        "rss_url": "https://paperswithcode.com/feed.xml",
        "website": "https://paperswithcode.com",
        "enabled": True,
        "priority": 1,
        "category": "learning",
        "content_type": "learn",
        "description": "State-of-the-art ML papers with code - free"
    },
    
    # Coursera - Free audit option
    {
        "name": "Coursera AI Courses (Free Audit)",
        "rss_url": "https://blog.coursera.org/feed/",
        "website": "https://www.coursera.org/browse/data-science/machine-learning",
        "enabled": True,
        "priority": 2,
        "category": "learning",
        "content_type": "learn",
        "description": "Audit Coursera AI courses for free"
    },
    
    # MIT OpenCourseWare - Free
    {
        "name": "MIT OpenCourseWare AI",
        "rss_url": "https://ocw.mit.edu/rss/new/",
        "website": "https://ocw.mit.edu/",
        "enabled": True,
        "priority": 1,
        "category": "learning",
        "content_type": "learn",
        "description": "MIT courses freely available online"
    },
    
    # Stanford CS229, CS231n, etc. - Free
    {
        "name": "Stanford AI Courses",
        "rss_url": "https://ai.stanford.edu/blog/rss.xml",
        "website": "http://cs229.stanford.edu/",
        "enabled": True,
        "priority": 1,
        "category": "learning", 
        "content_type": "learn",
        "description": "Stanford AI course materials - free"
    },

    # ============================================================================
    # FREE EVENTS - Priority 1
    # ============================================================================
    
    # Meetup.com AI Events - Free
    {
        "name": "Meetup AI Events",
        "rss_url": "https://www.meetup.com/topics/artificial-intelligence/rss/",
        "website": "https://www.meetup.com/topics/artificial-intelligence/",
        "enabled": True,
        "priority": 1,
        "category": "events",
        "content_type": "events", 
        "description": "Local AI meetups - mostly free"
    },
    
    # AI Events aggregator
    {
        "name": "AI Events Aggregator",
        "rss_url": "https://eventil.com/events.rss?q=artificial+intelligence",
        "website": "https://eventil.com/events?q=artificial+intelligence",
        "enabled": True,
        "priority": 1,
        "category": "events",
        "content_type": "events",
        "description": "Free AI events and conferences"
    },
    
    # University AI Seminars (Many free)
    {
        "name": "Stanford HAI Events",
        "rss_url": "https://hai.stanford.edu/events/rss.xml",
        "website": "https://hai.stanford.edu/events",
        "enabled": True,
        "priority": 1,
        "category": "events",
        "content_type": "events",
        "description": "Stanford AI seminars - often free and online"
    },
    
    # MIT CSAIL Events
    {
        "name": "MIT CSAIL Events",
        "rss_url": "https://www.csail.mit.edu/news/rss.xml",
        "website": "https://www.csail.mit.edu/events",
        "enabled": True,
        "priority": 1,
        "category": "events", 
        "content_type": "events",
        "description": "MIT AI events - many free"
    },

    # ============================================================================
    # PRIORITY 2: Major AI Companies & Research Labs (All Free Blogs)
    # ============================================================================
    
    # Meta AI - Free
    {
        "name": "Meta AI",
        "rss_url": "https://ai.meta.com/blog/rss",
        "website": "https://ai.meta.com/blog",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # Microsoft AI - Free  
    {
        "name": "Microsoft AI",
        "rss_url": "https://azure.microsoft.com/en-us/blog/feed",
        "website": "https://azure.microsoft.com/en-us/blog",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # Google DeepMind - Free
    {
        "name": "Google DeepMind",
        "rss_url": "https://deepmind.google/discover/blog/rss.xml",
        "website": "https://deepmind.google/discover/blog",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # NVIDIA AI - Free
    {
        "name": "NVIDIA AI Blog",
        "rss_url": "https://blogs.nvidia.com/blog/category/artificial-intelligence/feed",
        "website": "https://blogs.nvidia.com/blog/category/artificial-intelligence",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },

    # ============================================================================
    # PRIORITY 2: Research Institutions (All Free)
    # ============================================================================
    
    # MIT CSAIL
    {
        "name": "MIT CSAIL",
        "rss_url": "https://www.csail.mit.edu/news/rss.xml",
        "website": "https://www.csail.mit.edu/news",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # Stanford HAI
    {
        "name": "Stanford HAI",
        "rss_url": "https://hai.stanford.edu/news/rss.xml",
        "website": "https://hai.stanford.edu/news",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # Berkeley AI Research
    {
        "name": "Berkeley AI Research",
        "rss_url": "https://bair.berkeley.edu/blog/feed.xml",
        "website": "https://bair.berkeley.edu/blog",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # CMU Machine Learning
    {
        "name": "CMU Machine Learning",
        "rss_url": "https://blog.ml.cmu.edu/feed",
        "website": "https://blog.ml.cmu.edu",
        "enabled": True,
        "priority": 2,
        "category": "blogs",
        "content_type": "blogs"
    },

    # ============================================================================
    # PRIORITY 3: Free Platform Sources
    # ============================================================================
    
    # Hugging Face - Free
    {
        "name": "Hugging Face Blog",
        "rss_url": "https://huggingface.co/blog/feed.xml",
        "website": "https://huggingface.co/blog",
        "enabled": True,
        "priority": 3,
        "category": "blogs",
        "content_type": "blogs"
    },
    
    # The Gradient - Free content available
    {
        "name": "The Gradient",
        "rss_url": "https://thegradient.pub/rss",
        "website": "https://thegradient.pub",
        "enabled": True,
        "priority": 3,
        "category": "blogs",
        "content_type": "blogs"
    },

    # ============================================================================
    # PRIORITY 3: Additional Free YouTube Educational Channels
    # ============================================================================
    
    # Computerphile - Free
    {
        "name": "Computerphile AI Videos",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC9-y-6csu5WGm29I7JiwpnA",
        "website": "https://www.youtube.com/c/Computerphile",
        "enabled": True,
        "priority": 3,
        "category": "videos",
        "content_type": "videos"
    },
    
    # StatQuest - Free
    {
        "name": "StatQuest with Josh Starmer",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCtYLUTtgS3k1Fg4y5tAhLbw",
        "website": "https://www.youtube.com/c/joshstarmer",
        "enabled": True,
        "priority": 3,
        "category": "videos",
        "content_type": "videos"
    },
    
    # MIT OpenCourseWare YouTube - Free
    {
        "name": "MIT OpenCourseWare",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCEBb1b_L6zDS3xTUrIALZOw",
        "website": "https://www.youtube.com/c/mitocw",
        "enabled": True,
        "priority": 3,
        "category": "videos", 
        "content_type": "videos"
    },
    
    # Stanford University - Free
    {
        "name": "Stanford University YouTube",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC-EnprmCZ3OXyAoG7vjVNCA",
        "website": "https://www.youtube.com/c/stanford",
        "enabled": True,
        "priority": 3,
        "category": "videos",
        "content_type": "videos"
    }
]

# Configuration for fallback URLs when RSS feeds are not available
FALLBACK_SCRAPING = {
    "enabled": True,
    "max_articles_per_source": 5,  # Reduced for free sources
    "timeout_seconds": 30,
    "respect_robots_txt": True,
    "delay_between_requests": 1.0  # Be respectful to free sources
}

# Categories for filtering and organization
CATEGORIES = [
    "newsletters", "blogs", "podcasts", "videos", "events", "learning",
    "company", "research", "platform", "free_resources"
]

# Content type filtering functions
def get_sources_by_content_type(content_type):
    """Filter sources by content type"""
    if content_type == "all_sources":
        return AI_SOURCES
    return [source for source in AI_SOURCES if source.get("content_type") == content_type]

def get_enabled_sources_by_type(content_type):
    """Get enabled sources filtered by content type"""
    sources = get_sources_by_content_type(content_type)
    return [source for source in sources if source.get("enabled", True)]

def get_free_sources_only():
    """Get only completely free sources (priority 1-2)"""
    return [source for source in AI_SOURCES if source.get("priority", 5) <= 2]

def get_newsletter_sources():
    """Get only newsletter/digest sources"""
    return [source for source in AI_SOURCES if source.get("content_type") == "newsletters"]

# Helper function to validate RSS feeds
def validate_rss_feed(url):
    """Basic RSS feed validation (implement with requests/feedparser)"""
    # This would be implemented in the actual scraper
    pass

# Free resource recommendations for users
FREE_RESOURCE_RECOMMENDATIONS = {
    "daily_digest": ["The Batch by DeepLearning.AI", "AI Breakfast", "The Rundown AI"],
    "deep_learning": ["Fast.ai", "3Blue1Brown", "DeepLearning.AI YouTube"],
    "research_papers": ["Papers With Code", "Distill.pub", "Two Minute Papers"], 
    "industry_insights": ["Import AI", "Lex Fridman Podcast", "The AI Podcast"],
    "learning_paths": ["MIT OpenCourseWare AI", "Stanford AI Courses", "Coursera AI Courses (Free Audit)"],
    "community_events": ["Meetup AI Events", "Stanford HAI Events", "MIT CSAIL Events"]
}