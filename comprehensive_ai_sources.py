# Comprehensive AI Sources covering all 23 AI topics with proper content types and meta tags
# This data will be integrated into the main API index.py file

COMPREHENSIVE_AI_SOURCES = [
    
    # ============================================================================
    # NOVICE-FRIENDLY SOURCES (ai-explained, ai-in-everyday-life, fun-and-interesting-ai, basic-ethics)
    # ============================================================================
    {
        "name": "AI Explained",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw",
        "website": "https://www.youtube.com/@ai-explained-",
        "content_type": "videos",
        "category": "education",
        "ai_topics": '["ai-explained", "fun-and-interesting-ai", "ai-in-everyday-life"]',
        "meta_tags": '["ai explained", "artificial intelligence", "beginner ai", "ai basics", "simple ai", "what is ai", "ai concepts", "understanding ai", "ai for beginners"]',
        "description": "Breaking down AI developments in simple terms",
        "verified": True,
        "priority": 1
    },
    {
        "name": "MIT Technology Review AI",
        "rss_url": "https://www.technologyreview.com/category/artificial-intelligence/feed/",
        "website": "https://www.technologyreview.com/artificial-intelligence/",
        "content_type": "blogs",
        "category": "news",
        "ai_topics": '["ai-explained", "ai-in-everyday-life", "basic-ethics", "fun-and-interesting-ai"]',
        "meta_tags": '["ai applications", "everyday ai", "consumer ai", "ai ethics", "responsible ai", "ai safety", "ai fairness", "practical ai"]',
        "description": "AI impact on society and everyday applications",
        "verified": True,
        "priority": 1
    },
    {
        "name": "AI for Everyone - Coursera Blog",
        "rss_url": "https://blog.coursera.org/feed/",
        "website": "https://blog.coursera.org",
        "content_type": "learning",
        "category": "education",
        "ai_topics": '["ai-explained", "basic-ethics", "ai-in-everyday-life"]',
        "meta_tags": '["ai for everyone", "non-technical ai", "ai literacy", "ai awareness", "ai impact", "ai basics", "beginner friendly"]',
        "description": "Making AI accessible to everyone",
        "verified": True,
        "priority": 2
    },
    
    # ============================================================================
    # STUDENT RESOURCES (educational-content, project-ideas, career-trends, machine-learning, deep-learning, tools-and-frameworks, data-science)
    # ============================================================================
    {
        "name": "Fast.ai",
        "rss_url": "https://www.fast.ai/feed.xml",
        "website": "https://www.fast.ai",
        "content_type": "learning",
        "category": "education",
        "ai_topics": '["educational-content", "machine-learning", "deep-learning", "project-ideas"]',
        "meta_tags": '["practical ai", "deep learning course", "machine learning tutorial", "ai projects", "coding ai", "hands-on ai"]',
        "description": "Practical deep learning for coders",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Towards Data Science",
        "rss_url": "https://towardsdatascience.com/feed",
        "website": "https://towardsdatascience.com",
        "content_type": "blogs",
        "category": "education",
        "ai_topics": '["machine-learning", "deep-learning", "data-science", "educational-content", "project-ideas"]',
        "meta_tags": '["data science", "machine learning", "python", "tensorflow", "pytorch", "data analysis", "statistics", "algorithms"]',
        "description": "Comprehensive data science and ML articles",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Papers With Code",
        "rss_url": "https://paperswithcode.com/feed.xml",
        "website": "https://paperswithcode.com",
        "content_type": "blogs",
        "category": "research",
        "ai_topics": '["machine-learning", "deep-learning", "educational-content", "project-ideas"]',
        "meta_tags": '["research papers", "code implementation", "ml papers", "github", "reproducible research", "open source ai"]',
        "description": "Latest ML papers with code implementations",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Kaggle Blog",
        "rss_url": "https://medium.com/feed/kaggle-blog",
        "website": "https://www.kaggle.com/blog",
        "content_type": "blogs",
        "category": "education",
        "ai_topics": '["data-science", "machine-learning", "project-ideas", "career-trends"]',
        "meta_tags": '["data science competitions", "kaggle", "machine learning projects", "data analysis", "predictive modeling"]',
        "description": "Data science competitions and career insights",
        "verified": True,
        "priority": 2
    },
    {
        "name": "3Blue1Brown",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw",
        "website": "https://www.youtube.com/@3blue1brown",
        "content_type": "videos",
        "category": "education",
        "ai_topics": '["educational-content", "machine-learning", "deep-learning"]',
        "meta_tags": '["neural networks", "mathematics", "linear algebra", "calculus", "visual learning", "math intuition"]',
        "description": "Mathematical concepts behind ML visually explained",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Hugging Face",
        "rss_url": "https://huggingface.co/blog/feed.xml",
        "website": "https://huggingface.co/blog",
        "content_type": "blogs",
        "category": "tools",
        "ai_topics": '["tools-and-frameworks", "machine-learning", "project-ideas", "educational-content"]',
        "meta_tags": '["transformers", "nlp", "pytorch", "tensorflow", "pretrained models", "open source", "democratizing ai"]',
        "description": "Open-source AI models and tools",
        "verified": True,
        "priority": 1
    },
    {
        "name": "DeepLearning.AI",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCcIXc5mJsHVYTZR1maL5l9w",
        "website": "https://www.youtube.com/@DeepLearningAI",
        "content_type": "videos",
        "category": "education",
        "ai_topics": '["educational-content", "deep-learning", "machine-learning", "career-trends"]',
        "meta_tags": '["andrew ng", "deep learning course", "neural networks", "ai career", "machine learning specialization"]',
        "description": "Educational AI content from Andrew Ng's team",
        "verified": True,
        "priority": 1
    },
    
    # ============================================================================
    # PROFESSIONAL SOURCES (industry-news, applied-ai, case-studies, podcasts-and-interviews, cloud-computing, robotics)
    # ============================================================================
    {
        "name": "VentureBeat AI",
        "rss_url": "https://venturebeat.com/category/ai/feed/",
        "website": "https://venturebeat.com/ai/",
        "content_type": "blogs",
        "category": "news",
        "ai_topics": '["industry-news", "applied-ai", "case-studies"]',
        "meta_tags": '["ai industry", "ai startups", "enterprise ai", "ai adoption", "business ai", "ai market", "ai trends"]',
        "description": "AI industry news and enterprise adoption",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Harvard Business Review AI",
        "rss_url": "https://feeds.hbr.org/harvardbusiness",
        "website": "https://hbr.org/topic/artificial-intelligence",
        "content_type": "blogs",
        "category": "business",
        "ai_topics": '["applied-ai", "case-studies", "strategic-implications"]',
        "meta_tags": '["business strategy", "ai transformation", "enterprise ai", "ai leadership", "digital transformation", "business cases"]',
        "description": "AI business strategy and case studies",
        "verified": True,
        "priority": 2
    },
    {
        "name": "Lex Fridman Podcast",
        "rss_url": "https://lexfridman.com/feed/podcast/",
        "website": "https://lexfridman.com/podcast/",
        "content_type": "podcasts",
        "category": "media",
        "ai_topics": '["podcasts-and-interviews", "ai-research", "leadership-and-innovation"]',
        "meta_tags": '["ai interviews", "tech leaders", "ai researchers", "long-form conversations", "artificial intelligence"]',
        "description": "Long-form conversations with AI leaders",
        "verified": True,
        "priority": 1
    },
    {
        "name": "TWIML AI Podcast",
        "rss_url": "https://twimlai.com/feed/podcast",
        "website": "https://twimlai.com",
        "content_type": "podcasts",
        "category": "media",
        "ai_topics": '["podcasts-and-interviews", "machine-learning", "applied-ai", "case-studies"]',
        "meta_tags": '["machine learning interviews", "ai practitioners", "ml in production", "real-world ai", "ai implementation"]',
        "description": "Interviews with ML practitioners and researchers",
        "verified": True,
        "priority": 1
    },
    {
        "name": "AWS AI Blog",
        "rss_url": "https://aws.amazon.com/blogs/machine-learning/feed/",
        "website": "https://aws.amazon.com/blogs/machine-learning/",
        "content_type": "blogs",
        "category": "cloud",
        "ai_topics": '["cloud-computing", "applied-ai", "tools-and-frameworks", "case-studies"]',
        "meta_tags": '["aws", "cloud ai", "sagemaker", "machine learning", "cloud computing", "scalable ai", "mlops"]',
        "description": "AWS machine learning services and case studies",
        "verified": True,
        "priority": 2
    },
    {
        "name": "Google Cloud AI Blog",
        "rss_url": "https://cloud.google.com/blog/products/ai-machine-learning/rss",
        "website": "https://cloud.google.com/blog/products/ai-machine-learning",
        "content_type": "blogs",
        "category": "cloud",
        "ai_topics": '["cloud-computing", "applied-ai", "tools-and-frameworks"]',
        "meta_tags": '["google cloud", "vertex ai", "tensorflow", "cloud ml", "automl", "ai platform"]',
        "description": "Google Cloud AI and ML platform updates",
        "verified": True,
        "priority": 2
    },
    {
        "name": "MIT Technology Review Robotics",
        "rss_url": "https://www.technologyreview.com/category/robotics/feed/",
        "website": "https://www.technologyreview.com/robotics/",
        "content_type": "blogs",
        "category": "robotics",
        "ai_topics": '["robotics", "applied-ai", "industry-news"]',
        "meta_tags": '["robotics", "autonomous systems", "robot learning", "industrial robots", "humanoid robots", "automation"]',
        "description": "Latest developments in AI-powered robotics",
        "verified": True,
        "priority": 2
    },
    
    # ============================================================================
    # EXECUTIVE SOURCES (ai-ethics-and-safety, investment-and-funding, strategic-implications, policy-and-regulation, leadership-and-innovation, ai-research)
    # ============================================================================
    {
        "name": "Stanford HAI",
        "rss_url": "https://hai.stanford.edu/news/rss.xml",
        "website": "https://hai.stanford.edu/news",
        "content_type": "blogs",
        "category": "research",
        "ai_topics": '["ai-ethics-and-safety", "policy-and-regulation", "strategic-implications", "ai-research"]',
        "meta_tags": '["human-centered ai", "ai ethics", "ai policy", "responsible ai", "ai governance", "ai safety"]',
        "description": "Stanford Human-Centered AI Institute research",
        "verified": True,
        "priority": 1
    },
    {
        "name": "AI Now Institute",
        "rss_url": "https://ainowinstitute.org/feed",
        "website": "https://ainowinstitute.org",
        "content_type": "blogs",
        "category": "policy",
        "ai_topics": '["ai-ethics-and-safety", "policy-and-regulation", "strategic-implications"]',
        "meta_tags": '["ai policy", "algorithmic bias", "ai accountability", "social impact", "ai regulation", "tech policy"]',
        "description": "AI policy and social implications research",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Anthropic",
        "rss_url": "https://www.anthropic.com/news/rss.xml",
        "website": "https://www.anthropic.com/news",
        "content_type": "blogs",
        "category": "company",
        "ai_topics": '["ai-ethics-and-safety", "ai-research", "strategic-implications"]',
        "meta_tags": '["ai safety", "constitutional ai", "ai alignment", "responsible ai", "ai research", "claude"]',
        "description": "AI safety and responsible AI development",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Brookings AI Governance",
        "rss_url": "https://www.brookings.edu/feed/",
        "website": "https://www.brookings.edu/research/?topic=artificial-intelligence",
        "content_type": "blogs",
        "category": "policy",
        "ai_topics": '["policy-and-regulation", "strategic-implications", "leadership-and-innovation"]',
        "meta_tags": '["ai governance", "technology policy", "ai regulation", "public policy", "government ai", "ai legislation"]',
        "description": "AI governance and public policy research",
        "verified": True,
        "priority": 2
    },
    {
        "name": "CB Insights AI",
        "rss_url": "https://www.cbinsights.com/research/feed/",
        "website": "https://www.cbinsights.com/research-artificial-intelligence",
        "content_type": "blogs",
        "category": "finance",
        "ai_topics": '["investment-and-funding", "strategic-implications", "industry-news"]',
        "meta_tags": '["ai investments", "venture capital", "ai startups", "funding rounds", "market analysis", "ai trends"]',
        "description": "AI investment trends and market analysis",
        "verified": True,
        "priority": 2
    },
    {
        "name": "McKinsey AI Insights",
        "rss_url": "https://www.mckinsey.com/feed/rss",
        "website": "https://www.mckinsey.com/capabilities/quantumblack/our-insights",
        "content_type": "blogs",
        "category": "consulting",
        "ai_topics": '["strategic-implications", "leadership-and-innovation", "applied-ai"]',
        "meta_tags": '["ai strategy", "digital transformation", "business innovation", "ai adoption", "enterprise ai", "c-suite"]',
        "description": "Strategic AI insights for business leaders",
        "verified": True,
        "priority": 2
    },
    
    # ============================================================================
    # RESEARCH INSTITUTIONS & CUTTING-EDGE RESEARCH
    # ============================================================================
    {
        "name": "OpenAI Blog",
        "rss_url": "https://openai.com/blog/rss.xml",
        "website": "https://openai.com/blog",
        "content_type": "blogs",
        "category": "company",
        "ai_topics": '["ai-research", "strategic-implications", "industry-news"]',
        "meta_tags": '["gpt", "chatgpt", "large language models", "openai", "artificial general intelligence", "ai research"]',
        "description": "OpenAI's latest research and developments",
        "verified": True,
        "priority": 1
    },
    {
        "name": "DeepMind Blog",
        "rss_url": "https://deepmind.google/discover/blog/rss.xml",
        "website": "https://deepmind.google/discover/blog",
        "content_type": "blogs",
        "category": "research",
        "ai_topics": '["ai-research", "deep-learning", "robotics"]',
        "meta_tags": '["alphago", "deepmind", "reinforcement learning", "protein folding", "game ai", "scientific ai"]',
        "description": "DeepMind's cutting-edge AI research",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Google AI Blog",
        "rss_url": "https://ai.googleblog.com/feeds/posts/default",
        "website": "https://ai.googleblog.com",
        "content_type": "blogs",
        "category": "company",
        "ai_topics": '["ai-research", "machine-learning", "applied-ai"]',
        "meta_tags": '["google ai", "tensorflow", "bert", "transformer", "neural networks", "google research"]',
        "description": "Google's AI research and applications",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Meta AI",
        "rss_url": "https://ai.meta.com/blog/rss/",
        "website": "https://ai.meta.com/blog",
        "content_type": "blogs",
        "category": "company",
        "ai_topics": '["ai-research", "applied-ai", "industry-news"]',
        "meta_tags": '["meta ai", "facebook ai", "pytorch", "llama", "computer vision", "nlp", "metaverse"]',
        "description": "Meta's AI research and product development",
        "verified": True,
        "priority": 1
    },
    {
        "name": "MIT CSAIL",
        "rss_url": "https://www.csail.mit.edu/news/rss.xml",
        "website": "https://www.csail.mit.edu/news",
        "content_type": "blogs",
        "category": "research",
        "ai_topics": '["ai-research", "robotics", "educational-content"]',
        "meta_tags": '["mit", "computer science", "artificial intelligence", "robotics", "machine learning", "research"]',
        "description": "MIT Computer Science and AI Lab research",
        "verified": True,
        "priority": 1
    },
    {
        "name": "Berkeley AI Research",
        "rss_url": "https://bair.berkeley.edu/blog/feed.xml",
        "website": "https://bair.berkeley.edu/blog",
        "content_type": "blogs",
        "category": "research",
        "ai_topics": '["ai-research", "machine-learning", "robotics"]',
        "meta_tags": '["berkeley", "bair", "reinforcement learning", "computer vision", "robotics", "deep learning"]',
        "description": "UC Berkeley AI research insights",
        "verified": True,
        "priority": 1
    },
    
    # ============================================================================
    # ADDITIONAL MULTIMEDIA CONTENT
    # ============================================================================
    {
        "name": "Two Minute Papers",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg",
        "website": "https://www.youtube.com/@TwoMinutePapers",
        "content_type": "videos",
        "category": "media",
        "ai_topics": '["fun-and-interesting-ai", "ai-research", "educational-content"]',
        "meta_tags": '["research papers", "ai breakthroughs", "computer graphics", "machine learning", "scientific visualization"]',
        "description": "Visual explanations of AI research papers",
        "verified": True,
        "priority": 1
    },
    {
        "name": "AI Podcast by NVIDIA",
        "rss_url": "https://blogs.nvidia.com/ai-podcast/feed/",
        "website": "https://blogs.nvidia.com/ai-podcast/",
        "content_type": "podcasts",
        "category": "media",
        "ai_topics": '["podcasts-and-interviews", "applied-ai", "robotics"]',
        "meta_tags": '["nvidia", "gpu computing", "ai applications", "deep learning", "autonomous vehicles", "robotics"]',
        "description": "AI innovators and breakthrough technologies",
        "verified": True,
        "priority": 2
    },
    
    # ============================================================================
    # EVENTS & CONFERENCES
    # ============================================================================
    {
        "name": "AI Events Global",
        "rss_url": "https://www.eventbrite.com/rss/organizer_list_events/12345",
        "website": "https://www.ai-events.org",
        "content_type": "events",
        "category": "events",
        "ai_topics": '["industry-news", "leadership-and-innovation", "career-trends"]',
        "meta_tags": '["ai conferences", "machine learning events", "ai workshops", "tech conferences", "networking", "ai summit"]',
        "description": "Global AI conferences and events",
        "verified": True,
        "priority": 3
    },
    
    # ============================================================================
    # DEMOS & INTERACTIVE CONTENT
    # ============================================================================
    {
        "name": "AI Demo Gallery",
        "rss_url": "https://aidemos.gallery/feed.xml",
        "website": "https://aidemos.gallery",
        "content_type": "demos",
        "category": "interactive",
        "ai_topics": '["fun-and-interesting-ai", "ai-explained", "project-ideas", "tools-and-frameworks"]',
        "meta_tags": '["ai demos", "interactive ai", "ai playground", "try ai", "ai examples", "hands-on ai", "ai experiments"]',
        "description": "Interactive AI demonstrations and experiments",
        "verified": True,
        "priority": 2
    },
    {
        "name": "Replicate AI Models",
        "rss_url": "https://replicate.com/blog/feed.xml",
        "website": "https://replicate.com/explore",
        "content_type": "demos",
        "category": "platform",
        "ai_topics": '["fun-and-interesting-ai", "tools-and-frameworks", "project-ideas"]',
        "meta_tags": '["ai models", "machine learning api", "ai playground", "model hosting", "ai experiments", "try models"]',
        "description": "Run AI models in the cloud with simple API",
        "verified": True,
        "priority": 2
    },
    {
        "name": "OpenAI Playground",
        "rss_url": "https://openai.com/blog/rss.xml",
        "website": "https://platform.openai.com/playground",
        "content_type": "demos",
        "category": "platform",
        "ai_topics": '["ai-explained", "fun-and-interesting-ai", "educational-content"]',
        "meta_tags": '["gpt playground", "language models", "ai chat", "text generation", "openai api", "ai interaction"]',
        "description": "Interactive playground for OpenAI's language models",
        "verified": True,
        "priority": 1
    }
]

# Topic coverage verification - all 23 topics should be covered:
TOPICS_COVERED = [
    "ai-explained", "ai-in-everyday-life", "fun-and-interesting-ai", "basic-ethics",
    "educational-content", "project-ideas", "career-trends", "machine-learning", 
    "deep-learning", "tools-and-frameworks", "data-science",
    "industry-news", "applied-ai", "case-studies", "podcasts-and-interviews", 
    "cloud-computing", "robotics", 
    "ai-ethics-and-safety", "investment-and-funding", "strategic-implications", 
    "policy-and-regulation", "leadership-and-innovation", "ai-research"
]

CONTENT_TYPES_COVERED = ["blogs", "podcasts", "videos", "learning", "events", "demos"]