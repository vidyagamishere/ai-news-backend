# top_stories_config.py - Top Stories Configuration with Verified URLs

TOP_STORIES = [
    {
        "title": "OpenAI Announces GPT-4 Turbo Model", 
        "source": "OpenAI",
        "significanceScore": 9.2,
        "url": "https://openai.com/blog/chatgpt",
        "imageUrl": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=500&h=300&fit=crop&q=80",
        "summary": "OpenAI unveils GPT-4 Turbo with enhanced capabilities, longer context windows, and improved performance for complex reasoning tasks."
    },
    {
        "title": "Google Introduces Gemini AI Model",
        "source": "Google",
        "significanceScore": 8.8,
        "url": "https://blog.google/technology/ai/google-gemini-ai/",
        "imageUrl": "https://images.unsplash.com/photo-1573804633927-bfcbcd909acd?w=400&h=250&fit=crop&q=80",
        "summary": "Google's most capable AI model yet, designed to understand and operate across text, code, audio, image and video."
    },
    {
        "title": "Anthropic Claude 3 Family Released",
        "source": "Anthropic", 
        "significanceScore": 9.0,
        "url": "https://www.anthropic.com/news/claude-3-family",
        "imageUrl": "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=400&h=250&fit=crop&q=80",
        "summary": "Three new models - Opus, Sonnet, and Haiku - setting new industry benchmarks across cognitive tasks."
    },
    {
        "title": "Meta AI Releases Llama 2 Open Source Model",
        "source": "Meta",
        "significanceScore": 8.5,
        "url": "https://ai.meta.com/blog/llama-2/",
        "imageUrl": "https://images.unsplash.com/photo-1611224923853-80b023f02d71?w=400&h=250&fit=crop&q=80",
        "summary": "Open-source large language model available for research and commercial use, promoting democratized AI development."
    },
    {
        "title": "NVIDIA H100 Tensor Core GPU Powers AI Training",
        "source": "NVIDIA",
        "significanceScore": 8.3,
        "url": "https://www.nvidia.com/en-us/data-center/h100/",
        "imageUrl": "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=250&fit=crop&q=80",
        "summary": "Advanced GPU architecture designed specifically for large-scale AI training and inference workloads."
    },
    {
        "title": "Microsoft Copilot for Microsoft 365",
        "source": "Microsoft",
        "significanceScore": 8.1,
        "url": "https://www.microsoft.com/en-us/microsoft-365/microsoft-copilot",
        "imageUrl": "https://images.unsplash.com/photo-1633419461186-7d40a38105ec?w=400&h=250&fit=crop&q=80",
        "summary": "AI-powered productivity assistant integrated across Microsoft 365 applications for enhanced workplace efficiency."
    },
    {
        "title": "Stability AI Releases Stable Diffusion XL",
        "source": "Stability AI",
        "significanceScore": 7.9,
        "url": "https://stability.ai/blog/stable-diffusion-sdxl-1-announcement",
        "imageUrl": "https://images.unsplash.com/photo-1686191128892-16fd78e90d3a?w=400&h=250&fit=crop&q=80",
        "summary": "Latest version of text-to-image AI model with improved image quality and more detailed prompt understanding."
    },
    {
        "title": "Amazon Bedrock Foundation Model Service",
        "source": "AWS",
        "significanceScore": 7.7,
        "url": "https://aws.amazon.com/bedrock/",
        "imageUrl": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&h=250&fit=crop&q=80",
        "summary": "Fully managed service providing access to foundation models from leading AI companies through a single API."
    },
    {
        "title": "Hugging Face Transformers Library Updates",
        "source": "Hugging Face",
        "significanceScore": 7.5,
        "url": "https://huggingface.co/blog/transformers",
        "imageUrl": "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=400&h=250&fit=crop&q=80",
        "summary": "Major updates to the popular open-source library enabling easy access to state-of-the-art machine learning models."
    },
    {
        "title": "DeepMind AlphaFold Protein Structure Prediction",
        "source": "DeepMind",
        "significanceScore": 7.3,
        "url": "https://deepmind.google/technologies/alphafold/",
        "imageUrl": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400&h=250&fit=crop&q=80",
        "summary": "Revolutionary AI system that predicts protein structures with unprecedented accuracy, advancing biological research."
    }
]

# Configuration for dynamic URL validation and updates
URL_VALIDATION_CONFIG = {
    "enable_dynamic_scraping": False,  # Set to True to scrape images and summaries from URLs (currently using curated content)
    "check_urls_on_startup": False,  # Set to True to validate URLs when API starts
    "fallback_url": "https://news.google.com/search?q=AI+technology",  # Fallback if URL is broken
    "update_interval_hours": 24,  # How often to refresh story URLs
    "fallback_image": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=500&h=300&fit=crop&q=80"  # Default AI image
}