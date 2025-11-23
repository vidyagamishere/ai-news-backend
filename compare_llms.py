import asyncio
import time
from crawl4ai_scraper import Crawl4AIScraper

async def compare_llm_performance():
    """Compare Claude, Gemini, and Ollama performance"""
    scraper = Crawl4AIScraper()
    
    test_url = "https://www.theverge.com/ai-artificial-intelligence"
    
    models = ['claude', 'gemini', 'ollama']
    results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {model.upper()}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            article = await scraper.scrape_article(test_url, llm_model=model)
            
            elapsed = time.time() - start_time
            
            results[model] = {
                'success': article is not None,
                'time': elapsed,
                'title': article.title if article else 'Failed',
                'score': article.significance_score if article else 0,
                'model_used': article.llm_processed if article else 'N/A'
            }
            
            print(f"✅ Success: {results[model]['success']}")
            print(f"⏱️  Time: {elapsed:.2f}s")
            print(f"📝 Title: {results[model]['title'][:60]}...")
            print(f"🎯 Score: {results[model]['score']}")
            print(f"🤖 Model: {results[model]['model_used']}")
            
        except Exception as e:
            results[model] = {
                'success': False,
                'time': time.time() - start_time,
                'error': str(e)
            }
            print(f"❌ Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}\n")
    
    for model, data in results.items():
        if data['success']:
            print(f"{model.upper():10} | ⏱️  {data['time']:6.2f}s | ✅ Success")
        else:
            print(f"{model.upper():10} | ⏱️  {data['time']:6.2f}s | ❌ Failed")

if __name__ == "__main__":
    asyncio.run(compare_llm_performance())