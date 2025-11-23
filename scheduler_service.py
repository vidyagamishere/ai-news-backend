#!/usr/bin/env python3
"""
Auto Scheduler Service for RSS Feed Scraping
Runs scraping job every 8 hours automatically using APScheduler
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from db_service import get_database_service
from crawl4ai_scraper import AdminScrapingInterface

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseAdapter:
    """Adapter to make database service compatible with AdminScrapingInterface"""
    
    def __init__(self, db_service):
        self.db_service = db_service
    
    def get_ai_sources(self):
        """Get AI sources for scraping"""
        return self.db_service.get_ai_sources()
    
    def insert_article(self, article_data):
        """Insert article into database"""
        return self.db_service.insert_article(article_data)
    def get_ai_sources_by_frequency(self, scrape_frequency_days: int = 1):
        """Forward call to db_service.get_ai_sources_by_frequency"""
        return self.db_service.get_ai_sources_by_frequency(scrape_frequency_days)

class AutoScrapingScheduler:
    """Automated RSS feed scraping scheduler"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self.is_running = False
        self.scraping_enabled = os.getenv('AUTO_SCRAPING_ENABLED', 'true').lower() == 'true'
        self.scraping_interval_hours = int(os.getenv('SCRAPING_INTERVAL_HOURS', '8'))
        
        # Configure scheduler settings
        self.scheduler.configure({
            'apscheduler.jobstores.default': {
                'type': 'memory'
            },
            'apscheduler.executors.default': {
                'class': 'apscheduler.executors.asyncio:AsyncIOExecutor',
            },
            'apscheduler.job_defaults.coalesce': 'false',
            'apscheduler.job_defaults.max_instances': '1',
            'apscheduler.timezone': 'UTC',
        })
        
        logger.info(f"🕐 Auto-scraping scheduler initialized")
        logger.info(f"📊 Scraping enabled: {self.scraping_enabled}")
        logger.info(f"⏰ Scraping interval: every {self.scraping_interval_hours} hours")
    
    async def scrape_rss_feeds(self):
        """Perform automated RSS feed scraping"""
        try:
            scrape_start_time = datetime.now(timezone.utc)
            logger.info(f"🕷️ Starting automated RSS feed scraping at {scrape_start_time.isoformat()}")
            
            # Initialize database and scraper
            db = get_database_service()
            db_adapter = DatabaseAdapter(db)
            admin_scraper = AdminScrapingInterface(db_adapter)
            
            # Run the scraping process
            result = await admin_scraper.initiate_scraping(admin_email="admin@vidyagam.com")
            
            scrape_end_time = datetime.now(timezone.utc)
            duration = (scrape_end_time - scrape_start_time).total_seconds()
            
            if result.get('success', False):
                logger.info(f"✅ Automated scraping completed successfully in {duration:.1f}s")
                logger.info(f"📊 Sources scraped: {result.get('sources_scraped', 0)}")
                logger.info(f"📄 Articles found: {result.get('articles_found', 0)}")  
                logger.info(f"💾 Articles processed: {result.get('articles_processed', 0)}")
                logger.info(f"⏰ Next scraping scheduled in {self.scraping_interval_hours} hours")
            else:
                logger.error(f"❌ Automated scraping failed: {result.get('message', 'Unknown error')}")
                logger.error(f"⏰ Will retry in {self.scraping_interval_hours} hours")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Automated scraping process failed: {str(e)}")
            logger.error(f"⏰ Will retry in {self.scraping_interval_hours} hours")
            return {
                "success": False,
                "message": f"Scheduler error: {str(e)}",
                "articles_processed": 0
            }
    
    def start_scheduler(self):
        """Start the automated scraping scheduler"""
        try:
            if not self.scraping_enabled:
                logger.info("⏭️ Auto-scraping is disabled via environment variable")
                return
                
            if self.is_running:
                logger.warning("⚠️ Scheduler is already running")
                return
            
            # Add the scraping job with interval trigger
            self.scheduler.add_job(
                func=self.scrape_rss_feeds,
                trigger=IntervalTrigger(hours=self.scraping_interval_hours),
                id='auto_rss_scraping',
                name=f'Auto RSS Scraping (every {self.scraping_interval_hours}h)',
                replace_existing=True,
                misfire_grace_time=3600,  # Allow up to 1 hour delay
                coalesce=True,  # Combine missed executions
                max_instances=1  # Only one scraping job at a time
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            next_run = self.scheduler.get_job('auto_rss_scraping').next_run_time
            logger.info(f"🚀 Auto-scraping scheduler started successfully")
            logger.info(f"📅 First scraping scheduled for: {next_run.isoformat()}")
            logger.info(f"⏰ Interval: every {self.scraping_interval_hours} hours")
            
            # Optionally run initial scraping immediately
            run_initial = os.getenv('RUN_INITIAL_SCRAPING', 'false').lower() == 'true'
            if run_initial:
                logger.info("🏃 Running initial scraping immediately...")
                # Schedule immediate execution
                self.scheduler.add_job(
                    func=self.scrape_rss_feeds,
                    trigger='date',
                    run_date=datetime.now(timezone.utc),
                    id='initial_scraping',
                    name='Initial RSS Scraping',
                    misfire_grace_time=300
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {str(e)}")
            self.is_running = False
            raise e
    
    def stop_scheduler(self):
        """Stop the automated scraping scheduler"""
        try:
            if not self.is_running:
                logger.warning("⚠️ Scheduler is not running")
                return
            
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("🛑 Auto-scraping scheduler stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop scheduler: {str(e)}")
    
    def get_scheduler_status(self):
        """Get current scheduler status and job information"""
        try:
            if not self.is_running:
                return {
                    "status": "stopped",
                    "scraping_enabled": self.scraping_enabled,
                    "interval_hours": self.scraping_interval_hours,
                    "jobs": []
                }
            
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
            
            return {
                "status": "running",
                "scraping_enabled": self.scraping_enabled,
                "interval_hours": self.scraping_interval_hours,
                "jobs": jobs,
                "scheduler_running": self.scheduler.running
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get scheduler status: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "scraping_enabled": self.scraping_enabled,
                "interval_hours": self.scraping_interval_hours
            }
    

# Global scheduler instance
auto_scheduler = None

def get_scheduler() -> AutoScrapingScheduler:
    """Get global scheduler instance"""
    global auto_scheduler
    if auto_scheduler is None:
        auto_scheduler = AutoScrapingScheduler()
    return auto_scheduler

def start_auto_scheduler():
    """Start the global auto-scraping scheduler"""
    scheduler = get_scheduler()
    scheduler.start_scheduler()
    return scheduler

def stop_auto_scheduler():
    """Stop the global auto-scraping scheduler"""
    global auto_scheduler
    if auto_scheduler and auto_scheduler.is_running:
        auto_scheduler.stop_scheduler()

# Cleanup function
import atexit
atexit.register(stop_auto_scheduler)
#
#if __name__ == "__main__":
#    import asyncio

#    async def main():
        # Start the scheduler (will schedule jobs as per interval)
#        start_auto_scheduler()
        # Manually trigger a scraping job for immediate test
#        scheduler = get_scheduler()
#        await scheduler.scrape_rss_feeds()
        # Keep the event loop running so APScheduler jobs can execute
#        while True:
#            await asyncio.sleep(3600)

#    asyncio.run(main())
#