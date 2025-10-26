#!/usr/bin/env python3
"""
Scheduler service for automated scraping
Runs in the background at specified intervals
"""

import os
import logging
import asyncio
import traceback
from typing import Optional
from fastapi import Depends

from app.services.content_service import ContentService
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

# Get DEBUG mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

class SchedulerService:
    """Background scheduler for automated scraping"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.content_service = ContentService()
        # ✅ SET DEFAULT LLM MODEL TO GEMINI
        self.default_llm_model = 'gemini'
        logger.info(f"🤖 Scheduler initialized with default LLM model: {self.default_llm_model}")
    
    async def run_scraping_job(self):
        """Run the scraping job (async)"""
        try:
            logger.info("⏰ Scheduled scraping job triggered")
            logger.info(f"🤖 Using LLM model: {self.default_llm_model}")
            
            # ✅ PASS GEMINI AS DEFAULT LLM MODEL
            result = await self.content_service.scrape_content(llm_model=self.default_llm_model)
            
            logger.info(f"✅ Scheduled scraping completed: {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Scheduled scraping failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    # ✅ NEW: Synchronous wrapper for APScheduler
    def _run_scraping_job_sync(self):
        """
        Synchronous wrapper for async scraping job
        APScheduler needs synchronous functions, so we use asyncio.run()
        """
        try:
            logger.info("🔄 Sync wrapper executing async scraping job...")
            # Run the async function in a new event loop
            result = asyncio.run(self.run_scraping_job())
            logger.info(f"✅ Sync wrapper completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Sync wrapper failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def start(self):
        """Start the scheduler"""
        try:
            # ✅ Use synchronous wrapper instead of lambda
            self.scheduler.add_job(
                func=self._run_scraping_job_sync,  # ✅ Call sync wrapper
                trigger="interval",
                hours=12,
                id='scraping_job',
                name='AI News Scraping',
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info(f"✅ Scheduler started - scraping will run every 12 hours using {self.default_llm_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {str(e)}")
    
    def stop(self):
        """Stop the scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("⏹️ Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Failed to stop scheduler: {str(e)}")
    
    # ✅ UPDATED: Manual trigger method for testing
    def trigger_now(self):
        """
        Manually trigger the scraping job immediately (for testing/admin use)
        This bypasses the schedule and runs the job right away
        """
        try:
            logger.info("🔧 Manual scraping job trigger requested")
            # ✅ Use synchronous wrapper instead of lambda
            self.scheduler.add_job(
                func=self._run_scraping_job_sync,  # ✅ Call sync wrapper
                trigger="date",  # Run once, immediately
                id='manual_scraping_job',
                name='Manual AI News Scraping',
                replace_existing=True
            )
            logger.info("✅ Manual scraping job scheduled for immediate execution")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to trigger manual scraping: {str(e)}")
            return False


# ✅ CREATE GLOBAL SINGLETON INSTANCE
scheduler_service = SchedulerService()

# ✅ START SCHEDULER AUTOMATICALLY WHEN MODULE IS IMPORTED
logger.info("🚀 Starting scheduler service...")
scheduler_service.start()