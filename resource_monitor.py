#!/usr/bin/env python3
"""
Resource Monitor for scraping operations
Monitors CPU, memory, and GPU usage to prevent system overload
"""

import psutil
import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResourceThresholds:
    """Resource usage thresholds"""
    max_cpu_percent: float = 80.0  # Max CPU usage %
    max_memory_percent: float = 75.0  # Max memory usage %
    max_memory_mb: int = 6000  # Max memory in MB (6GB for safety on 8GB systems)
    warning_memory_percent: float = 60.0  # Warning threshold
    critical_memory_percent: float = 70.0  # Critical threshold
    # ABORT thresholds - stop scraping immediately
    abort_cpu_percent: float = 90.0  # Abort if CPU exceeds this
    abort_memory_percent: float = 85.0  # Abort if memory exceeds this
    abort_memory_mb: int = 7000  # Abort if process memory exceeds this (7GB)

@dataclass
class ResourceUsage:
    """Current resource usage stats"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    available_memory_mb: float
    is_safe: bool
    should_abort: bool  # True if resources are critically high - STOP immediately
    warnings: list

class ResourceMonitor:
    """Monitor system resources and control execution"""
    
    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.thresholds = thresholds or ResourceThresholds()
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss / 1024 / 1024
        
        logger.info(f"📊 Resource Monitor initialized:")
        logger.info(f"   Max CPU: {self.thresholds.max_cpu_percent}%")
        logger.info(f"   Max Memory: {self.thresholds.max_memory_percent}% ({self.thresholds.max_memory_mb}MB)")
        logger.info(f"   Initial Process Memory: {self._initial_memory:.1f}MB")
    
    def get_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process_memory_mb = self._process.memory_info().rss / 1024 / 1024
            
            warnings = []
            
            # Check ABORT thresholds first (critical - must stop immediately)
            should_abort = (
                cpu_percent > self.thresholds.abort_cpu_percent or
                memory.percent > self.thresholds.abort_memory_percent or
                process_memory_mb > self.thresholds.abort_memory_mb
            )
            
            if should_abort:
                warnings.append(f"🚨 ABORT THRESHOLD EXCEEDED - STOPPING SCRAPING 🚨")
                if cpu_percent > self.thresholds.abort_cpu_percent:
                    warnings.append(f"CPU: {cpu_percent:.1f}% > {self.thresholds.abort_cpu_percent}%")
                if memory.percent > self.thresholds.abort_memory_percent:
                    warnings.append(f"Memory: {memory.percent:.1f}% > {self.thresholds.abort_memory_percent}%")
                if process_memory_mb > self.thresholds.abort_memory_mb:
                    warnings.append(f"Process Memory: {process_memory_mb:.1f}MB > {self.thresholds.abort_memory_mb}MB")
            
            # Check CPU
            if cpu_percent > self.thresholds.max_cpu_percent:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check memory
            if memory.percent > self.thresholds.critical_memory_percent:
                warnings.append(f"CRITICAL memory usage: {memory.percent:.1f}%")
            elif memory.percent > self.thresholds.warning_memory_percent:
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            
            if process_memory_mb > self.thresholds.max_memory_mb:
                warnings.append(f"Process memory too high: {process_memory_mb:.1f}MB")
            
            is_safe = (
                cpu_percent < self.thresholds.max_cpu_percent and
                memory.percent < self.thresholds.max_memory_percent and
                process_memory_mb < self.thresholds.max_memory_mb
            )
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=process_memory_mb,
                available_memory_mb=memory.available / 1024 / 1024,
                is_safe=is_safe,
                should_abort=should_abort,
                warnings=warnings
            )
        
        except Exception as e:
            logger.error(f"❌ Error getting resource usage: {e}")
            # Return safe defaults on error
            return ResourceUsage(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                available_memory_mb=0.0,
                is_safe=True,
                should_abort=False,
                warnings=[]
            )
    
    def should_throttle(self) -> bool:
        """Check if we should throttle operations"""
        usage = self.get_usage()
        return not usage.is_safe
    
    def should_abort(self) -> bool:
        """Check if we should abort/stop operations immediately"""
        usage = self.get_usage()
        if usage.should_abort:
            logger.error("🚨 RESOURCE ABORT THRESHOLD EXCEEDED - STOPPING IMMEDIATELY 🚨")
            for warning in usage.warnings:
                logger.error(f"  {warning}")
        return usage.should_abort
    
    def get_max_concurrent_tasks(self) -> int:
        """Calculate maximum concurrent tasks based on current resources"""
        usage = self.get_usage()
        
        # Base on available memory
        available_gb = usage.available_memory_mb / 1024
        
        if available_gb > 4:
            return 5  # Plenty of memory
        elif available_gb > 2:
            return 3  # Moderate memory
        elif available_gb > 1:
            return 2  # Low memory
        else:
            return 1  # Critical - one at a time
    
    async def wait_for_resources(self, timeout: int = 300):
        """Wait until resources are available (with timeout)"""
        start_time = asyncio.get_event_loop().time()
        check_interval = 5  # Check every 5 seconds
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.error(f"⏰ Resource wait timeout after {timeout}s")
                raise TimeoutError(f"Resources did not become available within {timeout}s")
            
            usage = self.get_usage()
            
            if usage.is_safe:
                logger.info(f"✅ Resources available - proceeding")
                return
            
            # Log warnings
            for warning in usage.warnings:
                logger.warning(f"⚠️ {warning}")
            
            logger.warning(
                f"⏳ Waiting for resources... "
                f"CPU: {usage.cpu_percent:.1f}%, "
                f"Memory: {usage.memory_percent:.1f}% ({usage.memory_mb:.1f}MB), "
                f"Available: {usage.available_memory_mb:.1f}MB"
            )
            
            await asyncio.sleep(check_interval)
    
    def log_status(self):
        """Log current resource status"""
        usage = self.get_usage()
        memory_delta = usage.memory_mb - self._initial_memory
        
        logger.info(
            f"📊 Resources: "
            f"CPU: {usage.cpu_percent:.1f}%, "
            f"Memory: {usage.memory_percent:.1f}% "
            f"({usage.memory_mb:.1f}MB, +{memory_delta:.1f}MB from start), "
            f"Available: {usage.available_memory_mb:.1f}MB"
        )
        
        for warning in usage.warnings:
            logger.warning(f"⚠️ {warning}")
    
    async def monitor_during_execution(self, check_interval: int = 10):
        """Continuously monitor resources during execution"""
        while True:
            await asyncio.sleep(check_interval)
            self.log_status()
            
            usage = self.get_usage()
            if not usage.is_safe:
                logger.error(f"🚨 Resource limit exceeded - scraping should pause!")
                for warning in usage.warnings:
                    logger.error(f"  {warning}")
