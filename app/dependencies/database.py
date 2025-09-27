#!/usr/bin/env python3
"""
Database dependencies for FastAPI dependency injection
"""

from db_service import get_database_service, PostgreSQLService


def get_db() -> PostgreSQLService:
    """Dependency to get database service instance"""
    return get_database_service()