#!/usr/bin/env python3
"""
Authentication dependencies for FastAPI dependency injection
"""

from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.auth_service import AuthService
from app.models.schemas import UserResponse

# Security scheme
security = HTTPBearer(auto_error=False)


def get_auth_service() -> AuthService:
    """Dependency to get AuthService instance"""
    return AuthService()


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[UserResponse]:
    """Get current user from token (optional - returns None if no token)"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_data = auth_service.verify_jwt_token(token)
        
        if user_data:
            user = auth_service.get_user_by_email(user_data['email'])
            if user:
                return UserResponse(**user)
        
        return None
        
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """Get current user from token (required - raises 401 if no valid token)"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token = credentials.credentials
        user_data = auth_service.verify_jwt_token(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = auth_service.get_user_by_email(user_data['email'])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return UserResponse(**user)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_admin_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get current user and verify admin privileges"""
    # Add admin check logic here if needed
    # For now, any authenticated user can access admin endpoints
    return current_user