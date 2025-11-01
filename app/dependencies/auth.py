#!/usr/bin/env python3
"""
Authentication dependencies for FastAPI dependency injection
"""

import os
from datetime import datetime, timedelta
import logging
from typing import Optional
from datetime import datetime
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.services.auth_service import AuthService
from app.models.schemas import UserResponse

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# ...existing code...

class AuthHandler:
    """Handler for authentication operations"""
    
    def __init__(self):
        self.secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60 * 24 * 7  # 7 days
    
    def create_access_token(self, user_id: str, email: str) -> str:
        """Create JWT access token"""
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> dict:
        """Decode JWT token"""
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)

# ...existing code...


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
            # Use data from JWT token directly, including admin flag
            is_admin = user_data.get('is_admin', False)
            email = user_data.get('email', '')
            
            logger.info(f"🔐 Token verified for: {email} (Admin: {is_admin})")
            
            user_response_data = {
                'id': user_data.get('sub', ''),
                'email': email,
                'name': user_data.get('name', ''),
                'profile_image': user_data.get('picture', ''),
                'subscription_tier': 'free',
                'preferences': {},
                'created_at': datetime.utcnow(),
                'verified_email': True,
                'is_admin': is_admin
            }
            return UserResponse(**user_response_data)
        
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
        
        # Use data from JWT token directly, including admin flag
        is_admin = user_data.get('is_admin', False)
        email = user_data.get('email', '')
        
        logger.info(f"🔐 Token verified (required) for: {email} (Admin: {is_admin})")
        
        user_response_data = {
            'id': user_data.get('sub', ''),
            'email': email,
            'name': user_data.get('name', ''),
            'profile_image': user_data.get('picture', ''),
            'subscription_tier': 'free',
            'preferences': {},
            'created_at': datetime.utcnow(),
            'verified_email': True,
            'is_admin': is_admin
        }
        
        return UserResponse(**user_response_data)
        
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