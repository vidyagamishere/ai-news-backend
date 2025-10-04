#!/usr/bin/env python3
"""
Enhanced Authentication router for new mobile-first interface
Supports Google One Tap and password-based authentication (no OTP)
"""

import json
import base64
import logging
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.models.schemas import (
    GoogleAuthRequest, PasswordAuthRequest, UserSignupRequest, 
    TokenResponse, UserResponse, UserPreferences
)
from app.dependencies.database import get_database_service
from db_service import DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

JWT_SECRET = os.getenv('JWT_SECRET_KEY', 'vidyagam-jwt-secret-2025')
JWT_ALGORITHM = 'HS256'
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hashed version"""
    try:
        salt, password_hash = hashed_password.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False


def create_jwt_token(user_data: dict) -> str:
    """Create JWT token"""
    payload = {
        'sub': user_data.get('id'),
        'email': user_data.get('email'),
        'name': user_data.get('name'),
        'is_admin': user_data.get('is_admin', False),
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(days=7)  # 7 days expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def verify_google_token(credential: str) -> dict:
    """Verify Google ID token and extract user info"""
    try:
        # For production, you should verify the token with Google's API
        # For development, we'll decode it directly (NOT SECURE FOR PRODUCTION)
        
        # Split the JWT token
        header, payload, signature = credential.split('.')
        
        # Add padding if necessary
        payload += '=' * (4 - len(payload) % 4)
        
        # Decode the payload
        decoded_payload = base64.urlsafe_b64decode(payload)
        user_info = json.loads(decoded_payload)
        
        # Verify the audience (client_id)
        if GOOGLE_CLIENT_ID and user_info.get('aud') != GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Invalid audience")
        
        return {
            'sub': user_info.get('sub'),
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'picture': user_info.get('picture'),
            'email_verified': user_info.get('email_verified', False)
        }
    except Exception as e:
        logger.error(f"Google token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Google token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseService = Depends(get_database_service)
) -> dict:
    """Get current authenticated user"""
    try:
        payload = verify_jwt_token(credentials.credentials)
        user_id = payload.get('sub')
        
        # Get user from database
        user_query = """
            SELECT id, email, name, profile_image, verified_email, is_admin, 
                   subscription_tier, created_at, preferences
            FROM users WHERE id = %s
        """
        user = db.execute_query(user_query, (user_id,), fetch_one=True)
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return dict(user)
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.post("/auth/google", response_model=TokenResponse)
async def google_auth(
    request: GoogleAuthRequest,
    db: DatabaseService = Depends(get_database_service)
):
    """Authenticate with Google One Tap"""
    try:
        # Verify Google token
        google_user = verify_google_token(request.credential)
        
        # Check if user exists
        user_query = "SELECT * FROM users WHERE email = %s"
        existing_user = db.execute_query(user_query, (google_user['email'],), fetch_one=True)
        
        if existing_user:
            # Update existing user
            update_query = """
                UPDATE users SET 
                    name = %s, 
                    profile_image = %s, 
                    verified_email = %s,
                    last_login = CURRENT_TIMESTAMP
                WHERE email = %s
                RETURNING *
            """
            user = db.execute_query(
                update_query, 
                (google_user['name'], google_user['picture'], True, google_user['email']),
                fetch_one=True
            )
        else:
            # Create new user
            insert_query = """
                INSERT INTO users (
                    email, name, profile_image, verified_email, 
                    subscription_tier, created_at, last_login
                ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING *
            """
            user = db.execute_query(
                insert_query,
                (google_user['email'], google_user['name'], google_user['picture'], True, 'free'),
                fetch_one=True
            )
        
        # Create JWT token
        token = create_jwt_token(dict(user))
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            profile_image=user.get('profile_image'),
            subscription_tier=user.get('subscription_tier', 'free'),
            preferences=user.get('preferences', {}),
            created_at=user['created_at'],
            verified_email=user.get('verified_email', False),
            is_admin=user.get('is_admin', False)
        )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=user_response
        )
    
    except Exception as e:
        logger.error(f"Google authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Google authentication failed")


@router.post("/auth/signup", response_model=TokenResponse)
async def signup(
    request: UserSignupRequest,
    db: DatabaseService = Depends(get_database_service)
):
    """Sign up with email and password"""
    try:
        # Validate password confirmation
        if request.password != request.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # Check if user already exists
        existing_user = db.execute_query(
            "SELECT id FROM users WHERE email = %s", 
            (request.email,), 
            fetch_one=True
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Create new user
        insert_query = """
            INSERT INTO users (
                email, name, password_hash, verified_email, 
                subscription_tier, created_at, last_login
            ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING *
        """
        user = db.execute_query(
            insert_query,
            (request.email, request.name, hashed_password, False, 'free'),
            fetch_one=True
        )
        
        # Create JWT token
        token = create_jwt_token(dict(user))
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            profile_image=user.get('profile_image'),
            subscription_tier=user.get('subscription_tier', 'free'),
            preferences=user.get('preferences', {}),
            created_at=user['created_at'],
            verified_email=user.get('verified_email', False),
            is_admin=user.get('is_admin', False)
        )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=user_response
        )
    
    except Exception as e:
        logger.error(f"Signup failed: {str(e)}")
        if "User already exists" in str(e):
            raise e
        raise HTTPException(status_code=500, detail="Signup failed")


@router.post("/auth/signin", response_model=TokenResponse)
async def signin(
    request: PasswordAuthRequest,
    db: DatabaseService = Depends(get_database_service)
):
    """Sign in with email and password"""
    try:
        # Get user from database
        user_query = "SELECT * FROM users WHERE email = %s"
        user = db.execute_query(user_query, (request.email,), fetch_one=True)
        
        if not user or not user.get('password_hash'):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not verify_password(request.password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        db.execute_query(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
            (user['id'],),
            fetch_all=False
        )
        
        # Create JWT token
        token = create_jwt_token(dict(user))
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            profile_image=user.get('profile_image'),
            subscription_tier=user.get('subscription_tier', 'free'),
            preferences=user.get('preferences', {}),
            created_at=user['created_at'],
            verified_email=user.get('verified_email', False),
            is_admin=user.get('is_admin', False)
        )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=user_response
        )
    
    except Exception as e:
        logger.error(f"Signin failed: {str(e)}")
        if "Invalid credentials" in str(e):
            raise e
        raise HTTPException(status_code=500, detail="Signin failed")


@router.get("/auth/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return UserResponse(
        id=str(current_user['id']),
        email=current_user['email'],
        name=current_user['name'],
        profile_image=current_user.get('profile_image'),
        subscription_tier=current_user.get('subscription_tier', 'free'),
        preferences=current_user.get('preferences', {}),
        created_at=current_user['created_at'],
        verified_email=current_user.get('verified_email', False),
        is_admin=current_user.get('is_admin', False)
    )


@router.put("/auth/preferences")
async def update_preferences(
    preferences: UserPreferences,
    current_user: dict = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service)
):
    """Update user preferences"""
    try:
        # Convert preferences to JSON
        preferences_json = preferences.dict(exclude_unset=True)
        
        # Update user preferences in database
        update_query = """
            UPDATE users SET 
                preferences = %s,
                name = COALESCE(%s, name)
            WHERE id = %s
        """
        db.execute_query(
            update_query,
            (json.dumps(preferences_json), preferences.name, current_user['id']),
            fetch_all=False
        )
        
        return {"message": "Preferences updated successfully", "preferences": preferences_json}
    
    except Exception as e:
        logger.error(f"Failed to update preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should delete token)"""
    return {"message": "Logged out successfully"}