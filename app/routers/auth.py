#!/usr/bin/env python3
"""
Auth router for modular FastAPI architecture
Handles user authentication, registration, and OAuth
"""

import os
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import UserResponse, UserCreate, UserSignIn
from app.dependencies.auth import AuthHandler

router = APIRouter()
logger = logging.getLogger(__name__)

# Get DEBUG mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if DEBUG:
    logger.debug("🔍 Auth router initialized in DEBUG mode")

auth_handler = AuthHandler()

def is_gmail_email(email: str) -> bool:
    """
    Check if email is a Gmail/Google account
    Gmail users should use OAuth instead of email/password
    """
    normalized_email = email.lower().strip()
    return normalized_email.endswith('@gmail.com') or normalized_email.endswith('@googlemail.com')

@router.post("/signup", response_model=UserResponse)
async def signup(user_data: UserCreate):
    """
    User registration endpoint - email/password signup
    Gmail users are redirected to use Google OAuth instead
    """
    try:
        logger.info(f"📝 Signup request for: {user_data.email}")
        
        # ✅ NEW: Check if Gmail user trying to sign up with password
        if is_gmail_email(user_data.email):
            logger.warning(f"⚠️ Gmail user {user_data.email} attempted password signup")
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Gmail users must use Google Sign-In',
                    'message': 'Please click "Continue with Google" to sign up with your Gmail account.',
                    'action': 'use_google_signin'
                }
            )
        
        # Check if user already exists
        existing_user = auth_handler.get_user_by_email(user_data.email)
        if existing_user:
            logger.warning(f"⚠️ Signup failed - User already exists: {user_data.email}")
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'User already exists',
                    'message': 'An account with this email already exists. Please sign in.',
                    'action': 'signin'
                }
            )
        
        # Hash password and create user
        hashed_password = auth_handler.get_password_hash(user_data.password)
        user = auth_handler.create_user(
            email=user_data.email,
            password=hashed_password,
            name=user_data.name
        )
        
        logger.info(f"✅ User registered successfully: {user.email}")
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Signup failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Signup failed',
                'message': str(e)
            }
        )

@router.post("/signin", response_model=UserResponse)
async def signin(user_data: UserSignIn):
    """
    User signin endpoint - email/password authentication
    Gmail users are redirected to use Google OAuth instead
    """
    try:
        logger.info(f"🔐 Signin request for: {user_data.email}")
        
        # ✅ NEW: Check if Gmail user trying to sign in with password
        if is_gmail_email(user_data.email):
            logger.warning(f"⚠️ Gmail user {user_data.email} attempted password signin")
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Gmail users must use Google Sign-In',
                    'message': 'Please click "Continue with Google" to sign in with your Gmail account.',
                    'action': 'use_google_signin'
                }
            )
        
        # Authenticate user
        user = auth_handler.authenticate_user(user_data.email, user_data.password)
        if not user:
            logger.warning(f"❌ Invalid credentials for user: {user_data.email}")
            raise HTTPException(
                status_code=401,
                detail={
                    'error': 'Invalid credentials',
                    'message': 'Incorrect email or password'
                }
            )
        
        logger.info(f"✅ User signed in successfully: {user.email}")
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Signin failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Signin failed',
                'message': str(e)
            }
        )

# Google OAuth endpoints
@router.get("/google/login")
async def google_login():
    """
    Redirect to Google for OAuth login
    """
    try:
        logger.info("🔗 Redirecting to Google for OAuth login")
        redirect_uri = auth_handler.get_google_auth_url()
        return {"redirect_url": redirect_uri}
    except Exception as e:
        logger.error(f"❌ Google login failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Google login failed',
                'message': str(e)
            }
        )

@router.get("/google/callback")
async def google_callback(code: str):
    """
    Google OAuth callback endpoint
    Exchanges code for access token and retrieves user info
    """
    try:
        logger.info(f"🔄 Google callback received with code: {code}")
        
        # Exchange code for tokens
        tokens = auth_handler.get_google_tokens(code)
        if not tokens:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Invalid code',
                    'message': 'Failed to exchange code for tokens'
                }
            )
        
        # Get user info from Google
        user_info = auth_handler.get_google_user_info(tokens['access_token'])
        if not user_info:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Invalid token',
                    'message': 'Failed to retrieve user info from Google'
                }
            )
        
        logger.info(f"✅ Google user info retrieved: {user_info['email']}")
        
        # Check if user already exists
        user = auth_handler.get_user_by_email(user_info['email'])
        if not user:
            # Register new user
            logger.info(f"📝 Registering new user from Google: {user_info['email']}")
            user = auth_handler.create_user(
                email=user_info['email'],
                password=None,  # No password for OAuth users
                name=user_info.get('name')
            )
        
        # Generate JWT token
        token = auth_handler.create_token(user.id)
        
        logger.info(f"✅ User signed in with Google: {user.email}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google callback handling failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Google callback handling failed',
                'message': str(e)
            }
        )

# Password recovery endpoints
@router.post("/forgot-password")
async def forgot_password(email: str):
    """
    Password recovery - sends email with reset link
    """
    try:
        logger.info(f"🔑 Password recovery requested for: {email}")
        
        # Check if user exists
        user = auth_handler.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'User not found',
                    'message': 'No user found with this email address'
                }
            )
        
        # Generate password reset token
        token = auth_handler.create_password_reset_token(user.id)
        
        # Send email with reset link
        auth_handler.send_password_reset_email(user.email, token)
        
        logger.info(f"✅ Password recovery email sent to: {user.email}")
        return {"message": "Password recovery email sent"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Password recovery failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Password recovery failed',
                'message': str(e)
            }
        )

@router.post("/reset-password")
async def reset_password(token: str, new_password: str):
    """
    Reset password - updates user password
    """
    try:
        logger.info(f"🔑 Password reset requested with token: {token}")
        
        # Decode token to get user ID
        user_id = auth_handler.decode_password_reset_token(token)
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Invalid token',
                    'message': 'Failed to decode password reset token'
                }
            )
        
        # Update user password
        auth_handler.update_user_password(user_id, new_password)
        
        logger.info(f"✅ Password updated successfully for user ID: {user_id}")
        return {"message": "Password updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Password reset failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Password reset failed',
                'message': str(e)
            }
        )

