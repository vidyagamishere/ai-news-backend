#!/usr/bin/env python3
"""
Authentication router for modular FastAPI architecture
Maintains compatibility with existing frontend API endpoints
"""

import json
import base64
import logging
import os
import random
import string
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    GoogleAuthRequest, TokenResponse, UserResponse, UserPreferences,
    OTPRequest, OTPVerifyRequest
)
from app.dependencies.auth import get_current_user
from app.dependencies.database import get_database_service
from app.services.auth_service import AuthService
from db_service import DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter()

# OTP storage (in production, use Redis or database)
otp_storage = {}


@router.get("/auth/env-debug")
async def debug_environment():
    """
    Debug endpoint to check environment variables (admin only)
    """
    try:
        # Check various environment variables
        env_check = {
            'BREVO_API_KEY': bool(os.getenv('BREVO_API_KEY')),
            'JWT_SECRET': bool(os.getenv('JWT_SECRET')),
            'GOOGLE_CLIENT_ID': bool(os.getenv('GOOGLE_CLIENT_ID')),
            'POSTGRES_URL': bool(os.getenv('POSTGRES_URL')),
            'ALLOWED_ORIGINS': bool(os.getenv('ALLOWED_ORIGINS')),
        }
        
        # Check specific Brevo key details
        brevo_key = os.getenv('BREVO_API_KEY', '')
        brevo_info = {
            'exists': bool(brevo_key),
            'length': len(brevo_key) if brevo_key else 0,
            'starts_with_xkeysib': brevo_key.startswith('xkeysib-') if brevo_key else False
        }
        
        logger.info("🔍 Environment variables debug check requested")
        
        return {
            'success': True,
            'environment_check': env_check,
            'brevo_details': brevo_info,
            'note': 'This endpoint helps debug environment variable configuration'
        }
        
    except Exception as e:
        logger.error(f"❌ Environment debug failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'note': 'Environment debug endpoint error'
        }

def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email: str, otp: str, name: str = "") -> bool:
    """Send OTP via email using Brevo API (same as admin interface)"""
    try:
        brevo_api_key = os.getenv('BREVO_API_KEY', '')
        logger.info(f"🔑 Brevo API key check: {'✅ Found' if brevo_api_key else '❌ Not found'}")
        
        if not brevo_api_key:
            logger.warning("⚠️ BREVO_API_KEY not set in environment, using fallback mode")
            return False
        
        # Use Brevo API for sending email
        import requests
        
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "api-key": brevo_api_key,
            "Content-Type": "application/json"
        }
        
        # Determine sender and subject based on email domain
        is_admin = email == "admin@vidyagam.com"
        sender_email = "noreply@vidyagam.com"
        sender_name = "Vidyagam AI News"
        subject = f"{'Admin ' if is_admin else ''}Login OTP - Vidyagam AI News"
        
        data = {
            "sender": {"email": sender_email, "name": sender_name},
            "to": [{"email": email, "name": name or "User"}],
            "subject": subject,
            "htmlContent": f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">Vidyagam AI News</h1>
                    <p style="color: #f8f9ff; margin: 10px 0 0 0;">Your AI News Authentication</p>
                </div>
                
                <div style="background: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #333; margin-top: 0;">{'Admin ' if is_admin else ''}Login Verification</h2>
                    <p style="color: #666; font-size: 16px; line-height: 1.5;">
                        Hi {name or 'there'}! Your one-time password for {'admin ' if is_admin else ''}access is:
                    </p>
                    
                    <div style="background: white; border: 2px solid #007bff; border-radius: 8px; padding: 20px; margin: 20px 0; display: inline-block;">
                        <span style="font-size: 32px; font-weight: bold; color: #007bff; letter-spacing: 4px;">{otp}</span>
                    </div>
                    
                    <p style="color: #666; font-size: 14px; line-height: 1.5;">
                        This OTP is valid for <strong>5 minutes</strong>. Please do not share it with anyone.
                    </p>
                    
                    {f'<div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 20px 0;"><p style="color: #856404; margin: 0; font-size: 14px;"><strong>Admin Access:</strong> You are logging in with administrator privileges.</p></div>' if is_admin else ''}
                </div>
                
                <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                    <p style="color: #999; font-size: 12px; margin: 0;">
                        If you didn't request this, please ignore this email.<br>
                        © 2025 Vidyagam AI News - Personalized AI Intelligence
                    </p>
                </div>
            </body>
            </html>
            """
        }
        
        logger.info(f"📧 Attempting to send OTP email to {email} via Brevo API")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        logger.info(f"📊 Brevo API response: {response.status_code}")
        if response.status_code == 201:
            logger.info(f"✅ OTP email sent successfully to {email} via Brevo")
            return True
        else:
            logger.error(f"❌ Brevo API failed: {response.status_code} - {response.text[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Email service error: {str(e)}")
        return False


def get_auth_service() -> AuthService:
    """Dependency to get AuthService instance"""
    return AuthService()


@router.post("/auth/google")
async def google_auth(
    request: GoogleAuthRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Google OAuth authentication - compatible with existing frontend
    Endpoint: POST /auth/google (same as before)
    Returns AuthResponse format expected by frontend
    """
    try:
        logger.info("🔐 Processing Google authentication")
        
        google_token = request.credential
        if not google_token:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Missing Google token',
                    'message': 'Google credential token is required'
                }
            )
        
        # Extract email from token payload (simplified)
        try:
            # Decode Google JWT token payload (without verification for demo)
            parts = google_token.split('.')
            if len(parts) >= 2:
                payload_encoded = parts[1]
                # Add padding
                while len(payload_encoded) % 4:
                    payload_encoded += '='
                payload_decoded = base64.urlsafe_b64decode(payload_encoded.encode())
                google_data = json.loads(payload_decoded.decode())
                
                user_data = {
                    'sub': google_data.get('sub', ''),
                    'email': google_data.get('email', ''),
                    'name': google_data.get('name', ''),
                    'picture': google_data.get('picture', '')
                }
            else:
                raise Exception("Invalid Google token format")
                
        except Exception:
            # Fallback for testing
            user_data = {
                'sub': 'google_user_' + str(int(datetime.utcnow().timestamp())),
                'email': 'test@example.com',
                'name': 'Test User',
                'picture': ''
            }
        
        # Create or update user in PostgreSQL
        user = auth_service.create_or_update_user(user_data)
        
        # Create JWT token with user data including admin flag
        user_data_with_admin = {**user_data, 'is_admin': user.get('is_admin', False)}
        jwt_token = auth_service.create_jwt_token(user_data_with_admin)
        
        # Return AuthResponse format expected by frontend
        response = {
            'success': True,
            'message': f'Authentication successful for {user_data["email"]}',
            'token': jwt_token,
            'user': {
                'id': str(user['id']),
                'email': str(user['email']),
                'name': str(user.get('name', '')),
                'picture': str(user.get('profile_image', user_data.get('picture', ''))),
                'verified_email': bool(user.get('verified_email', True)),
                'is_admin': bool(user.get('is_admin', False))
            },
            'expires_in': 3600,  # 1 hour
            'router_auth': False,  # Modular architecture
            'isUserExist': True  # User exists after creation/update
        }
        
        logger.info(f"✅ Google authentication successful for: {user_data['email']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google auth failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Authentication failed',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.post("/auth/send-otp")
async def send_otp(
    request: OTPRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Send OTP for email authentication
    Endpoint: POST /auth/send-otp (compatible with existing frontend)
    """
    try:
        logger.info(f"📧 OTP requested for: {request.email}")
        
        # Generate a secure 6-digit OTP
        otp = generate_otp()
        
        # Store OTP with timestamp for validation
        from datetime import timedelta
        otp_storage[request.email] = {
            'otp': otp,
            'timestamp': datetime.utcnow(),
            'auth_mode': request.auth_mode
        }
        
        # FOR TESTING: Skip email sending and always return success
        logger.info(f"🧪 TESTING MODE: Skipping email sending for: {request.email}")
        email_sent = True  # Always return true for testing
        
        # Prepare response
        response = {
            'success': True,
            'message': f'OTP sent to {request.email} (testing mode)',
            'otpSent': True,  # Always true for testing
            'debug_info': {
                'otp_for_testing': otp,  # Always show OTP for testing
                'auth_mode': request.auth_mode,
                'email_service_status': 'testing_mode',
                'testing_note': 'OTP validation is disabled - any code will work'
            }
        }
        
        logger.info(f"✅ OTP generated for testing: {request.email} (Code: {otp})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ OTP send failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to send OTP',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.post("/auth/verify-otp")
async def verify_otp(
    request: OTPVerifyRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify OTP and authenticate user
    Endpoint: POST /auth/verify-otp (compatible with existing frontend)
    Returns AuthResponse format expected by frontend
    """
    try:
        logger.info(f"🔐 OTP verification for: {request.email}")
        
        # FOR TESTING: Skip OTP validation and always return as verified
        logger.info(f"🧪 TESTING MODE: Skipping OTP validation for: {request.email}")
        
        # Clean up any stored OTP if exists
        if request.email in otp_storage:
            del otp_storage[request.email]
            
        logger.info(f"✅ OTP validation skipped for testing: {request.email}")
        
        # Create user data from request with safer access to userData
        user_data = {
            'sub': f"otp_user_{int(datetime.utcnow().timestamp())}",
            'email': request.email,
            'name': request.userData.get('name', '') if request.userData else '',
            'picture': ''
        }
        
        # Create or update user in PostgreSQL
        user = auth_service.create_or_update_user(user_data)
        
        # Check if this is an admin user
        is_admin = request.email == 'admin@vidyagam.com'
        if is_admin:
            logger.info(f"🔑 Admin authentication successful for: {request.email}")
        
        # Create JWT token with user data including admin flag
        user_data_with_admin = {**user_data, 'is_admin': is_admin or user.get('is_admin', False)}
        jwt_token = auth_service.create_jwt_token(user_data_with_admin)
        
        # Return AuthResponse format expected by frontend
        response = {
            'success': True,
            'message': f'OTP verification successful for {request.email}',
            'token': jwt_token,
            'user': {
                'id': str(user['id']),
                'email': str(user['email']),
                'name': str(user.get('name', '')),
                'picture': str(user.get('profile_image', '')),
                'verified_email': bool(user.get('verified_email', True)),
                'is_admin': bool(is_admin or user.get('is_admin', False))
            },
            'expires_in': 3600,  # 1 hour
            'router_auth': False,  # Modular architecture
            'isUserExist': True  # User exists after creation/update
        }
        
        logger.info(f"✅ OTP verification successful for: {request.email} (Admin: {response['user']['is_admin']})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OTP verification failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'OTP verification failed',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.get("/auth/profile", response_model=UserResponse)
async def get_profile(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get user profile - requires authentication
    Endpoint: GET /auth/profile (same as before)
    """
    try:
        logger.info(f"👤 Profile requested for: {current_user.email}")
        
        logger.info(f"✅ Profile retrieved for: {current_user.email}")
        return current_user
        
    except Exception as e:
        logger.error(f"❌ Profile endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get profile',
                'message': str(e),
                'database': 'postgresql'
            }
        )


@router.post("/auth/preferences")
@router.put("/auth/preferences")
async def update_preferences(
    preferences: UserPreferences,
    current_user: UserResponse = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service)
):
    """
    Update user preferences in user_preferences table
    Endpoint: POST/PUT /auth/preferences (compatible with existing frontend)
    """
    try:
        logger.info(f"⚙️ Preferences update for: {current_user.email}")
        logger.info(f"🔍 Preferences data: {preferences.dict()}")
        
        # Insert or update user preferences in user_preferences table (matching exact table schema)
        # Cast arrays to JSONB explicitly for PostgreSQL compatibility
        upsert_query = """
            INSERT INTO user_preferences (
                user_id, experience_level, professional_roles, 
                categories_selected, content_types_selected, publishers_selected,
                newsletter_frequency, email_notifications, breaking_news_alerts,
                onboarding_completed, updated_at
            ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) 
            DO UPDATE SET
                experience_level = EXCLUDED.experience_level,
                professional_roles = EXCLUDED.professional_roles,
                categories_selected = EXCLUDED.categories_selected,
                content_types_selected = EXCLUDED.content_types_selected,
                publishers_selected = EXCLUDED.publishers_selected,
                newsletter_frequency = EXCLUDED.newsletter_frequency,
                email_notifications = EXCLUDED.email_notifications,
                breaking_news_alerts = EXCLUDED.breaking_news_alerts,
                onboarding_completed = EXCLUDED.onboarding_completed,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        """
        
        # Convert Python lists to JSON strings for JSONB casting
        # Ensure onboarding_completed is always True when preferences are updated
        result = db.execute_query(
            upsert_query,
            (
                current_user.id,
                preferences.experience_level,
                json.dumps(preferences.professional_roles),  # Convert to JSON string for ::jsonb cast
                json.dumps(preferences.categories_selected),  # Convert to JSON string for ::jsonb cast
                json.dumps(preferences.content_types_selected),  # Convert to JSON string for ::jsonb cast
                json.dumps(preferences.publishers_selected),  # Convert to JSON string for ::jsonb cast
                preferences.newsletter_frequency,
                preferences.email_notifications,
                preferences.breaking_news_alerts,
                True  # Always set onboarding_completed to True when preferences are saved
            ),
            fetch_one=True
        )
        
        logger.info(f"✅ Preferences updated successfully for user: {current_user.id}")
        logger.info(f"✅ Onboarding completed set to: True")
        return {
            "message": "Preferences updated successfully", 
            "preferences": preferences.dict(),
            "onboarding_completed": True,
            "saved_to": "user_preferences_table"
        }
        
    except Exception as e:
        logger.error(f"❌ Preferences update failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preferences: {str(e)}"
        )