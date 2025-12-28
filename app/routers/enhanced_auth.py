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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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


def send_verification_email_smtp(to_email: str, name: str, verification_url: str):
    """
    Send verification email using Gmail SMTP
    """
    smtp_user = os.getenv('SMTP_USER', 'admin@vidyagam.com')
    smtp_password = os.getenv('SMTP_PASSWORD')
    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    
    if not smtp_password:
        logger.warning("⚠️ SMTP_PASSWORD not configured, email not sent")
        raise Exception("SMTP not configured. Please set SMTP_PASSWORD in environment variables.")
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Verify your email - Vidyagam'
    msg['From'] = f'Vidyagam <{smtp_user}>'
    msg['To'] = to_email
    
    # HTML email template
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f5f5f5; padding: 20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: bold;">Vidyagam</h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 16px;">AI Knowledge Hub</p>
                        </td>
                    </tr>
                    <!-- Content -->
                    <tr>
                        <td style="padding: 40px 30px;">
                            <h2 style="margin: 0 0 20px 0; color: #333333; font-size: 24px;">Welcome{f', {name}' if name else ''}! 👋</h2>
                            <p style="margin: 0 0 20px 0; color: #666666; font-size: 16px; line-height: 1.5;">
                                Thank you for signing up for Vidyagam! We're excited to have you on board.
                            </p>
                            <p style="margin: 0 0 30px 0; color: #666666; font-size: 16px; line-height: 1.5;">
                                Please verify your email address by clicking the button below:
                            </p>
                            <!-- Button -->
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td align="center" style="padding: 20px 0;">
                                        <a href="{verification_url}" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: bold;">Verify Email Address</a>
                                    </td>
                                </tr>
                            </table>
                            <p style="margin: 20px 0 0 0; color: #999999; font-size: 14px; line-height: 1.5;">
                                Or copy and paste this link into your browser:<br>
                                <a href="{verification_url}" style="color: #667eea; word-break: break-all;">{verification_url}</a>
                            </p>
                            <p style="margin: 30px 0 0 0; color: #999999; font-size: 14px; line-height: 1.5;">
                                This link will expire in 24 hours for security reasons.
                            </p>
                        </td>
                    </tr>
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8f9fa; padding: 30px; text-align: center; border-top: 1px solid #e9ecef;">
                            <p style="margin: 0 0 10px 0; color: #999999; font-size: 14px;">
                                If you didn't create an account, you can safely ignore this email.
                            </p>
                            <p style="margin: 0; color: #999999; font-size: 12px;">
                                © 2025 Vidyagam. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
        
    # Attach HTML content
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    # Send email
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    
    logger.info(f"✅ Verification email sent to {to_email}")


def generate_otp() -> str:
    """Generate a secure 6-digit OTP"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])


def send_otp_email_smtp(to_email: str, name: str, otp: str):
    """
    Send OTP via Zoho SMTP
    Raises exception if email fails to send
    """
    smtp_host = os.getenv('SMTP_HOST', 'smtp.zoho.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    logger.info(f"📧 Sending OTP to {to_email} via Zoho SMTP")
    
    if not smtp_password:
        logger.warning("⚠️ SMTP_PASSWORD not configured, email not sent")
        raise Exception("SMTP not configured. Please set SMTP_PASSWORD in environment variables.")
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Your Vidyagam Verification Code'
    msg['From'] = f'Vidyagam <{smtp_user}>'
    msg['To'] = to_email
    
    # HTML email template with OTP
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f5f5f5; padding: 20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <tr>
                        <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: bold;">Vidyagam</h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 16px;">AI Knowledge Hub</p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 40px 30px;">
                            <h2 style="margin: 0 0 20px 0; color: #333333; font-size: 24px;">Welcome{f', {name}' if name else ''}! 👋</h2>
                            <p style="margin: 0 0 20px 0; color: #666666; font-size: 16px; line-height: 1.5;">
                                Thank you for signing up for Vidyagam! Use the verification code below to complete your registration:
                            </p>
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td align="center" style="padding: 30px 0;">
                                        <div style="display: inline-block; background: #f8f9fa; padding: 20px 40px; border-radius: 8px; border: 2px dashed #667eea;">
                                            <span style="font-size: 36px; font-weight: bold; color: #667eea; letter-spacing: 8px; font-family: monospace;">{otp}</span>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                            <p style="margin: 20px 0 0 0; color: #999999; font-size: 14px; line-height: 1.5; text-align: center;">
                                This code will expire in <strong>10 minutes</strong> for security reasons.
                            </p>
                            <p style="margin: 10px 0 0 0; color: #999999; font-size: 14px; line-height: 1.5; text-align: center;">
                                If you didn't request this code, please ignore this email.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="background-color: #f8f9fa; padding: 30px; text-align: center; border-top: 1px solid #e9ecef;">
                            <p style="margin: 0 0 10px 0; color: #999999; font-size: 14px;">
                                Need help? Contact us at support@vidyagam.com
                            </p>
                            <p style="margin: 0; color: #999999; font-size: 12px;">
                                © 2025 Vidyagam. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    # Attach HTML content
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    # Send email
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    
    logger.info(f"✅ OTP email sent to {to_email}")


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


def handle_profile_image(profile_image) -> str:
    """Convert profile_image from memoryview/bytes to string"""
    if isinstance(profile_image, memoryview):
        return profile_image.tobytes().decode('utf-8') if profile_image else None
    elif isinstance(profile_image, bytes):
        return profile_image.decode('utf-8') if profile_image else None
    else:
        return profile_image


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
    """Get current authenticated user with preferences from user_preferences table"""
    try:
        payload = verify_jwt_token(credentials.credentials)
        user_id = payload.get('sub')
        
        # Get user from database with preferences from user_preferences table
        # CRITICAL: Include ID-based preference fields for proper filtering
        user_query = """
            SELECT u.id, u.email, u.first_name, u.last_name, u.profile_image, u.verified_email, u.is_admin,
                   u.created_at,
                   up.experience_level, up.professional_roles,
                   up.categories_selected, up.content_types_selected, up.publishers_selected,
                   up.category_ids_selected, up.content_type_ids_selected, up.publisher_ids_selected,
                   up.newsletter_frequency, up.email_notifications, up.breaking_news_alerts,
                   up.onboarding_completed
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
            WHERE u.id = %s
        """
        user = db.execute_query(user_query, (user_id,), fetch_one=True)
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        user_dict = dict(user)
        
        # Handle profile_image conversion (memoryview to string)
        profile_image = handle_profile_image(user_dict.get('profile_image'))
        
        # Determine onboarding completion based on meaningful preferences (same logic as Google auth)
        # CRITICAL FIX: Check BOTH name-based AND ID-based preferences for onboarding detection
        has_categories = (
            (user_dict.get('categories_selected') and len(user_dict.get('categories_selected', [])) > 0) or
            (user_dict.get('category_ids_selected') and len(user_dict.get('category_ids_selected', [])) > 0)
        )
        has_content_types = (
            (user_dict.get('content_types_selected') and len(user_dict.get('content_types_selected', [])) > 0) or
            (user_dict.get('content_type_ids_selected') and len(user_dict.get('content_type_ids_selected', [])) > 0)
        )
        has_experience_level = user_dict.get('experience_level') is not None
        onboarding_completed = has_categories or has_content_types or has_experience_level or user_dict.get('onboarding_completed', False)

        # Construct the user object with preferences structure expected by frontend
        # CRITICAL FIX: Include ID-based preference fields in response
        result = {
            'id': user_dict['id'],
            'email': user_dict['email'],
            'name': f"{user_dict.get('first_name', '')} {user_dict.get('last_name', '')}".strip() or user_dict['email'],
            'profile_image': profile_image,
            'verified_email': user_dict.get('verified_email', False),
            'is_admin': user_dict.get('is_admin', False),
            'subscription_tier': 'free',  # Default subscription tier
            'created_at': user_dict.get('created_at'),
            'preferences': {
                'experience_level': user_dict.get('experience_level'),
                'professional_roles': user_dict.get('professional_roles', []),
                # Name-based preferences (backward compatibility)
                'categories_selected': user_dict.get('categories_selected', []),
                'content_types_selected': user_dict.get('content_types_selected', []),
                'publishers_selected': user_dict.get('publishers_selected', []),
                # ID-based preferences (preferred, more performant)
                'category_ids_selected': user_dict.get('category_ids_selected', []),
                'content_type_ids_selected': user_dict.get('content_type_ids_selected', []),
                'publisher_ids_selected': user_dict.get('publisher_ids_selected', []),
                # Other preferences
                'newsletter_frequency': user_dict.get('newsletter_frequency', 'weekly'),
                'email_notifications': user_dict.get('email_notifications', True),
                'breaking_news_alerts': user_dict.get('breaking_news_alerts', False),
                'onboarding_completed': onboarding_completed  # Use calculated value based on preference data
            }
        }
        
        logger.info(f"🔍 get_current_user: User {result['email']} - onboarding_completed: {result['preferences']['onboarding_completed']}")
        return result
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
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
            # Update existing user - use first_name and last_name instead of name
            # Update last_login_at only during actual login events
            name_parts = google_user['name'].split(' ', 1) if google_user.get('name') else ['', '']
            first_name = name_parts[0] if len(name_parts) > 0 else 'User'
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            update_query = """
                UPDATE users SET 
                    first_name = %s,
                    last_name = %s, 
                    profile_image = %s, 
                    verified_email = %s,
                    is_google = %s,
                    last_login_at = CURRENT_TIMESTAMP
                WHERE email = %s
                RETURNING *
            """
            user = db.execute_query(
                update_query, 
                (first_name, last_name, google_user['picture'], True, True, google_user['email']),
                fetch_one=True
            )
        else:
            # Create new user - use first_name and last_name instead of name
            name_parts = google_user['name'].split(' ', 1) if google_user.get('name') else ['', '']
            first_name = name_parts[0] if len(name_parts) > 0 else 'User'
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            insert_query = """
                INSERT INTO users (
                    email, first_name, last_name, profile_image, verified_email, is_google,
                    created_at, last_login_at
                ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING *
            """
            user = db.execute_query(
                insert_query,
                (google_user['email'], first_name, last_name, google_user['picture'], True, True),
                fetch_one=True
            )
        
        # Get user preferences from user_preferences table (including ID-based fields)
        preferences_query = """
            SELECT experience_level, professional_roles, categories_selected, 
                   content_types_selected, publishers_selected, newsletter_frequency,
                   email_notifications, breaking_news_alerts, onboarding_completed,
                   category_ids_selected, content_type_ids_selected, publisher_ids_selected
            FROM user_preferences WHERE user_id = %s
        """
        preferences_data = db.execute_query(preferences_query, (user['id'],), fetch_one=True)
        
        logger.info(f"🔍 Google auth: Raw preferences data from database: {preferences_data}")
        logger.info(f"🔍 Google auth: User ID: {user['id']}, Email: {user['email']}")
        
        # Determine onboarding completion based on whether user has meaningful preferences set
        onboarding_completed = False
        is_existing_user = existing_user is not None
        
        if preferences_data:
            # Check if user has meaningful preference data (any categories or content types selected)
            has_categories = preferences_data.get('categories_selected') and len(preferences_data.get('categories_selected', [])) > 0
            has_content_types = preferences_data.get('content_types_selected') and len(preferences_data.get('content_types_selected', [])) > 0
            has_experience_level = preferences_data.get('experience_level') is not None
            
            # Onboarding is complete if user has set any meaningful preferences
            onboarding_completed = has_categories or has_content_types or has_experience_level or preferences_data.get('onboarding_completed', False)
            
            logger.info(f"🔍 Google auth: Preference analysis - categories: {has_categories}, content_types: {has_content_types}, experience: {has_experience_level}")
            logger.info(f"🔍 Google auth: Calculated onboarding_completed: {onboarding_completed}")
        else:
            logger.info(f"🔍 Google auth: NO preferences data found for user {user['id']} - onboarding needed")
        
        # Handle profile_image conversion (memoryview to string)
        profile_image = handle_profile_image(user.get('profile_image'))
            
        # Construct user object with preferences for JWT and response
        user_with_preferences = {
            'id': str(user['id']),  # Ensure ID is string for JWT
            'email': user['email'],
            'name': user.get('name', f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()),
            'profile_image': profile_image,
            'subscription_tier': 'free',  # Default subscription tier
            'verified_email': user.get('verified_email', False),
            'is_admin': user.get('is_admin', False),
            'created_at': user.get('created_at'),
            'preferences': {
                'experience_level': preferences_data.get('experience_level') if preferences_data else None,
                'professional_roles': preferences_data.get('professional_roles', []) if preferences_data else [],
                # Backward compatibility fields
                'categories_selected': preferences_data.get('categories_selected', []) if preferences_data else [],
                'content_types_selected': preferences_data.get('content_types_selected', []) if preferences_data else [],
                'publishers_selected': preferences_data.get('publishers_selected', []) if preferences_data else [],
                # New ID-based fields (preferred)
                'category_ids_selected': preferences_data.get('category_ids_selected', []) if preferences_data else [],
                'content_type_ids_selected': preferences_data.get('content_type_ids_selected', []) if preferences_data else [],
                'publisher_ids_selected': preferences_data.get('publisher_ids_selected', []) if preferences_data else [],
                # Other settings
                'newsletter_frequency': preferences_data.get('newsletter_frequency', 'weekly') if preferences_data else 'weekly',
                'email_notifications': preferences_data.get('email_notifications', True) if preferences_data else True,
                'breaking_news_alerts': preferences_data.get('breaking_news_alerts', False) if preferences_data else False,
                'onboarding_completed': onboarding_completed  # Use calculated value based on preference data
            }
        }
        
        logger.info(f"🔍 Google auth: User {user_with_preferences['email']} - onboarding_completed: {user_with_preferences['preferences']['onboarding_completed']}")
        
        # Create JWT token
        token = create_jwt_token(user_with_preferences)
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user_with_preferences['id']),
            email=user_with_preferences['email'],
            name=user_with_preferences['name'],
            profile_image=user_with_preferences.get('profile_image'),
            subscription_tier='free',  # Default subscription tier
            preferences=user_with_preferences['preferences'],
            created_at=user_with_preferences['created_at'],
            verified_email=user_with_preferences.get('verified_email', False),
            is_admin=user_with_preferences.get('is_admin', False)
        )
        
        logger.info(f"🔍 Google auth: UserResponse created: {user_response.dict()}")
        logger.info(f"🔍 Google auth: Preferences in response: {user_response.preferences}")
        logger.info(f"🔍 Google auth: Onboarding completed in response: {user_response.preferences.get('onboarding_completed')}")
        
        # Return enhanced response with isUserExist flag for frontend logic
        response_data = {
            "access_token": token,
            "token_type": "bearer", 
            "user": user_response,
            "isUserExist": is_existing_user  # Flag to help frontend determine user flow
        }
        
        logger.info(f"🔍 Google auth: Final response - isUserExist: {is_existing_user}, onboarding_completed: {onboarding_completed}")
        
        return response_data
    
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
            "SELECT id, is_google FROM users WHERE email = %s", 
            (request.email,), 
            fetch_one=True
        )
        
        if existing_user:
            if existing_user.get('is_google', False):
                raise HTTPException(status_code=400, detail="This email is registered with Google sign-in. Please use the Google sign-in button.")
            else:
                raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Create new user - use first_name and last_name
        name_parts = request.name.split(' ', 1) if request.name else ['', '']
        first_name = name_parts[0] if len(name_parts) > 0 else 'User'
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        insert_query = """
            INSERT INTO users (
                email, first_name, last_name, password, verified_email, is_google,
                created_at, last_login_at
            ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING *
        """
        user = db.execute_query(
            insert_query,
            (request.email, first_name, last_name, hashed_password, False, False),
            fetch_one=True
        )
        
        # Create JWT token - ensure we have the right data structure
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']
        token_data = {
            'id': str(user['id']),  # Ensure ID is string for JWT
            'email': user['email'],
            'name': full_name,
            'is_admin': user.get('is_admin', False)
        }
        token = create_jwt_token(token_data)
        
        # Convert user to response format - handle first_name/last_name
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=full_name,
            profile_image=handle_profile_image(user.get('profile_image')),
            subscription_tier='free',  # Default subscription tier
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
        # Get user from database with preferences from user_preferences table
        # CRITICAL: Include ID-based preference fields for proper filtering
        user_query = """
            SELECT u.*,
                   up.experience_level, up.professional_roles,
                   up.categories_selected, up.content_types_selected, up.publishers_selected,
                   up.category_ids_selected, up.content_type_ids_selected, up.publisher_ids_selected,
                   up.newsletter_frequency, up.email_notifications, up.breaking_news_alerts,
                   up.onboarding_completed
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
            WHERE u.email = %s
        """
        user = db.execute_query(user_query, (request.email,), fetch_one=True)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if this is a Google user trying to use password
        if user.get('is_google', False):
            raise HTTPException(status_code=400, detail="This account uses Google sign-in. Please use the Google sign-in button.")

        # Check if password exists
        if not user.get('password'):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify password
        if not verify_password(request.password, user['password']):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Update last login
        db.execute_query(
            "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = %s",
            (user['id'],),
            fetch_all=False
        )

        # Create JWT token - ensure we have the right data structure
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']
        token_data = {
            'id': str(user['id']),  # Ensure ID is string for JWT
            'email': user['email'],
            'name': full_name,
            'is_admin': user.get('is_admin', False)
        }
        token = create_jwt_token(token_data)

        # Convert user to response format - handle first_name/last_name
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']

        # CRITICAL FIX: Build preferences object with ID-based fields for signin
        preferences = {
            'experience_level': user.get('experience_level'),
            'professional_roles': user.get('professional_roles', []),
            # Name-based preferences (backward compatibility)
            'categories_selected': user.get('categories_selected', []),
            'content_types_selected': user.get('content_types_selected', []),
            'publishers_selected': user.get('publishers_selected', []),
            # ID-based preferences (preferred, more performant)
            'category_ids_selected': user.get('category_ids_selected', []),
            'content_type_ids_selected': user.get('content_type_ids_selected', []),
            'publisher_ids_selected': user.get('publisher_ids_selected', []),
            # Other preferences
            'newsletter_frequency': user.get('newsletter_frequency', 'weekly'),
            'email_notifications': user.get('email_notifications', True),
            'breaking_news_alerts': user.get('breaking_news_alerts', False),
            'onboarding_completed': user.get('onboarding_completed', False)
        }

        logger.info(f"✅ Signin successful for {user['email']} - category_ids: {preferences.get('category_ids_selected', [])}")

        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=full_name,
            profile_image=handle_profile_image(user.get('profile_image')),
            subscription_tier='free',  # Default subscription tier
            preferences=preferences,
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
        subscription_tier='free',  # Default subscription tier
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
    """Update user preferences in user_preferences table"""
    try:
        logger.info(f"🔍 Updating preferences for user: {current_user['id']}")
        logger.info(f"🔍 Preferences data: {preferences.dict()}")
        
        # Insert or update user preferences in user_preferences table (including ID-based fields)
        upsert_query = """
            INSERT INTO user_preferences (
                user_id, experience_level, professional_roles, 
                categories_selected, content_types_selected, publishers_selected,
                category_ids_selected, content_type_ids_selected, publisher_ids_selected,
                newsletter_frequency, email_notifications, breaking_news_alerts,
                onboarding_completed, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) 
            DO UPDATE SET
                experience_level = EXCLUDED.experience_level,
                professional_roles = EXCLUDED.professional_roles,
                categories_selected = EXCLUDED.categories_selected,
                content_types_selected = EXCLUDED.content_types_selected,
                publishers_selected = EXCLUDED.publishers_selected,
                category_ids_selected = EXCLUDED.category_ids_selected,
                content_type_ids_selected = EXCLUDED.content_type_ids_selected,
                publisher_ids_selected = EXCLUDED.publisher_ids_selected,
                newsletter_frequency = EXCLUDED.newsletter_frequency,
                email_notifications = EXCLUDED.email_notifications,
                breaking_news_alerts = EXCLUDED.breaking_news_alerts,
                onboarding_completed = EXCLUDED.onboarding_completed,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        """
        
        # Convert lists to JSON strings for JSONB columns (psycopg2 converts lists to arrays, not JSONB)
        import json
        
        # Ensure onboarding_completed is always True when preferences are updated
        result = db.execute_query(
            upsert_query,
            (
                current_user['id'],
                preferences.experience_level,
                json.dumps(preferences.professional_roles),  # Convert to JSON string for JSONB
                json.dumps(preferences.categories_selected),  # Convert to JSON string for JSONB
                json.dumps(preferences.content_types_selected),  # Convert to JSON string for JSONB
                json.dumps(preferences.publishers_selected),  # Convert to JSON string for JSONB
                json.dumps(preferences.category_ids_selected),  # Convert to JSON string for JSONB
                json.dumps(preferences.content_type_ids_selected),  # Convert to JSON string for JSONB
                json.dumps(preferences.publisher_ids_selected),  # Convert to JSON string for JSONB
                preferences.newsletter_frequency,
                preferences.email_notifications,
                preferences.breaking_news_alerts,
                True  # Always set onboarding_completed to True when preferences are saved
            ),
            fetch_one=True
        )
        
        logger.info(f"✅ Preferences updated successfully for user: {current_user['id']}")
        logger.info(f"✅ Onboarding completed set to: True")
        return {
            "message": "Preferences updated successfully", 
            "preferences": preferences.dict(),
            "onboarding_completed": True,
            "saved_to": "user_preferences_table"
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to update preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should delete token)"""
    return {"message": "Logged out successfully"}


@router.post("/auth/send-verification")
async def send_verification_email(
    request: dict,
    db: DatabaseService = Depends(get_database_service)
):
    """
    Send email verification link to non-Gmail users
    Uses JWT token for stateless verification (no database table needed)
    """
    try:
        email = request.get('email')
        name = request.get('name')
        password = request.get('password')
        debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Check if Gmail user (should use Google Sign-In instead)
        if email.lower().endswith('@gmail.com') or email.lower().endswith('@googlemail.com'):
            raise HTTPException(
                status_code=400, 
                detail="Gmail users must use Google Sign-In"
            )
        
        # Check if user already exists
        existing_user = db.execute_query(
            "SELECT id FROM users WHERE email = %s", 
            (email,), 
            fetch_one=True
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create JWT verification token with user data (expires in 24 hours)
        hashed_password = hash_password(password)
        verification_payload = {
            'email': email,
            'name': name,
            'password_hash': hashed_password,
            'purpose': 'email_verification',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        verification_token = jwt.encode(verification_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        # Generate verification URL
        verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:5173')}/verify-email?token={verification_token}"
        
        # Send verification email via Zoho SMTP (will raise exception if fails)
        try:
            send_verification_email_smtp(email, name, verification_url)
            logger.info(f"📧 Verification email sent to {email}")
        except Exception as email_error:
            logger.error(f"❌ Email sending failed: {str(email_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send verification email: {str(email_error)}. Please check SMTP configuration."
            )
        return {
            "message": "Verification email sent! Please check your inbox.",
            "verification_url": verification_url if debug_mode else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to send verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/verify-email", response_model=TokenResponse)
async def verify_email(
    request: dict,
    db: DatabaseService = Depends(get_database_service)
):
    """
    Verify email and complete registration using JWT token
    """
    try:
        token = request.get('token')
        
        if not token:
            raise HTTPException(status_code=400, detail="Verification token required")
        
        # Decode and verify JWT token
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=400, detail="Verification link expired. Please sign up again.")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=400, detail="Invalid verification link")
        
        # Verify token purpose
        if payload.get('purpose') != 'email_verification':
            raise HTTPException(status_code=400, detail="Invalid token")
        
        # Extract user data from token
        email = payload.get('email')
        name = payload.get('name')
        password_hash = payload.get('password_hash')
        
        # Check if user already exists (race condition protection)
        existing_user = db.execute_query(
            "SELECT id FROM users WHERE email = %s", 
            (email,), 
            fetch_one=True
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user with verified_email=true and is_google=false
        name_parts = name.split(' ', 1) if name else ['', '']
        first_name = name_parts[0] if len(name_parts) > 0 else 'User'
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        user = db.execute_query("""
            INSERT INTO users (
                email, first_name, last_name, password, verified_email, is_google,
                created_at, last_login_at
            ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING *
        """, (email, first_name, last_name, password_hash, True, False), fetch_one=True)
        
        # Create JWT access token
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']
        token_data = {
            'id': str(user['id']),
            'email': user['email'],
            'name': full_name,
            'is_admin': user.get('is_admin', False)
        }
        access_token = create_jwt_token(token_data)
        
        # Return response
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=full_name,
            first_name=user.get('first_name'),
            last_name=user.get('last_name'),
            profile_image=handle_profile_image(user.get('profile_image')),
            is_admin=user.get('is_admin', False),
            verified_email=True,
            preferences=UserPreferences()
        )
        
        logger.info(f"✅ Email verified and user created: {user['email']}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Email verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/send-otp")
async def send_otp(
    request: dict,
    db: DatabaseService = Depends(get_database_service)
):
    """
    Send OTP to email for verification (non-Gmail users)
    Stores OTP in database with 10-minute expiry
    """
    try:
        email = request.get('email')
        name = request.get('name', '')
        auth_mode = request.get('auth_mode', 'signup')
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check if Gmail user (should use Google Sign-In)
        if email.lower().endswith('@gmail.com') or email.lower().endswith('@googlemail.com'):
            raise HTTPException(status_code=400, detail="Gmail users must use Google Sign-In")
        
        # For signin, check if user exists
        if auth_mode == 'signin':
            existing_user = db.execute_query(
                "SELECT id FROM users WHERE email = %s",
                (email,),
                fetch_one=True
            )
            if not existing_user:
                raise HTTPException(status_code=404, detail="No account found with this email")
        
        # For signup, check if user already exists
        if auth_mode == 'signup':
            existing_user = db.execute_query(
                "SELECT id FROM users WHERE email = %s",
                (email,),
                fetch_one=True
            )
            if existing_user:
                raise HTTPException(status_code=400, detail="User already exists")
        
        # Generate 6-digit OTP
        otp = generate_otp()
        
        # Store OTP in database with 10-minute expiry
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        
        # Delete any existing OTPs for this email (no fetch needed)
        try:
            db.execute_query(
                "DELETE FROM email_otps WHERE email = %s",
                (email,),
                fetch_one=False,
                fetch_all=False
            )
        except:
            pass  # No previous OTP to delete
        
        # Insert new OTP
        db.execute_query("""
            INSERT INTO email_otps (email, otp, user_data, expires_at)
            VALUES (%s, %s, %s, %s)
        """, (email, otp, json.dumps({'name': name, 'auth_mode': auth_mode}), expires_at), fetch_one=False, fetch_all=False)
        
        # Send OTP via email
        try:
            send_otp_email_smtp(email, name, otp)
            logger.info(f"📧 OTP sent to {email}")
        except Exception as email_error:
            logger.error(f"❌ Email sending failed: {str(email_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send OTP email: {str(email_error)}. Please check SMTP configuration."
            )
        
        return {
            "message": "OTP sent successfully! Please check your email.",
            "otpSent": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to send OTP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/verify-otp")
async def verify_otp(
    request: dict,
    db: DatabaseService = Depends(get_database_service)
):
    """
    Verify OTP and complete user registration/signin
    """
    try:
        email = request.get('email')
        otp = request.get('otp')
        user_data = request.get('userData', {})
        
        if not email or not otp:
            raise HTTPException(status_code=400, detail="Email and OTP are required")
        
        # Retrieve OTP from database
        otp_record = db.execute_query("""
            SELECT otp, user_data, expires_at, verified
            FROM email_otps
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (email,), fetch_one=True)
        
        if not otp_record:
            raise HTTPException(status_code=400, detail="No OTP found for this email")
        
        # Check if OTP expired
        if datetime.utcnow() > otp_record['expires_at']:
            raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
        
        # Check if already verified
        if otp_record['verified']:
            raise HTTPException(status_code=400, detail="OTP has already been used")
        
        # Verify OTP
        if otp_record['otp'] != otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        
        # Mark OTP as verified
        db.execute_query(
            "UPDATE email_otps SET verified = TRUE WHERE email = %s",
            (email,),
            fetch_one=False,
            fetch_all=False
        )
        
        # Get stored user data
        stored_data = otp_record['user_data']
        auth_mode = stored_data.get('auth_mode', 'signup')
        name = user_data.get('name') or stored_data.get('name', '')
        password = user_data.get('password', '')  # Get password from frontend
        
        # Check if user exists
        existing_user = db.execute_query(
            "SELECT * FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if auth_mode == 'signin':
            # Sign in existing user
            if not existing_user:
                raise HTTPException(status_code=404, detail="User not found")
            
            user = existing_user
            is_new_user = False
        else:
            # Create new user for signup
            if existing_user:
                raise HTTPException(status_code=400, detail="User already exists")
            
            # Parse name
            name_parts = name.split(' ', 1) if name else ['', '']
            first_name = name_parts[0] if len(name_parts) > 0 else 'User'
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            # Hash password for non-Gmail users
            password_hash = hash_password(password) if password else None
            
            # Create user with verified_email=true, is_google=false, and hashed password
            user = db.execute_query("""
                INSERT INTO users (
                    email, first_name, last_name, password, verified_email, is_google,
                    created_at, last_login_at
                ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING *
            """, (email, first_name, last_name, password_hash, True, False), fetch_one=True)
            
            is_new_user = True
        
        # Update last login
        db.execute_query(
            "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = %s",
            (user['id'],),
            fetch_one=False,
            fetch_all=False
        )
        
        # Create JWT access token
        full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user['email']
        token_data = {
            'id': str(user['id']),
            'email': user['email'],
            'name': full_name,
            'is_admin': user.get('is_admin', False)
        }
        access_token = create_jwt_token(token_data)
        
        # Get user preferences from database
        preferences_query = """
            SELECT experience_level, professional_roles, categories_selected, 
                   content_types_selected, publishers_selected, newsletter_frequency,
                   email_notifications, breaking_news_alerts, onboarding_completed,
                   category_ids_selected, content_type_ids_selected, publisher_ids_selected
            FROM user_preferences WHERE user_id = %s
        """
        preferences_data = db.execute_query(preferences_query, (user['id'],), fetch_one=True)
        
        # Build preferences object (default empty for new users)
        preferences = UserPreferences(
            experience_level=preferences_data.get('experience_level') if preferences_data else None,
            professional_roles=preferences_data.get('professional_roles', []) if preferences_data else [],
            categories_selected=preferences_data.get('categories_selected', []) if preferences_data else [],
            content_types_selected=preferences_data.get('content_types_selected', []) if preferences_data else [],
            publishers_selected=preferences_data.get('publishers_selected', []) if preferences_data else [],
            category_ids_selected=preferences_data.get('category_ids_selected', []) if preferences_data else [],
            content_type_ids_selected=preferences_data.get('content_type_ids_selected', []) if preferences_data else [],
            publisher_ids_selected=preferences_data.get('publisher_ids_selected', []) if preferences_data else [],
            newsletter_frequency=preferences_data.get('newsletter_frequency', 'weekly') if preferences_data else 'weekly',
            email_notifications=preferences_data.get('email_notifications', True) if preferences_data else True,
            breaking_news_alerts=preferences_data.get('breaking_news_alerts', False) if preferences_data else False,
            onboarding_completed=preferences_data.get('onboarding_completed', False) if preferences_data else False
        )
        
        return {
            'user': UserResponse(
                id=str(user['id']),  # Convert to string
                email=user['email'],
                name=full_name,
                is_admin=user.get('is_admin', False),
                preferences=preferences.dict(),  # Convert to dict
                created_at=user.get('created_at'),  # Add created_at
                verified_email=user.get('verified_email', True)
            ),
            'access_token': access_token,
            'isUserExist': not is_new_user
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OTP verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))