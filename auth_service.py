"""
Authentication service for AI News Scraper
"""
import hashlib
import os
import secrets
import sqlite3
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from auth_models import (
    User, UserCreate, UserLogin, PasswordHash, UserSession, 
    SubscriptionTier, UserPreferences, AITopic, TopicCategory, ContentType
)

class AuthService:
    def __init__(self, db_path: str, jwt_secret: str, google_client_id: str = ""):
        self.db_path = db_path
        self.jwt_secret = jwt_secret
        self.google_client_id = google_client_id
        self._init_database()
        self._init_default_topics()
    
    def _init_database(self):
        """Initialize the authentication database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                profile_image TEXT,
                subscription_tier TEXT DEFAULT 'free',
                preferences TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Password hashes table (separate for security)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_passwords (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # AI Topics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # User sessions table (for token management)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_default_topics(self):
        """Initialize default AI topics"""
        default_topics = [
            ("ml_general", "Machine Learning", "Latest developments in machine learning", "technology"),
            ("ai_tools", "AI Tools", "New AI tools and platforms", "tools"),
            ("ai_ethics", "AI Ethics", "Ethical considerations in AI development", "ethics"),
            ("computer_vision", "Computer Vision", "Image and video AI technologies", "technology"),
            ("nlp", "Natural Language Processing", "Language AI and chatbots", "technology"),
            ("ai_healthcare", "AI in Healthcare", "Medical AI applications", "industry"),
            ("ai_research", "AI Research", "Academic research and papers", "research"),
            ("autonomous_vehicles", "Autonomous Vehicles", "Self-driving car technology", "technology"),
            ("ai_robotics", "AI & Robotics", "Robotics and automation", "technology"),
            ("ai_business", "AI in Business", "Enterprise AI applications", "industry"),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for topic_id, name, description, category in default_topics:
            cursor.execute('''
                INSERT OR IGNORE INTO ai_topics (id, name, description, category)
                VALUES (?, ?, ?, ?)
            ''', (topic_id, name, description, category))
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> PasswordHash:
        """Hash a password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                           password.encode('utf-8'), 
                                           salt.encode('utf-8'), 
                                           100000)
        return PasswordHash(
            password_hash=password_hash.hex(),
            salt=salt
        )
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against its hash"""
        computed_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode('utf-8'),
                                          salt.encode('utf-8'),
                                          100000)
        return computed_hash.hex() == password_hash
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID"""
        return f"user_{secrets.token_urlsafe(16)}"
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier.value,
            'exp': datetime.utcnow() + timedelta(days=30),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_available_topics(self) -> List[AITopic]:
        """Get all available AI topics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, category 
            FROM ai_topics 
            WHERE is_active = 1
            ORDER BY name
        ''')
        
        topics = []
        for row in cursor.fetchall():
            topics.append(AITopic(
                id=row[0],
                name=row[1],
                description=row[2],
                category=TopicCategory(row[3]),
                selected=False
            ))
        
        conn.close()
        return topics
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_data.email,))
        if cursor.fetchone():
            conn.close()
            raise ValueError("User with this email already exists")
        
        # Generate user ID and hash password
        user_id = self._generate_user_id()
        password_hash = self._hash_password(user_data.password)
        
        # Create default preferences
        default_preferences = UserPreferences()
        preferences_json = default_preferences.model_dump_json()
        
        try:
            # Insert user
            cursor.execute('''
                INSERT INTO users (id, email, name, subscription_tier, preferences, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_data.email,
                user_data.name,
                SubscriptionTier.FREE.value,
                preferences_json,
                datetime.utcnow()
            ))
            
            # Insert password hash
            cursor.execute('''
                INSERT INTO user_passwords (user_id, password_hash, salt)
                VALUES (?, ?, ?)
            ''', (user_id, password_hash.password_hash, password_hash.salt))
            
            conn.commit()
            
            # Return created user
            user = User(
                id=user_id,
                email=user_data.email,
                name=user_data.name,
                subscription_tier=SubscriptionTier.FREE,
                preferences=default_preferences,
                created_at=datetime.utcnow()
            )
            
            return user
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate user with email and password"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user and password data
        cursor.execute('''
            SELECT u.id, u.email, u.name, u.profile_image, u.subscription_tier, 
                   u.preferences, u.created_at, u.last_login_at,
                   p.password_hash, p.salt
            FROM users u
            JOIN user_passwords p ON u.id = p.user_id
            WHERE u.email = ? AND u.is_active = 1
        ''', (login_data.email,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        # Verify password
        if not self._verify_password(login_data.password, row[8], row[9]):
            conn.close()
            return None
        
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login_at = ? WHERE id = ?
        ''', (datetime.utcnow(), row[0]))
        conn.commit()
        conn.close()
        
        # Parse preferences
        preferences_data = json.loads(row[5]) if row[5] else {}
        preferences = UserPreferences(**preferences_data)
        
        return User(
            id=row[0],
            email=row[1],
            name=row[2],
            profile_image=row[3],
            subscription_tier=SubscriptionTier(row[4]),
            preferences=preferences,
            created_at=row[6],
            last_login_at=row[7]
        )
    
    def google_authenticate(self, id_token_str: str) -> Optional[User]:
        """Authenticate user with Google OAuth"""
        if not self.google_client_id:
            raise ValueError("Google authentication not configured")
        
        try:
            # Verify the ID token
            idinfo = id_token.verify_oauth2_token(
                id_token_str, 
                google_requests.Request(), 
                self.google_client_id
            )
            
            # Check if the token is for the right app
            if idinfo['aud'] != self.google_client_id:
                raise ValueError('Invalid Google client ID')
            
            email = idinfo['email']
            name = idinfo.get('name', email.split('@')[0])
            profile_image = idinfo.get('picture')
            
            # Check if user exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, profile_image, subscription_tier, 
                       preferences, created_at, last_login_at
                FROM users WHERE email = ? AND is_active = 1
            ''', (email,))
            
            row = cursor.fetchone()
            
            if row:
                # Existing user - update last login and profile image
                cursor.execute('''
                    UPDATE users 
                    SET last_login_at = ?, profile_image = ?
                    WHERE id = ?
                ''', (datetime.utcnow(), profile_image, row[0]))
                conn.commit()
                
                preferences_data = json.loads(row[5]) if row[5] else {}
                preferences = UserPreferences(**preferences_data)
                
                user = User(
                    id=row[0],
                    email=row[1],
                    name=row[2],
                    profile_image=profile_image,
                    subscription_tier=SubscriptionTier(row[4]),
                    preferences=preferences,
                    created_at=row[6],
                    last_login_at=datetime.utcnow()
                )
            else:
                # New user - create account
                user_id = self._generate_user_id()
                default_preferences = UserPreferences()
                preferences_json = default_preferences.model_dump_json()
                
                cursor.execute('''
                    INSERT INTO users 
                    (id, email, name, profile_image, subscription_tier, preferences, created_at, last_login_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, email, name, profile_image, 
                    SubscriptionTier.FREE.value, preferences_json,
                    datetime.utcnow(), datetime.utcnow()
                ))
                conn.commit()
                
                user = User(
                    id=user_id,
                    email=email,
                    name=name,
                    profile_image=profile_image,
                    subscription_tier=SubscriptionTier.FREE,
                    preferences=default_preferences,
                    created_at=datetime.utcnow(),
                    last_login_at=datetime.utcnow()
                )
            
            conn.close()
            return user
            
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Google authentication failed: {str(e)}")
    
    def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user from JWT token"""
        payload = self._verify_jwt_token(token)
        if not payload:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, name, profile_image, subscription_tier, 
                   preferences, created_at, last_login_at
            FROM users WHERE id = ? AND is_active = 1
        ''', (payload['user_id'],))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        preferences_data = json.loads(row[5]) if row[5] else {}
        preferences = UserPreferences(**preferences_data)
        
        return User(
            id=row[0],
            email=row[1],
            name=row[2],
            profile_image=row[3],
            subscription_tier=SubscriptionTier(row[4]),
            preferences=preferences,
            created_at=row[6],
            last_login_at=row[7]
        )
    
    def update_user_preferences(self, user_id: str, preferences_update: dict) -> Optional[User]:
        """Update user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current preferences
        cursor.execute('SELECT preferences FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        current_preferences = json.loads(row[0]) if row[0] else {}
        
        # Update preferences
        current_preferences.update(preferences_update)
        updated_preferences_json = json.dumps(current_preferences)
        
        cursor.execute('''
            UPDATE users SET preferences = ? WHERE id = ?
        ''', (updated_preferences_json, user_id))
        conn.commit()
        conn.close()
        
        return self.get_user_by_id(user_id)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, name, profile_image, subscription_tier, 
                   preferences, created_at, last_login_at
            FROM users WHERE id = ? AND is_active = 1
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        preferences_data = json.loads(row[5]) if row[5] else {}
        preferences = UserPreferences(**preferences_data)
        
        return User(
            id=row[0],
            email=row[1],
            name=row[2],
            profile_image=row[3],
            subscription_tier=SubscriptionTier(row[4]),
            preferences=preferences,
            created_at=row[6],
            last_login_at=row[7]
        )
    
    def upgrade_user_subscription(self, user_id: str) -> Optional[User]:
        """Upgrade user to premium subscription"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET subscription_tier = ? WHERE id = ?
        ''', (SubscriptionTier.PREMIUM.value, user_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return None
        
        conn.commit()
        conn.close()
        
        return self.get_user_by_id(user_id)
    
    def generate_auth_response(self, user: User) -> dict:
        """Generate authentication response with token"""
        token = self._generate_jwt_token(user)
        return {
            "user": user.model_dump(),
            "token": token,
            "message": "Authentication successful"
        }