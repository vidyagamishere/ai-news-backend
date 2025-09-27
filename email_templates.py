"""
Email template system for AI News Scraper subscriptions
This module provides HTML email templates for different subscription types
"""

from datetime import datetime
from typing import List, Dict

def generate_daily_digest_email(user_name: str, articles: List[Dict], multimedia_content: Dict = None) -> str:
    """Generate HTML email for daily digest"""
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Build articles HTML
    articles_html = ""
    for i, article in enumerate(articles[:10], 1):
        significance_color = "#059669" if article.get('significance_score', 0) >= 7 else "#3B82F6" if article.get('significance_score', 0) >= 5 else "#6B7280"
        
        articles_html += f"""
        <div style="margin-bottom: 24px; padding: 20px; background: #f8fafc; border-radius: 12px; border-left: 4px solid {significance_color};">
            <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 18px; font-weight: 600;">
                <a href="{article.get('url', '#')}" style="color: #1e293b; text-decoration: none;">{article.get('title', 'Untitled')}</a>
            </h3>
            <p style="margin: 0 0 12px 0; color: #475569; font-size: 14px; line-height: 1.5;">
                {article.get('summary', article.get('content', ''))[:200]}...
            </p>
            <div style="display: flex; align-items: center; gap: 16px; font-size: 12px; color: #64748b;">
                <span>📰 {article.get('source', 'Unknown')}</span>
                <span>📊 Impact: {article.get('significance_score', 0):.1f}/10</span>
                <span>🕒 {article.get('time', 'Recently')}</span>
            </div>
        </div>
        """
    
    # Build multimedia HTML if provided
    multimedia_html = ""
    if multimedia_content:
        if multimedia_content.get('audio'):
            multimedia_html += """
            <h2 style="color: #1e293b; font-size: 20px; margin: 32px 0 16px 0;">🎧 Audio Content</h2>
            """
            for audio in multimedia_content['audio'][:3]:
                multimedia_html += f"""
                <div style="margin-bottom: 16px; padding: 16px; background: #fef3c7; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1e293b;">
                        <a href="{audio.get('url', '#')}" style="color: #1e293b; text-decoration: none;">{audio.get('title', 'Untitled')}</a>
                    </h4>
                    <p style="margin: 0; color: #475569; font-size: 14px;">{audio.get('description', '')[:150]}...</p>
                </div>
                """
        
        if multimedia_content.get('video'):
            multimedia_html += """
            <h2 style="color: #1e293b; font-size: 20px; margin: 32px 0 16px 0;">📹 Video Content</h2>
            """
            for video in multimedia_content['video'][:3]:
                multimedia_html += f"""
                <div style="margin-bottom: 16px; padding: 16px; background: #ecfccb; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1e293b;">
                        <a href="{video.get('url', '#')}" style="color: #1e293b; text-decoration: none;">{video.get('title', 'Untitled')}</a>
                    </h4>
                    <p style="margin: 0; color: #475569; font-size: 14px;">{video.get('description', '')[:150]}...</p>
                </div>
                """
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Daily Digest - {current_date}</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #ffffff;">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 32px; padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white;">
            <h1 style="margin: 0; font-size: 28px; font-weight: 800;">🤖 AI Daily</h1>
            <p style="margin: 8px 0 0 0; font-size: 16px; opacity: 0.9;">Your curated AI news digest</p>
            <p style="margin: 4px 0 0 0; font-size: 14px; opacity: 0.8;">{current_date}</p>
        </div>
        
        <!-- Greeting -->
        <div style="margin-bottom: 32px;">
            <p style="font-size: 16px; color: #1e293b;">Hello {user_name},</p>
            <p style="font-size: 16px; color: #475569;">Here's your personalized AI news digest with the latest developments in artificial intelligence.</p>
        </div>
        
        <!-- Articles -->
        <h2 style="color: #1e293b; font-size: 24px; margin-bottom: 24px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">📰 Top Stories</h2>
        {articles_html}
        
        {multimedia_html}
        
        <!-- Footer -->
        <div style="margin-top: 48px; padding: 24px; background: #f1f5f9; border-radius: 12px; text-align: center;">
            <p style="margin: 0 0 16px 0; color: #64748b; font-size: 14px;">
                This digest was curated by AI and delivered to you by AI Daily.
            </p>
            <p style="margin: 0; color: #64748b; font-size: 12px;">
                <a href="[PREFERENCES_URL]" style="color: #3b82f6; text-decoration: none;">Update preferences</a> • 
                <a href="[UNSUBSCRIBE_URL]" style="color: #3b82f6; text-decoration: none;">Unsubscribe</a>
            </p>
        </div>
        
    </body>
    </html>
    """
    
    return html_template

def generate_welcome_email(user_name: str) -> str:
    """Generate welcome email for new subscribers"""
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome to AI Daily</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #ffffff;">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 32px; padding: 24px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 16px; color: white;">
            <h1 style="margin: 0; font-size: 28px; font-weight: 800;">🤖 Welcome to AI Daily!</h1>
            <p style="margin: 8px 0 0 0; font-size: 16px; opacity: 0.9;">Your journey into AI news begins now</p>
        </div>
        
        <!-- Welcome Message -->
        <div style="margin-bottom: 32px;">
            <p style="font-size: 16px; color: #1e293b;">Hello {user_name},</p>
            <p style="font-size: 16px; color: #475569;">
                Welcome to AI Daily! We're excited to have you join our community of AI enthusiasts and professionals.
            </p>
            <p style="font-size: 16px; color: #475569;">
                You'll receive curated AI news digests based on your preferences, featuring:
            </p>
        </div>
        
        <!-- Features -->
        <div style="margin-bottom: 32px;">
            <div style="display: grid; gap: 16px;">
                <div style="padding: 16px; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #10b981;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px;">📰 Curated Articles</h3>
                    <p style="margin: 0; color: #475569; font-size: 14px;">Latest AI research, industry news, and breakthrough announcements</p>
                </div>
                <div style="padding: 16px; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px;">🎧 Audio & Podcasts</h3>
                    <p style="margin: 0; color: #475569; font-size: 14px;">AI-focused podcasts and audio content from leading experts</p>
                </div>
                <div style="padding: 16px; background: #ecfccb; border-radius: 8px; border-left: 4px solid #84cc16;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px;">📹 Video Content</h3>
                    <p style="margin: 0; color: #475569; font-size: 14px;">Conference talks, tutorials, and AI demonstrations</p>
                </div>
                <div style="padding: 16px; background: #dbeafe; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px;">🔍 Smart Curation</h3>
                    <p style="margin: 0; color: #475569; font-size: 14px;">AI-powered significance scoring and personalized recommendations</p>
                </div>
            </div>
        </div>
        
        <!-- CTA -->
        <div style="text-align: center; margin: 32px 0;">
            <a href="[PREFERENCES_URL]" style="display: inline-block; padding: 12px 24px; background: #3b82f6; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">Customize Your Preferences</a>
        </div>
        
        <!-- Footer -->
        <div style="margin-top: 48px; padding: 24px; background: #f1f5f9; border-radius: 12px; text-align: center;">
            <p style="margin: 0 0 16px 0; color: #64748b; font-size: 14px;">
                Thank you for subscribing to AI Daily. Your first digest will arrive soon!
            </p>
            <p style="margin: 0; color: #64748b; font-size: 12px;">
                <a href="[PREFERENCES_URL]" style="color: #3b82f6; text-decoration: none;">Update preferences</a> • 
                <a href="[UNSUBSCRIBE_URL]" style="color: #3b82f6; text-decoration: none;">Unsubscribe</a>
            </p>
        </div>
        
    </body>
    </html>
    """
    
    return html_template

def generate_text_digest(user_name: str, articles: List[Dict]) -> str:
    """Generate plain text email for users who prefer text"""
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    text_content = f"""
AI DAILY DIGEST - {current_date}

Hello {user_name},

Here's your personalized AI news digest with the latest developments in artificial intelligence.

TOP STORIES
-----------

"""
    
    for i, article in enumerate(articles[:10], 1):
        text_content += f"""
{i}. {article.get('title', 'Untitled')}
   Source: {article.get('source', 'Unknown')} | Impact Score: {article.get('significance_score', 0):.1f}/10
   {article.get('summary', article.get('content', ''))[:200]}...
   Read more: {article.get('url', '#')}

"""
    
    text_content += f"""
---

This digest was curated by AI and delivered to you by AI Daily.

Update preferences: [PREFERENCES_URL]
Unsubscribe: [UNSUBSCRIBE_URL]
    """
    
    return text_content