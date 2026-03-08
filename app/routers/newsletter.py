"""
Newsletter admin router - manages newsletter generation and sending via SMTP.
Follows same admin auth pattern as admin.py router.
Swappable to SendGrid later by changing EMAIL_PROVIDER env var.
"""
import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from pydantic import BaseModel

from db_service import get_database_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/newsletter")


# ==========================================
# AUTH - same pattern as admin.py
# ==========================================

def require_admin_access(request: Request) -> bool:
    admin_api_key = request.headers.get('X-Admin-API-Key')
    expected_api_key = os.getenv('ADMIN_API_KEY', 'admin-api-key-2024')
    if admin_api_key and admin_api_key == expected_api_key:
        return True
    raise HTTPException(status_code=401, detail="Admin access required")


# ==========================================
# EMAIL PROVIDER (SMTP first, SendGrid ready)
# ==========================================

class EmailProvider:
    def __init__(self):
        self.provider = os.getenv('EMAIL_PROVIDER', 'smtp')
        self.from_email = os.getenv('SMTP_USER', 'digest@vidyagam.com')
        self.from_name = os.getenv('NEWSLETTER_FROM_NAME', 'Vidyagam AI News')

    def send(self, to_email: str, to_name: str, subject: str, html_content: str) -> str:
        if self.provider == 'sendgrid':
            return self._send_sendgrid(to_email, to_name, subject, html_content)
        return self._send_smtp(to_email, to_name, subject, html_content)

    def _send_smtp(self, to_email: str, to_name: str, subject: str, html_content: str) -> str:
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASSWORD')

        if not smtp_user or not smtp_pass:
            raise ValueError("SMTP_USER and SMTP_PASSWORD environment variables required")

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{self.from_name} <{self.from_email}>"
        msg['To'] = f"{to_name} <{to_email}>" if to_name else to_email
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(self.from_email, to_email, msg.as_string())

        msg_id = f"smtp-{datetime.now().timestamp()}"
        logger.info(f"📧 SMTP email sent to {to_email}")
        return msg_id

    def _send_sendgrid(self, to_email, to_name, subject, html_content) -> str:
        # Placeholder for future SendGrid integration
        # pip install sendgrid
        # from sendgrid import SendGridAPIClient
        # from sendgrid.helpers.mail import Mail
        raise NotImplementedError("Set EMAIL_PROVIDER=smtp or install sendgrid")


email_provider = EmailProvider()


# ==========================================
# CONTENT QUERIES
# ==========================================

def get_top_articles(db, period: str, limit: int = 10) -> List[Dict]:
    since = datetime.now() - timedelta(days=1 if period == 'daily' else 7)
    rows = db.execute_query(
        """
        SELECT 
            a.id, a.title, a.url, a.source,
            a.summary AS summary,
            a.significance_score,
            a.published_date,
            c.name AS category_name,
            ct.name AS content_type
        FROM articles a
        LEFT JOIN ai_categories_master c ON a.category_id = c.id
        LEFT JOIN content_types ct ON a.content_type_id = ct.id
        WHERE a.published_date >= %s
        ORDER BY a.significance_score DESC NULLS LAST
        LIMIT %s
        """,
        (since, limit)
    )
    return [dict(r) for r in rows] if rows else []


def get_breaking_news(db, hours: int = 24, limit: int = 5) -> List[Dict]:
    since = datetime.now() - timedelta(hours=hours)
    rows = db.execute_query(
        """
        SELECT a.id, a.title, a.url, a.source, a.summary AS summary,
               a.significance_score, a.published_date,
               c.name AS category_name
        FROM articles a
        LEFT JOIN ai_categories_master c ON a.category_id = c.id
        WHERE a.published_date >= %s
          AND a.significance_score >= 8
        ORDER BY a.significance_score DESC, a.published_date DESC
        LIMIT %s
        """,
        (since, limit)
    )
    return [dict(r) for r in rows] if rows else []


def get_category_roundup(db, days: int = 7) -> Dict[str, List[Dict]]:
    since = datetime.now() - timedelta(days=days)
    rows = db.execute_query(
        """
        SELECT c.name AS category_name,
               a.id, a.title, a.url, a.source, a.summary AS summary,
               a.significance_score
        FROM articles a
        JOIN ai_categories_master c ON a.category_id = c.id
        WHERE a.published_date >= %s AND c.is_active = true
        ORDER BY c.name, a.significance_score DESC NULLS LAST
        """,
        (since,)
    )
    roundup: Dict[str, List[Dict]] = {}
    for row in (rows or []):
        cat = row['category_name']
        if cat not in roundup:
            roundup[cat] = []
        if len(roundup[cat]) < 3:
            roundup[cat].append(dict(row))
    return roundup


def get_subscribed_users(db, frequency: str) -> List[Dict]:
    """
    Get users subscribed to newsletters.
    Uses user_preferences table fields: newsletter_frequency, email_notifications
    """
    rows = db.execute_query(
        """
        SELECT u.id, u.email, u.first_name, u.last_name
        FROM users u
        JOIN user_preferences up ON u.id = up.user_id
        WHERE u.is_active = true
          AND u.verified_email = true
          AND up.email_notifications = true
          AND up.newsletter_frequency = %s
        """,
        (frequency,)
    )
    return [dict(r) for r in rows] if rows else []


# ==========================================
# HTML RENDERER
# ==========================================

def render_newsletter_html(
    frequency: str,
    greeting: str,
    top_stories: List[Dict],
    breaking: List[Dict],
    category_roundup: Optional[Dict] = None,
) -> str:
    is_weekly = frequency == 'weekly'
    date_str = datetime.now().strftime('%B %d, %Y')

    # Breaking news
    breaking_html = ''
    if breaking:
        items = ''.join(f"""
        <tr><td style="padding:10px 16px;border-left:4px solid #e53e3e;background:#fff5f5;">
            <a href="{a['url']}" style="color:#c53030;font-weight:600;text-decoration:none;font-size:15px;">
                🚨 {a['title']}
            </a>
            <div style="font-size:12px;color:#888;margin-top:4px;">{a.get('source','')} · Score: {a.get('significance_score',0)}/10</div>
        </td></tr>
        """ for a in breaking)
        breaking_html = f"""
        <tr><td style="padding:16px 24px 8px;font-size:18px;font-weight:700;border-bottom:2px solid #e53e3e;">
            🚨 Breaking News
        </td></tr>
        {items}
        """

    # Top stories
    stories_html = ''
    for a in top_stories:
        summary = (a.get('summary') or '')[:180]
        ellipsis = '...' if len(a.get('summary', '') or '') > 180 else ''
        stories_html += f"""
        <tr><td style="padding:16px 24px;border-bottom:1px solid #eee;">
            <a href="{a['url']}" style="color:#1a1a1a;font-weight:600;text-decoration:none;font-size:16px;">{a['title']}</a>
            <div style="font-size:12px;color:#888;margin:4px 0;">
                {a.get('source','')} · {a.get('content_type','Article')} ·
                <span style="background:#667eea;color:white;padding:1px 8px;border-radius:10px;font-size:11px;">
                    ⭐ {a.get('significance_score',0)}/10
                </span>
            </div>
            <div style="font-size:14px;color:#555;line-height:1.5;">{summary}{ellipsis}</div>
        </td></tr>
        """

    # Category roundup (weekly only)
    category_html = ''
    if is_weekly and category_roundup:
        cat_items = ''
        for cat, articles in category_roundup.items():
            art_links = ''.join(f"""
            <div style="padding:6px 0;">
                <a href="{a['url']}" style="color:#1a1a1a;text-decoration:none;font-size:14px;font-weight:500;">{a['title']}</a>
                <span style="font-size:11px;color:#888;margin-left:8px;">{a.get('source','')} · ⭐{a.get('significance_score',0)}</span>
            </div>
            """ for a in articles)
            cat_items += f"""
            <tr><td style="padding:12px 24px;">
                <div style="font-size:13px;font-weight:600;color:#667eea;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">{cat}</div>
                {art_links}
            </td></tr>
            """
        category_html = f"""
        <tr><td style="padding:16px 24px 8px;font-size:18px;font-weight:700;border-bottom:2px solid #667eea;">
            📂 Category Roundup
        </td></tr>
        {cat_items}
        """

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;max-width:600px;">
    <tr><td style="background:linear-gradient(135deg,#667eea,#764ba2);padding:32px 24px;text-align:center;color:white;">
        <h1 style="margin:0;font-size:24px;">{'📊 Weekly AI Roundup' if is_weekly else '🤖 AI Daily Digest'}</h1>
        <p style="margin:8px 0 0;opacity:0.9;font-size:14px;">{date_str}</p>
    </td></tr>
    <tr><td style="padding:24px 24px 8px;font-size:16px;color:#555;">{greeting}</td></tr>
    {breaking_html}
    <tr><td style="padding:16px 24px 8px;font-size:18px;font-weight:700;border-bottom:2px solid #667eea;">📰 Top Stories</td></tr>
    {stories_html}
    {category_html}
    <tr><td style="text-align:center;padding:32px 24px;">
        <a href="https://www.vidyagam.com" style="display:inline-block;background:#667eea;color:white;padding:12px 32px;border-radius:8px;text-decoration:none;font-weight:600;">
            Read More on Vidyagam →
        </a>
    </td></tr>
    <tr><td style="background:#f8f9fa;padding:24px;text-align:center;font-size:12px;color:#888;">
        <p>You're receiving this because you subscribed to {'weekly' if is_weekly else 'daily'} digests.</p>
        <p><a href="https://www.vidyagam.com/preferences" style="color:#667eea;text-decoration:none;">Manage Preferences</a> ·
           <a href="https://www.vidyagam.com/unsubscribe" style="color:#667eea;text-decoration:none;">Unsubscribe</a></p>
        <p>© {datetime.now().year} Vidyagam AI News</p>
    </td></tr>
</table>
</td></tr></table>
</body></html>"""


# ==========================================
# API ENDPOINTS
# ==========================================

@router.get("/stats")
async def newsletter_stats(admin_access: bool = Depends(require_admin_access)):
    db = get_database_service()
    stats = db.execute_query("""
        SELECT
            COALESCE((SELECT COUNT(*) FROM newsletter_editions WHERE status = 'sent'), 0) as total_editions,
            COALESCE((SELECT SUM(total_sent) FROM newsletter_editions WHERE status = 'sent'), 0) as total_emails_sent,
            COALESCE((SELECT SUM(total_opened) FROM newsletter_editions WHERE status = 'sent'), 0) as total_opened,
            COALESCE((SELECT SUM(total_clicked) FROM newsletter_editions WHERE status = 'sent'), 0) as total_clicked,
            COALESCE((SELECT COUNT(*) FROM user_preferences WHERE email_notifications = true), 0) as total_subscribers,
            COALESCE((SELECT COUNT(*) FROM user_preferences WHERE email_notifications = true AND newsletter_frequency = 'daily'), 0) as daily_subscribers,
            COALESCE((SELECT COUNT(*) FROM user_preferences WHERE email_notifications = true AND newsletter_frequency = 'weekly'), 0) as weekly_subscribers
    """)
    return dict(stats[0]) if stats else {
        "total_editions": 0, "total_emails_sent": 0, "total_opened": 0,
        "total_clicked": 0, "total_subscribers": 0, "daily_subscribers": 0, "weekly_subscribers": 0
    }


@router.get("/editions")
async def list_editions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    admin_access: bool = Depends(require_admin_access)
):
    db = get_database_service()
    offset = (page - 1) * limit
    rows = db.execute_query(
        """
        SELECT id, edition_type, edition_date, subject_line,
               total_recipients, total_sent, total_opened, total_clicked,
               status, sent_at, created_at
        FROM newsletter_editions
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        (limit, offset)
    )
    editions = []
    for r in (rows or []):
        d = dict(r)
        for field in ['edition_date', 'sent_at', 'created_at']:
            if d.get(field) and hasattr(d[field], 'isoformat'):
                d[field] = d[field].isoformat()
        editions.append(d)
    return {"editions": editions}


@router.get("/preview")
async def preview_newsletter(
    frequency: str = Query('daily'),
    admin_access: bool = Depends(require_admin_access)
):
    db = get_database_service()
    hours = 24 if frequency == 'daily' else 168
    limit = 10 if frequency == 'daily' else 20

    top_stories = get_top_articles(db, frequency, limit=limit)
    breaking = get_breaking_news(db, hours=hours)
    category_roundup = get_category_roundup(db) if frequency == 'weekly' else None

    html = render_newsletter_html(
        frequency=frequency, greeting='Hi there,',
        top_stories=top_stories, breaking=breaking,
        category_roundup=category_roundup,
    )
    return {
        "subject": f"{'📊 Weekly AI Roundup' if frequency == 'weekly' else '🤖 AI Daily Digest'} — {datetime.now().strftime('%b %d, %Y')}",
        "html": html,
        "article_count": len(top_stories),
        "breaking_count": len(breaking),
    }


@router.post("/trigger")
async def trigger_newsletter(
    frequency: str = Query(...),
    admin_access: bool = Depends(require_admin_access)
):
    db = get_database_service()
    hours = 24 if frequency == 'daily' else 168
    limit = 10 if frequency == 'daily' else 20

    top_stories = get_top_articles(db, frequency, limit=limit)
    breaking = get_breaking_news(db, hours=hours)
    category_roundup = get_category_roundup(db) if frequency == 'weekly' else None

    if not top_stories:
        return {"success": False, "result": {"sent": 0, "failed": 0, "reason": "no_content"}}

    users = get_subscribed_users(db, frequency)
    if not users:
        return {"success": False, "result": {"sent": 0, "failed": 0, "reason": "no_subscribers"}}

    subject = f"{'📊 Weekly AI Roundup' if frequency == 'weekly' else '🤖 AI Daily Digest'} — {datetime.now().strftime('%b %d, %Y')}"
    article_ids = list(set([a['id'] for a in top_stories] + [a['id'] for a in breaking]))

    # Render archive version (generic greeting, stored for future reference)
    archive_html = render_newsletter_html(
        frequency=frequency, greeting='Hi there,',
        top_stories=top_stories, breaking=breaking,
        category_roundup=category_roundup,
    )

    edition_row = db.execute_query(
        """
        INSERT INTO newsletter_editions
            (edition_type, edition_date, subject_line, html_content, articles_included, status)
        VALUES (%s, %s, %s, %s, %s, 'sending')
        ON CONFLICT (edition_type, edition_date)
        DO UPDATE SET subject_line = EXCLUDED.subject_line,
                      html_content = EXCLUDED.html_content,
                      articles_included = EXCLUDED.articles_included,
                      status = 'sending',
                      created_at = NOW()
        RETURNING id
        """,
        (frequency, datetime.now().date(), subject, archive_html, article_ids)
    )
    edition_id = edition_row[0]['id']

    sent = 0
    failed = 0
    for user in users:
        name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or 'there'
        html = render_newsletter_html(
            frequency=frequency, greeting=f"Hi {name},",
            top_stories=top_stories, breaking=breaking,
            category_roundup=category_roundup,
        )
        try:
            msg_id = email_provider.send(user['email'], name, subject, html)
            db.execute_query(
                """INSERT INTO newsletter_delivery_logs
                   (edition_id, user_id, email, status, provider_message_id, sent_at)
                   VALUES (%s, %s, %s, 'sent', %s, %s)
                   RETURNING id""",
                (edition_id, user['id'], user['email'], msg_id, datetime.now())
            )
            sent += 1
        except Exception as e:
            logger.error(f"❌ Failed to send to {user['email']}: {e}")
            db.execute_query(
                """INSERT INTO newsletter_delivery_logs
                   (edition_id, user_id, email, status, error_message)
                   VALUES (%s, %s, %s, 'failed', %s)
                   RETURNING id""",
                (edition_id, user['id'], user['email'], str(e))
            )
            failed += 1

    db.execute_query(
        """UPDATE newsletter_editions
           SET total_recipients = %s, total_sent = %s, status = 'sent', sent_at = %s
           WHERE id = %s
           RETURNING id""",
        (len(users), sent, datetime.now(), edition_id)
    )
    logger.info(f"✅ Newsletter sent: {sent}/{len(users)}, {failed} failed")
    return {"success": True, "result": {"sent": sent, "failed": failed, "total": len(users)}}


@router.post("/send-test")
async def send_test_newsletter(
    body: Dict[str, Any] = Body(...),
    admin_access: bool = Depends(require_admin_access)
):
    frequency = body.get('frequency', 'daily')
    email = body.get('email')
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    db = get_database_service()
    hours = 24 if frequency == 'daily' else 168
    limit = 10 if frequency == 'daily' else 20

    top_stories = get_top_articles(db, frequency, limit=limit)
    breaking = get_breaking_news(db, hours=hours)
    category_roundup = get_category_roundup(db) if frequency == 'weekly' else None

    html = render_newsletter_html(
        frequency=frequency, greeting='Hi [Test Recipient],',
        top_stories=top_stories, breaking=breaking,
        category_roundup=category_roundup,
    )
    subject = f"[TEST] {'📊 Weekly' if frequency == 'weekly' else '🤖 Daily'} — {datetime.now().strftime('%b %d')}"

    try:
        email_provider.send(email, 'Test Recipient', subject, html)
        return {"success": True, "message": f"Test email sent to {email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


# ==========================================
# NEWSLETTER ARCHIVE
# ==========================================

@router.get("/archive")
async def list_archived_newsletters(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    edition_type: Optional[str] = Query(None),
    admin_access: bool = Depends(require_admin_access)
):
    """List archived newsletters with metadata (without full HTML)."""
    db = get_database_service()
    offset = (page - 1) * limit

    type_filter = ""
    params: list = []
    if edition_type:
        type_filter = "WHERE edition_type = %s"
        params.append(edition_type)

    params.extend([limit, offset])

    rows = db.execute_query(
        f"""
        SELECT id, edition_type, edition_date, subject_line,
               total_recipients, total_sent, total_opened, total_clicked,
               status, sent_at, created_at,
               CASE WHEN html_content IS NOT NULL AND html_content != '' THEN true ELSE false END AS has_archive
        FROM newsletter_editions
        {type_filter}
        ORDER BY edition_date DESC, created_at DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params)
    )

    total_row = db.execute_query(
        f"SELECT COUNT(*) as count FROM newsletter_editions {type_filter}",
        tuple(params[:1]) if edition_type else None
    )

    editions = []
    for r in (rows or []):
        d = dict(r)
        for field in ['edition_date', 'sent_at', 'created_at']:
            if d.get(field) and hasattr(d[field], 'isoformat'):
                d[field] = d[field].isoformat()
        editions.append(d)

    return {
        "editions": editions,
        "total": total_row[0]["count"] if total_row else 0,
        "page": page,
        "limit": limit
    }


@router.get("/archive/{edition_id}")
async def get_archived_newsletter(
    edition_id: int,
    admin_access: bool = Depends(require_admin_access)
):
    """Get a single archived newsletter with full HTML content for viewing."""
    db = get_database_service()

    rows = db.execute_query(
        """
        SELECT id, edition_type, edition_date, subject_line, html_content,
               articles_included, total_recipients, total_sent,
               total_opened, total_clicked, status, sent_at, created_at
        FROM newsletter_editions
        WHERE id = %s
        """,
        (edition_id,)
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Newsletter edition not found")

    edition = dict(rows[0])
    for field in ['edition_date', 'sent_at', 'created_at']:
        if edition.get(field) and hasattr(edition[field], 'isoformat'):
            edition[field] = edition[field].isoformat()

    # Get delivery details
    delivery_rows = db.execute_query(
        """
        SELECT dl.email, dl.status, dl.sent_at, dl.error_message,
               u.first_name, u.last_name
        FROM newsletter_delivery_logs dl
        LEFT JOIN users u ON dl.user_id = u.id
        WHERE dl.edition_id = %s
        ORDER BY dl.sent_at DESC NULLS LAST
        """,
        (edition_id,)
    )

    deliveries = []
    for r in (delivery_rows or []):
        d = dict(r)
        if d.get('sent_at') and hasattr(d['sent_at'], 'isoformat'):
            d['sent_at'] = d['sent_at'].isoformat()
        deliveries.append(d)

    edition["deliveries"] = deliveries

    return edition