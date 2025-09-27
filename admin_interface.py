#!/usr/bin/env python3
"""
Admin Web Interface for AI News Scraper PostgreSQL Database Management
Features: Authentication, CRUD operations, Data export/import, Intuitive UI
"""

import os
import sys
import json
import csv
import io
import logging
import secrets
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'admin-interface-secret-2025')

# PostgreSQL connection
POSTGRES_URL = os.getenv('POSTGRES_URL', "postgresql://postgres:FgvftzrGueiGipLiRRMKMElppasuzBjptZlwPL@autorack.proxy.rlwy.net:51308/railway")

# Admin configuration
ADMIN_EMAIL = "admin@vidyagam.com"
BREVO_API_KEY = os.getenv('BREVO_API_KEY', '')  # Add your Brevo API key

class DatabaseManager:
    def __init__(self):
        self.connection_url = POSTGRES_URL
    
    def get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(self.connection_url, cursor_factory=RealDictCursor)
    
    def execute_query(self, query: str, params: tuple = None, fetch_all: bool = True):
        """Execute database query"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                result = cursor.fetchall() if fetch_all else cursor.fetchone()
            else:
                conn.commit()
                result = cursor.rowcount
            
            cursor.close()
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise e
    
    def get_tables(self):
        """Get all table names"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
        """
        return self.execute_query(query)
    
    def get_table_structure(self, table_name: str):
        """Get table column information"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns 
        WHERE table_name = %s AND table_schema = 'public'
        ORDER BY ordinal_position;
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_data(self, table_name: str, limit: int = 100, offset: int = 0):
        """Get table data with pagination"""
        query = f"SELECT * FROM {table_name} ORDER BY 1 LIMIT %s OFFSET %s;"
        return self.execute_query(query, (limit, offset))
    
    def get_table_count(self, table_name: str):
        """Get total row count for table"""
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        result = self.execute_query(query, fetch_all=False)
        return result['count'] if result else 0
    
    def insert_record(self, table_name: str, data: Dict[str, Any]):
        """Insert new record"""
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ','.join(['%s'] * len(values))
        
        query = f"""
        INSERT INTO {table_name} ({','.join(columns)}) 
        VALUES ({placeholders});
        """
        return self.execute_query(query, tuple(values))
    
    def update_record(self, table_name: str, data: Dict[str, Any], where_clause: str, where_params: tuple):
        """Update existing record"""
        set_clause = ','.join([f"{k} = %s" for k in data.keys()])
        values = list(data.values()) + list(where_params)
        
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause};"
        return self.execute_query(query, tuple(values))
    
    def delete_record(self, table_name: str, where_clause: str, where_params: tuple):
        """Delete record"""
        query = f"DELETE FROM {table_name} WHERE {where_clause};"
        return self.execute_query(query, where_params)
    
    def ensure_admin_role_column(self):
        """Add role column to users table if it doesn't exist"""
        try:
            # Check if role column exists
            query = """
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'users' AND column_name = 'role' AND table_schema = 'public';
            """
            result = self.execute_query(query)
            
            if not result:
                # Add role column
                self.execute_query("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'regular';")
                logger.info("✅ Added role column to users table")
                
                # Set admin role for admin email
                self.execute_query(
                    "UPDATE users SET role = 'admin' WHERE email = %s;", 
                    (ADMIN_EMAIL,)
                )
                logger.info(f"✅ Set admin role for {ADMIN_EMAIL}")
            else:
                logger.info("ℹ️ Role column already exists in users table")
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure admin role column: {str(e)}")

    def ensure_ai_sources_foreign_keys(self):
        """Add foreign key relationships for ai_sources table"""
        try:
            # Check if ai_topics_id column exists in ai_sources
            query = """
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'ai_sources' AND column_name = 'ai_topics_id' AND table_schema = 'public';
            """
            result = self.execute_query(query)
            
            if not result:
                # Add ai_topics_id column
                self.execute_query("ALTER TABLE ai_sources ADD COLUMN ai_topics_id INTEGER;")
                logger.info("✅ Added ai_topics_id column to ai_sources table")
                
                # Add foreign key constraint for ai_topics_id
                try:
                    self.execute_query("""
                        ALTER TABLE ai_sources 
                        ADD CONSTRAINT fk_ai_sources_ai_topics_id 
                        FOREIGN KEY (ai_topics_id) REFERENCES ai_topics(id) ON DELETE SET NULL;
                    """)
                    logger.info("✅ Added foreign key constraint for ai_topics_id")
                except Exception as fk_error:
                    logger.warning(f"⚠️ Could not add foreign key constraint for ai_topics_id: {str(fk_error)}")
            else:
                logger.info("ℹ️ ai_topics_id column already exists in ai_sources table")
            
            # Check if we need to add foreign key constraint for category field
            try:
                # First check if the constraint already exists
                constraint_query = """
                SELECT constraint_name FROM information_schema.table_constraints 
                WHERE table_name = 'ai_sources' AND constraint_type = 'FOREIGN KEY' 
                AND constraint_name = 'fk_ai_sources_category';
                """
                constraint_exists = self.execute_query(constraint_query)
                
                if not constraint_exists:
                    # Add foreign key constraint for category field
                    self.execute_query("""
                        ALTER TABLE ai_sources 
                        ADD CONSTRAINT fk_ai_sources_category 
                        FOREIGN KEY (category) REFERENCES ai_topics(category) ON DELETE SET NULL;
                    """)
                    logger.info("✅ Added foreign key constraint for category field")
                else:
                    logger.info("ℹ️ Category foreign key constraint already exists")
                    
            except Exception as cat_fk_error:
                logger.warning(f"⚠️ Could not add category foreign key constraint: {str(cat_fk_error)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure ai_sources foreign keys: {str(e)}")
            
    def get_foreign_key_options(self, table_name: str, column_name: str):
        """Get foreign key reference options for dropdown"""
        try:
            # Check if this column has foreign key constraints
            query = """
            SELECT 
                tc.constraint_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = %s 
                AND kcu.column_name = %s
                AND tc.table_schema = 'public';
            """
            fk_info = self.execute_query(query, (table_name, column_name))
            
            if fk_info:
                # Get the foreign key reference data
                foreign_table = fk_info[0]['foreign_table_name']
                foreign_column = fk_info[0]['foreign_column_name']
                
                # Fetch the reference data
                ref_query = f"SELECT {foreign_column}, " + \
                    ("category, description FROM " if foreign_table == 'ai_topics' else "name FROM ") + \
                    f"{foreign_table} ORDER BY {foreign_column};"
                
                return self.execute_query(ref_query)
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting foreign key options: {str(e)}")
            return []

class OTPManager:
    def __init__(self):
        self.otps = {}  # In production, use Redis or database
    
    def generate_otp(self, email: str) -> str:
        """Generate 6-digit OTP"""
        otp = str(secrets.randbelow(900000) + 100000)
        self.otps[email] = {
            'otp': otp,
            'created_at': datetime.now(),
            'verified': False
        }
        return otp
    
    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify OTP"""
        if email not in self.otps:
            return False
        
        stored_data = self.otps[email]
        
        # Check if OTP is expired (5 minutes)
        if datetime.now() - stored_data['created_at'] > timedelta(minutes=5):
            del self.otps[email]
            return False
        
        if stored_data['otp'] == otp:
            stored_data['verified'] = True
            return True
        
        return False
    
    def send_otp_email(self, email: str, otp: str) -> bool:
        """Send OTP via email (using Brevo API)"""
        try:
            if BREVO_API_KEY:
                # Use Brevo API for sending email
                import requests
                
                url = "https://api.brevo.com/v3/smtp/email"
                headers = {
                    "api-key": BREVO_API_KEY,
                    "Content-Type": "application/json"
                }
                
                data = {
                    "sender": {"email": "noreply@vidyagam.com", "name": "Vidyagam Admin"},
                    "to": [{"email": email}],
                    "subject": "Admin Login OTP - Vidyagam AI News",
                    "htmlContent": f"""
                    <html>
                    <body>
                        <h2>Admin Login OTP</h2>
                        <p>Your OTP for admin login is: <strong style="font-size: 24px; color: #007bff;">{otp}</strong></p>
                        <p>This OTP is valid for 5 minutes.</p>
                        <p>If you didn't request this, please ignore this email.</p>
                        <hr>
                        <small>Vidyagam AI News Admin Panel</small>
                    </body>
                    </html>
                    """
                }
                
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 201:
                    logger.info(f"✅ OTP email sent to {email}")
                    return True
                else:
                    logger.error(f"❌ Failed to send OTP email: {response.text}")
            
            # Fallback: Log OTP for testing
            logger.info(f"🔐 OTP for {email}: {otp}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send OTP: {str(e)}")
            logger.info(f"🔐 Fallback OTP for {email}: {otp}")
            return True

# Initialize managers
db_manager = DatabaseManager()
otp_manager = OTPManager()

@app.before_first_request
def initialize_database():
    """Initialize database with admin role column and foreign keys"""
    try:
        db_manager.ensure_admin_role_column()
        db_manager.ensure_ai_sources_foreign_keys()
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")

# Authentication routes
@app.route('/')
def index():
    """Main admin login page"""
    if 'admin_authenticated' in session:
        return redirect(url_for('dashboard'))
    return render_template('admin_login.html')

@app.route('/send_otp', methods=['POST'])
def send_otp():
    """Send OTP to admin email"""
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        
        if email != ADMIN_EMAIL:
            return jsonify({'success': False, 'message': 'Only admin@vidyagam.com is allowed'})
        
        # Generate and send OTP
        otp = otp_manager.generate_otp(email)
        otp_sent = otp_manager.send_otp_email(email, otp)
        
        return jsonify({
            'success': True, 
            'message': f'OTP sent to {email}',
            'debug_otp': otp if not BREVO_API_KEY else None  # Show OTP in response if no email service
        })
        
    except Exception as e:
        logger.error(f"❌ Send OTP failed: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to send OTP'})

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    """Verify OTP and authenticate admin"""
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        otp = data.get('otp', '').strip()
        
        if otp_manager.verify_otp(email, otp):
            session['admin_authenticated'] = True
            session['admin_email'] = email
            return jsonify({'success': True, 'message': 'Admin authenticated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Invalid or expired OTP'})
            
    except Exception as e:
        logger.error(f"❌ Verify OTP failed: {str(e)}")
        return jsonify({'success': False, 'message': 'OTP verification failed'})

@app.route('/logout')
def logout():
    """Logout admin"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

# Admin dashboard routes
@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    if 'admin_authenticated' not in session:
        return redirect(url_for('index'))
    
    try:
        tables = db_manager.get_tables()
        table_stats = []
        
        for table in tables:
            table_name = table['table_name']
            count = db_manager.get_table_count(table_name)
            table_stats.append({'name': table_name, 'count': count})
        
        return render_template('admin_dashboard.html', 
                             tables=table_stats, 
                             admin_email=session.get('admin_email'))
    except Exception as e:
        logger.error(f"❌ Dashboard error: {str(e)}")
        flash(f'Dashboard error: {str(e)}', 'error')
        return render_template('admin_dashboard.html', tables=[], admin_email=session.get('admin_email'))

@app.route('/table/<table_name>')
def view_table(table_name):
    """View table data with pagination"""
    if 'admin_authenticated' not in session:
        return redirect(url_for('index'))
    
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        offset = (page - 1) * per_page
        
        # Get table structure and data
        structure = db_manager.get_table_structure(table_name)
        data = db_manager.get_table_data(table_name, per_page, offset)
        total_count = db_manager.get_table_count(table_name)
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        
        return render_template('table_view.html',
                             table_name=table_name,
                             structure=structure,
                             data=data,
                             page=page,
                             per_page=per_page,
                             total_count=total_count,
                             total_pages=total_pages)
    except Exception as e:
        logger.error(f"❌ Table view error: {str(e)}")
        flash(f'Error viewing table {table_name}: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/download_table/<table_name>')
def download_table(table_name):
    """Download table data as CSV"""
    if 'admin_authenticated' not in session:
        return redirect(url_for('index'))
    
    try:
        # Get all data from table
        query = f"SELECT * FROM {table_name} ORDER BY 1;"
        data = db_manager.execute_query(query)
        
        if not data:
            flash(f'No data found in table {table_name}', 'warning')
            return redirect(url_for('view_table', table_name=table_name))
        
        # Convert to DataFrame and then to CSV
        df = pd.DataFrame(data)
        
        # Create CSV file in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Convert to bytes
        csv_data = io.BytesIO()
        csv_data.write(output.getvalue().encode('utf-8'))
        csv_data.seek(0)
        
        filename = f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return send_file(
            csv_data,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        flash(f'Error downloading {table_name}: {str(e)}', 'error')
        return redirect(url_for('view_table', table_name=table_name))

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Upload CSV data to table"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        table_name = request.form.get('table_name')
        file = request.files.get('file')
        
        if not file or not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'message': 'Please upload a valid CSV file'})
        
        # Read CSV file
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'success': False, 'message': 'CSV file is empty'})
        
        # Get table structure to validate columns
        structure = db_manager.get_table_structure(table_name)
        table_columns = [col['column_name'] for col in structure]
        csv_columns = list(df.columns)
        
        # Check if CSV columns match table columns (excluding auto-generated ones)
        auto_columns = ['id', 'created_at', 'updated_at']
        required_columns = [col for col in table_columns if col not in auto_columns]
        
        missing_columns = set(required_columns) - set(csv_columns)
        if missing_columns:
            return jsonify({
                'success': False, 
                'message': f'Missing required columns: {", ".join(missing_columns)}'
            })
        
        # Insert data row by row
        inserted_count = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Prepare data dictionary (exclude auto columns)
                data = {}
                for col in csv_columns:
                    if col in table_columns and col not in auto_columns:
                        value = row[col]
                        data[col] = None if pd.isna(value) else value
                
                db_manager.insert_record(table_name, data)
                inserted_count += 1
                
            except Exception as e:
                errors.append(f"Row {index + 1}: {str(e)}")
                if len(errors) > 10:  # Limit error reporting
                    errors.append("... and more errors")
                    break
        
        if inserted_count > 0:
            message = f'Successfully inserted {inserted_count} records'
            if errors:
                message += f' with {len(errors)} errors'
            return jsonify({'success': True, 'message': message, 'errors': errors[:5]})
        else:
            return jsonify({'success': False, 'message': 'No records were inserted', 'errors': errors[:5]})
            
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'})

@app.route('/api/record/<table_name>', methods=['POST'])
def add_record(table_name):
    """Add new record to table"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        data = request.json
        
        # Remove empty strings and convert to None for NULL values
        clean_data = {}
        for key, value in data.items():
            if value == '' or value is None:
                clean_data[key] = None
            else:
                clean_data[key] = value
        
        # Add timestamp if table has created_at column
        structure = db_manager.get_table_structure(table_name)
        table_columns = [col['column_name'] for col in structure]
        
        if 'created_at' in table_columns:
            clean_data['created_at'] = datetime.now()
        
        rows_affected = db_manager.insert_record(table_name, clean_data)
        
        return jsonify({
            'success': True, 
            'message': f'Record added successfully to {table_name}',
            'rows_affected': rows_affected
        })
        
    except Exception as e:
        logger.error(f"❌ Add record error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/record/<table_name>/<record_id>', methods=['PUT'])
def update_record(table_name, record_id):
    """Update existing record"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        data = request.json
        
        # Remove ID from data if present
        data.pop('id', None)
        
        # Add updated_at timestamp if column exists
        structure = db_manager.get_table_structure(table_name)
        table_columns = [col['column_name'] for col in structure]
        
        if 'updated_at' in table_columns:
            data['updated_at'] = datetime.now()
        
        # Clean data (empty strings to None)
        clean_data = {}
        for key, value in data.items():
            if value == '':
                clean_data[key] = None
            else:
                clean_data[key] = value
        
        rows_affected = db_manager.update_record(
            table_name, 
            clean_data, 
            'id = %s', 
            (record_id,)
        )
        
        return jsonify({
            'success': True, 
            'message': f'Record updated successfully in {table_name}',
            'rows_affected': rows_affected
        })
        
    except Exception as e:
        logger.error(f"❌ Update record error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/record/<table_name>/<record_id>', methods=['DELETE'])
def delete_record(table_name, record_id):
    """Delete record from table"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        rows_affected = db_manager.delete_record(table_name, 'id = %s', (record_id,))
        
        return jsonify({
            'success': True, 
            'message': f'Record deleted successfully from {table_name}',
            'rows_affected': rows_affected
        })
        
    except Exception as e:
        logger.error(f"❌ Delete record error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/records/<table_name>', methods=['DELETE'])
def delete_multiple_records(table_name):
    """Delete multiple records from table"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        data = request.json
        record_ids = data.get('ids', [])
        
        if not record_ids:
            return jsonify({'success': False, 'message': 'No records selected'})
        
        # Create placeholders for IN clause
        placeholders = ','.join(['%s'] * len(record_ids))
        where_clause = f'id IN ({placeholders})'
        
        rows_affected = db_manager.delete_record(table_name, where_clause, tuple(record_ids))
        
        return jsonify({
            'success': True, 
            'message': f'Deleted {rows_affected} records from {table_name}',
            'rows_affected': rows_affected
        })
        
    except Exception as e:
        logger.error(f"❌ Bulk delete error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/record/<table_name>/<record_id>', methods=['GET'])
def get_record(table_name, record_id):
    """Get single record for editing"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        query = f"SELECT * FROM {table_name} WHERE id = %s;"
        record = db_manager.execute_query(query, (record_id,), fetch_all=False)
        
        if record:
            # Convert datetime objects to strings for JSON serialization
            serialized_record = {}
            for key, value in record.items():
                if isinstance(value, datetime):
                    serialized_record[key] = value.isoformat()
                else:
                    serialized_record[key] = value
            
            return jsonify({'success': True, 'record': serialized_record})
        else:
            return jsonify({'success': False, 'message': 'Record not found'})
            
    except Exception as e:
        logger.error(f"❌ Get record error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/foreign_key_options/<table_name>/<column_name>')
def get_foreign_key_options(table_name, column_name):
    """Get foreign key reference options for dropdown"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        options = db_manager.get_foreign_key_options(table_name, column_name)
        return jsonify({'success': True, 'options': options})
    except Exception as e:
        logger.error(f"❌ Foreign key options error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/table_structure/<table_name>')
def get_table_structure_with_fk(table_name):
    """Get table structure with foreign key information"""
    if 'admin_authenticated' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        structure = db_manager.get_table_structure(table_name)
        
        # Enhance structure with foreign key information
        enhanced_structure = []
        for column in structure:
            column_info = dict(column)
            
            # Check if this column has foreign keys
            fk_options = db_manager.get_foreign_key_options(table_name, column['column_name'])
            if fk_options:
                column_info['has_foreign_key'] = True
                column_info['foreign_key_options'] = fk_options
            else:
                column_info['has_foreign_key'] = False
                
            enhanced_structure.append(column_info)
        
        return jsonify({'success': True, 'structure': enhanced_structure})
    except Exception as e:
        logger.error(f"❌ Table structure error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Starting AI News Scraper Admin Interface...")
    print(f"📧 Admin email: {ADMIN_EMAIL}")
    print(f"🔗 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)