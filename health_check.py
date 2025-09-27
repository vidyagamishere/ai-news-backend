#!/usr/bin/env python3
"""
Simple health check script to test if the FastAPI app is running
"""
import os
import time
import requests
import sys

def test_health_check():
    port = os.environ.get('PORT', '8000')
    url = f"http://localhost:{port}/health"
    
    print(f"=== HEALTH CHECK DEBUG ===")
    print(f"Testing health endpoint: {url}")
    print(f"PORT environment variable: {port}")
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}/{max_attempts}")
            response = requests.get(url, timeout=5)
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("✅ Health check passed!")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health check failed with error: {e}")
            
        if attempt < max_attempts - 1:
            print("Waiting 2 seconds before retry...")
            time.sleep(2)
    
    print("❌ All health check attempts failed")
    return False

if __name__ == "__main__":
    success = test_health_check()
    sys.exit(0 if success else 1)