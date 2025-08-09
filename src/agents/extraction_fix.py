# src/agents/extraction_fix.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time

def extract_content_properly(url: str) -> str:
    """Better content extraction that handles redirects"""
    
    # Handle Bing redirect URLs
    if 'bing.com/ck/a' in url:
        # Extract actual URL from Bing tracking
        try:
            response = requests.get(url, allow_redirects=True, timeout=10)
            actual_url = response.url
        except:
            return ""
    else:
        actual_url = url
    
    # Now extract from real URL
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(actual_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:2000]  # Return first 2000 chars
    except Exception as e:
        print(f"Extraction failed: {e}")
        return ""

# Test it
test_url = "https://techcrunch.com"
content = extract_content_properly(test_url)
print(f"Extracted {len(content)} characters")