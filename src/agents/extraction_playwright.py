# Save this as src/agents/extraction_playwright.py
from playwright.sync_api import sync_playwright
import time

def extract_content_better(url: str) -> str:
    """Better extraction with Playwright"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Navigate with timeout
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for content
            page.wait_for_load_state('domcontentloaded')
            
            # Remove unwanted elements
            page.evaluate("""
                const elements = document.querySelectorAll('script, style, nav, header, footer, aside');
                elements.forEach(el => el.remove());
            """)
            
            # Get text content
            content = page.inner_text('body')
            
            browser.close()
            return content[:3000]  # First 3000 chars
            
        except Exception as e:
            browser.close()
            print(f"Extraction failed: {e}")
            return ""

# Test it
if __name__ == "__main__":
    # Install first: pip install playwright && playwright install chromium
    test_url = "https://techcrunch.com"
    content = extract_content_better(test_url)
    print(f"Extracted {len(content)} characters")