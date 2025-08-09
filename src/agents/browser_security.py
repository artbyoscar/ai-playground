SECURITY_CONFIG = {
    "blocked_domains": [
        # Social Media
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
        
        # Potentially Harmful
        "4chan.org", "8kun.top", "reddit.com/r/[harmful_subreddits]",
        
        # Financial/Shopping (to prevent accidental purchases)
        "amazon.com", "ebay.com", "paypal.com", "stripe.com",
        
        # Add your specific blocked domains here
    ],
    
    "blocked_actions": [
        "delete", "rm", "format", "shutdown", "reboot",
        "purchase", "buy", "order", "payment", "checkout",
        "download", "install", "execute", "run",
        "sudo", "admin", "password", "login"
    ],
    
    "allowed_domains": [
        # Research Sources
        "google.com", "scholar.google.com", "wikipedia.org",
        "github.com", "stackoverflow.com", "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov", "ieee.org", "acm.org",
        
        # News Sources
        "reuters.com", "bbc.com", "cnn.com", "npr.org",
        
        # Add your trusted domains here
    ]
}