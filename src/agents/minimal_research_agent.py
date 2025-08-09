# src/agents/minimal_research_agent.py
"""
Minimal autonomous research agent - works with basic requirements only
"""
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.core.working_ai_playground import AIPlayground
except ImportError:
    print("❌ Cannot import AIPlayground. Make sure it exists!")
    exit(1)

class MinimalResearchAgent:
    """Simplified research agent using basic web scraping"""
    
    def __init__(self):
        self.ai = AIPlayground()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.research_log = []
    
    def search_google_basic(self, query: str, num_results: int = 5) -> list:
        """Basic Google search using requests + BeautifulSoup"""
        print(f"🔍 Searching Google for: {query}")
        
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find search result containers
            search_containers = soup.find_all('div', class_='g')[:num_results]
            
            for container in search_containers:
                try:
                    # Extract title
                    title_elem = container.find('h3')
                    title = title_elem.get_text() if title_elem else "No title"
                    
                    # Extract URL
                    link_elem = container.find('a')
                    url = link_elem.get('href') if link_elem else "No URL"
                    
                    # Extract snippet
                    snippet_elem = container.find('span', {'data-result-type': True}) or container.find('div', class_='VwiC3b')
                    snippet = snippet_elem.get_text() if snippet_elem else "No snippet"
                    
                    if title != "No title" and url != "No URL":
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet[:200]  # Limit snippet length
                        })
                        
                except Exception as e:
                    continue
            
            print(f"✅ Found {len(results)} search results")
            return results
            
        except Exception as e:
            print(f"❌ Google search failed: {e}")
            return []
    
    def extract_webpage_content(self, url: str) -> str:
        """Extract text content from a webpage"""
        try:
            print(f"📄 Extracting content from: {url[:50]}...")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            return text[:3000]  # First 3000 characters
            
        except Exception as e:
            print(f"❌ Failed to extract from {url[:30]}: {e}")
            return ""
    
    def conduct_research(self, topic: str, depth: str = "basic") -> dict:
        """Conduct autonomous research on a topic"""
        
        start_time = datetime.now()
        print(f"\n🔬 Starting research on: {topic}")
        print(f"📊 Depth level: {depth}")
        print("-" * 50)
        
        # Step 1: Generate search queries using AI
        query_prompt = f"""
        Generate 3 good Google search queries to research this topic thoroughly: "{topic}"
        
        Make the queries specific and likely to return high-quality information.
        Return only the queries, one per line.
        """
        
        ai_queries = self.ai.business_advisor(query_prompt)
        
        # Extract queries from AI response
        queries = [q.strip().strip('-').strip('*').strip() 
                  for q in ai_queries.split('\n') 
                  if q.strip() and len(q.strip()) > 5][:3]
        
        if not queries:
            queries = [topic, f"{topic} analysis", f"{topic} trends 2025"]
        
        print(f"🎯 Generated {len(queries)} search queries")
        
        # Step 2: Search and collect information
        all_results = []
        all_content = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n📋 Query {i}/{len(queries)}: {query}")
            
            search_results = self.search_google_basic(query, num_results=3)
            all_results.extend(search_results)
            
            # Extract content from top results
            for result in search_results[:2]:  # Top 2 per query
                content = self.extract_webpage_content(result['url'])
                if content:
                    all_content.append({
                        'url': result['url'],
                        'title': result['title'],
                        'content': content
                    })
            
            time.sleep(2)  # Be polite to servers
        
        # Step 3: Analyze findings using AI
        print(f"\n🧠 Analyzing {len(all_content)} sources...")
        
        analysis_prompt = f"""
        Research Topic: {topic}
        
        I've gathered information from {len(all_content)} web sources. Please analyze and summarize:
        
        Sources:
        {json.dumps([{'title': c['title'], 'content': c['content'][:500]} for c in all_content], indent=2)}
        
        Provide a comprehensive analysis including:
        1. Key findings and insights
        2. Important trends or patterns
        3. Practical implications
        4. Areas for further research
        
        Be specific and actionable.
        """
        
        final_analysis = self.ai.business_advisor(analysis_prompt)
        
        # Step 4: Generate research report
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        research_result = {
            'topic': topic,
            'start_time': start_time.isoformat(),
            'duration_minutes': round(duration, 2),
            'search_queries': queries,
            'sources_found': len(all_results),
            'content_extracted': len(all_content),
            'source_urls': [c['url'] for c in all_content],
            'analysis': final_analysis,
            'raw_data': all_content
        }
        
        # Log the research
        self.research_log.append(research_result)
        
        print(f"\n✅ Research completed in {duration:.1f} minutes")
        print(f"📊 Analyzed {len(all_content)} sources")
        
        return research_result
    
    def save_research_report(self, research_result: dict, filename: str = None):
        """Save research to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.json"
        
        filepath = Path("data/research_reports") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(research_result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Research saved to: {filepath}")
        return filepath

def main():
    """Interactive research session"""
    agent = MinimalResearchAgent()
    
    print("🤖 MINIMAL AUTONOMOUS RESEARCH AGENT")
    print("=" * 50)
    print("This agent can research any topic using web search + AI analysis")
    print()
    
    while True:
        topic = input("🔬 Enter research topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            break
        
        if not topic:
            continue
        
        try:
            # Conduct research
            result = agent.conduct_research(topic)
            
            # Display results
            print("\n" + "=" * 60)
            print("📋 RESEARCH REPORT")
            print("=" * 60)
            print(f"Topic: {result['topic']}")
            print(f"Duration: {result['duration_minutes']} minutes")
            print(f"Sources: {result['content_extracted']} analyzed")
            print("\n📊 ANALYSIS:")
            print("-" * 40)
            print(result['analysis'])
            
            # Save report
            save_option = input("\n💾 Save this report? (y/n): ").lower().strip()
            if save_option in ['y', 'yes']:
                filepath = agent.save_research_report(result)
                print(f"Report saved to: {filepath}")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n⏹️ Research interrupted by user")
        except Exception as e:
            print(f"❌ Research failed: {e}")
    
    print("\n👋 Research session ended. Have a great day!")

if __name__ == "__main__":
    main()