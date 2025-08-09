# src/agents/improved_research_agent.py
"""
Enhanced research agent that bypasses Google search limitations
"""
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import json
import sys
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.core.working_ai_playground import AIPlayground
except ImportError:
    print("❌ Cannot import AIPlayground. Make sure it exists!")
    exit(1)

class ImprovedResearchAgent:
    """Enhanced research agent with multiple search strategies"""
    
    def __init__(self):
        self.ai = AIPlayground()
        self.session = requests.Session()
        
        # Rotate user agents to avoid detection
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        self.research_log = []
    
    def search_multiple_sources(self, query: str) -> list:
        """Search multiple sources to avoid Google blocking"""
        
        print(f"🔍 Multi-source search for: {query}")
        all_results = []
        
        # 1. Try DuckDuckGo (more search-friendly)
        duckduckgo_results = self.search_duckduckgo(query)
        all_results.extend(duckduckgo_results)
        
        # 2. Try Bing (backup)
        bing_results = self.search_bing(query)
        all_results.extend(bing_results)
        
        # 3. Try direct sources
        direct_results = self.search_direct_sources(query)
        all_results.extend(direct_results)
        
        return all_results[:10]  # Top 10 results
    
    def search_duckduckgo(self, query: str) -> list:
        """Search using DuckDuckGo (more automation-friendly)"""
        try:
            # Update user agent
            self.session.headers.update({
                'User-Agent': random.choice(self.user_agents)
            })
            
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=15)
            
            if response.status_code != 200:
                print(f"⚠️ DuckDuckGo returned {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find DuckDuckGo result containers
            result_containers = soup.find_all('div', class_='result')[:5]
            
            for container in result_containers:
                try:
                    # Extract title
                    title_elem = container.find('a', class_='result__a')
                    title = title_elem.get_text().strip() if title_elem else "No title"
                    
                    # Extract URL
                    url = title_elem.get('href') if title_elem else "No URL"
                    
                    # Extract snippet
                    snippet_elem = container.find('div', class_='result__snippet')
                    snippet = snippet_elem.get_text().strip() if snippet_elem else "No snippet"
                    
                    if title != "No title" and url != "No URL":
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet[:200],
                            'source': 'DuckDuckGo'
                        })
                        
                except Exception as e:
                    continue
            
            print(f"✅ DuckDuckGo: Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ DuckDuckGo search failed: {e}")
            return []
    
    def search_bing(self, query: str) -> list:
        """Search using Bing as backup"""
        try:
            self.session.headers.update({
                'User-Agent': random.choice(self.user_agents)
            })
            
            search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=15)
            
            if response.status_code != 200:
                print(f"⚠️ Bing returned {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find Bing result containers
            result_containers = soup.find_all('li', class_='b_algo')[:3]
            
            for container in result_containers:
                try:
                    # Extract title
                    title_elem = container.find('h2')
                    title = title_elem.get_text().strip() if title_elem else "No title"
                    
                    # Extract URL
                    link_elem = container.find('a')
                    url = link_elem.get('href') if link_elem else "No URL"
                    
                    # Extract snippet
                    snippet_elem = container.find('div', class_='b_caption')
                    snippet = snippet_elem.get_text().strip() if snippet_elem else "No snippet"
                    
                    if title != "No title" and url != "No URL":
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet[:200],
                            'source': 'Bing'
                        })
                        
                except Exception as e:
                    continue
            
            print(f"✅ Bing: Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Bing search failed: {e}")
            return []
    
    def search_direct_sources(self, query: str) -> list:
        """Search direct sources for specific topics"""
        
        direct_sources = []
        query_lower = query.lower()
        
        # AI/Tech related sources
        if any(term in query_lower for term in ['ai', 'artificial intelligence', 'tech', 'software', 'agents']):
            ai_sources = [
                f"https://venturebeat.com/search/?q={query.replace(' ', '+')}",
                f"https://techcrunch.com/search/{query.replace(' ', '+')}",
                "https://www.artificialintelligence-news.com/",
                "https://ainowinstitute.org/",
            ]
            
            for source_url in ai_sources[:2]:  # Limit to avoid overloading
                try:
                    response = self.session.get(source_url, timeout=10)
                    if response.status_code == 200:
                        direct_sources.append({
                            'title': f"Direct source: {source_url.split('/')[2]}",
                            'url': source_url,
                            'snippet': f"Direct search results for {query}",
                            'source': 'Direct'
                        })
                except:
                    continue
        
        return direct_sources
    
    def extract_webpage_content_enhanced(self, url: str) -> str:
        """Enhanced content extraction with better parsing"""
        try:
            print(f"📄 Extracting from: {url[:50]}...")
            
            # Use random user agent
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = self.session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Focus on main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
            
            # Return substantial content
            return clean_text[:4000]  # Increased limit
            
        except Exception as e:
            print(f"❌ Failed to extract from {url[:30]}: {e}")
            return ""
    
    def conduct_enhanced_research(self, topic: str, depth: str = "medium") -> dict:
        """Enhanced research with better search and analysis"""
        
        start_time = datetime.now()
        print(f"\n🔬 Enhanced research on: {topic}")
        print(f"📊 Depth level: {depth}")
        print("-" * 50)
        
        # Step 1: Generate better search queries
        query_prompt = f"""
        Create 3 specific, effective search queries for researching: "{topic}"
        
        Make them:
        1. Specific and focused
        2. Likely to find authoritative sources
        3. Include relevant keywords and terms
        4. Avoid overly complex phrases
        
        Return ONLY the queries, one per line, no numbering or formatting.
        """
        
        ai_queries = self.ai.business_advisor(query_prompt)
        
        # Clean up queries
        queries = []
        for line in ai_queries.split('\n'):
            clean_query = line.strip().strip('-*•').strip('"').strip("'").strip()
            if len(clean_query) > 5 and len(clean_query) < 100:
                queries.append(clean_query)
        
        if not queries:
            queries = [topic, f"{topic} trends", f"{topic} analysis 2025"]
        
        queries = queries[:3]  # Limit to 3
        print(f"🎯 Generated {len(queries)} search queries")
        
        # Step 2: Multi-source search
        all_results = []
        all_content = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n📋 Query {i}/{len(queries)}: {query}")
            
            search_results = self.search_multiple_sources(query)
            all_results.extend(search_results)
            
            # Extract content from promising results
            for result in search_results[:2]:  # Top 2 per query
                if self.is_url_accessible(result['url']):
                    content = self.extract_webpage_content_enhanced(result['url'])
                    if content and len(content) > 200:  # Only substantial content
                        all_content.append({
                            'url': result['url'],
                            'title': result['title'],
                            'content': content,
                            'source': result.get('source', 'Unknown')
                        })
            
            time.sleep(3)  # Longer delay to be respectful
        
        # Step 3: Enhanced analysis
        print(f"\n🧠 Analyzing {len(all_content)} sources...")
        
        if all_content:
            # Create structured analysis prompt
            sources_summary = []
            for i, content in enumerate(all_content[:5], 1):  # Limit to top 5
                sources_summary.append(f"Source {i} ({content['source']}):")
                sources_summary.append(f"Title: {content['title']}")
                sources_summary.append(f"Content: {content['content'][:800]}...")
                sources_summary.append("---")
            
            analysis_prompt = f"""
            Research Topic: {topic}
            
            I've gathered information from {len(all_content)} high-quality sources. Please provide a comprehensive analysis:
            
            {chr(10).join(sources_summary)}
            
            Please provide:
            1. **Key Findings**: The most important insights discovered
            2. **Market Trends**: Current trends and future projections
            3. **Opportunities**: Potential opportunities identified
            4. **Challenges**: Key challenges or obstacles
            5. **Recommendations**: Actionable next steps
            
            Be specific, data-driven, and actionable. Focus on insights that would be valuable for business or strategic decisions.
            """
            
            final_analysis = self.ai.business_advisor(analysis_prompt)
        else:
            # Fallback analysis if no sources found
            fallback_prompt = f"""
            Based on your knowledge, provide a comprehensive analysis of: {topic}
            
            Include:
            1. Current market state and trends
            2. Key players and technologies
            3. Future projections for 2025-2026
            4. Opportunities and challenges
            5. Strategic recommendations
            
            Be specific and actionable, focusing on business and strategic insights.
            """
            final_analysis = self.ai.business_advisor(fallback_prompt)
        
        # Step 4: Enhanced reporting
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        research_result = {
            'topic': topic,
            'start_time': start_time.isoformat(),
            'duration_minutes': round(duration, 2),
            'search_queries': queries,
            'sources_found': len(all_results),
            'content_extracted': len(all_content),
            'source_breakdown': {
                'DuckDuckGo': len([r for r in all_results if r.get('source') == 'DuckDuckGo']),
                'Bing': len([r for r in all_results if r.get('source') == 'Bing']),
                'Direct': len([r for r in all_results if r.get('source') == 'Direct']),
            },
            'successful_extractions': len([c for c in all_content if len(c['content']) > 500]),
            'analysis': final_analysis,
            'source_urls': [c['url'] for c in all_content],
            'raw_data': all_content
        }
        
        # Log research
        self.research_log.append(research_result)
        
        print(f"\n✅ Enhanced research completed in {duration:.1f} minutes")
        print(f"📊 Found {len(all_results)} results, extracted {len(all_content)} sources")
        print(f"🎯 Source breakdown: DuckDuckGo: {research_result['source_breakdown']['DuckDuckGo']}, "
              f"Bing: {research_result['source_breakdown']['Bing']}, "
              f"Direct: {research_result['source_breakdown']['Direct']}")
        
        return research_result
    
    def is_url_accessible(self, url: str) -> bool:
        """Check if URL is accessible and worth processing"""
        
        # Skip problematic domains
        blocked_domains = [
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'reddit.com', 'pinterest.com'
        ]
        
        for domain in blocked_domains:
            if domain in url.lower():
                return False
        
        # Check for valid URL structure
        if not url.startswith(('http://', 'https://')):
            return False
        
        return True
    
    def save_enhanced_report(self, research_result: dict, filename: str = None):
        """Save enhanced research report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_clean = "".join(c for c in research_result['topic'] if c.isalnum() or c in (' ', '-', '_'))[:30]
            filename = f"enhanced_research_{topic_clean.replace(' ', '_')}_{timestamp}.json"
        
        filepath = Path("data/research_reports") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(research_result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Enhanced research saved to: {filepath}")
        return filepath

def main():
    """Interactive enhanced research session"""
    agent = ImprovedResearchAgent()
    
    print("🤖 ENHANCED AUTONOMOUS RESEARCH AGENT")
    print("=" * 50)
    print("Multi-source search with DuckDuckGo, Bing, and direct sources")
    print("Enhanced content extraction and AI analysis")
    print()
    
    while True:
        topic = input("🔬 Enter research topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            break
        
        if not topic:
            continue
        
        try:
            # Enhanced research
            result = agent.conduct_enhanced_research(topic, depth="medium")
            
            # Display enhanced results
            print("\n" + "=" * 60)
            print("📋 ENHANCED RESEARCH REPORT")
            print("=" * 60)
            print(f"Topic: {result['topic']}")
            print(f"Duration: {result['duration_minutes']:.1f} minutes")
            print(f"Sources Found: {result['sources_found']}")
            print(f"Content Extracted: {result['content_extracted']}")
            print(f"Successful Extractions: {result['successful_extractions']}")
            
            # Source breakdown
            breakdown = result['source_breakdown']
            print(f"Source Breakdown: DuckDuckGo({breakdown['DuckDuckGo']}), "
                  f"Bing({breakdown['Bing']}), Direct({breakdown['Direct']})")
            
            print("\n📊 COMPREHENSIVE ANALYSIS:")
            print("-" * 40)
            print(result['analysis'])
            
            # Save option
            save_option = input("\n💾 Save this enhanced report? (y/n): ").lower().strip()
            if save_option in ['y', 'yes']:
                filepath = agent.save_enhanced_report(result)
                print(f"Report saved to: {filepath}")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n⏹️ Research interrupted by user")
        except Exception as e:
            print(f"❌ Enhanced research failed: {e}")
    
    print("\n👋 Enhanced research session ended!")

if __name__ == "__main__":
    main()