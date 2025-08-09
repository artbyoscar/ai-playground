# test_research.py
"""
Quick test script to verify research functionality
"""
from src.agents.improved_research_agent import ImprovedResearchAgent

def test_research_with_better_topics():
    """Test with topics that work better with search engines"""
    
    agent = ImprovedResearchAgent()
    
    # Test topics that usually return good results
    test_topics = [
        "artificial intelligence market size 2025",
        "Python programming best practices",
        "climate change latest research",
        "startup funding trends 2025"
    ]
    
    print("🧪 TESTING ENHANCED RESEARCH AGENT")
    print("=" * 50)
    
    for topic in test_topics[:1]:  # Test just one for now
        print(f"\n📋 Testing: {topic}")
        print("-" * 40)
        
        result = agent.conduct_enhanced_research(
            topic=topic,
            depth="medium"
        )
        
        print(f"✅ Sources found: {result['sources_found']}")
        print(f"📄 Content extracted: {result['content_extracted']}")
        
        if result['content_extracted'] > 0:
            print("🎉 SUCCESS - Content extraction working!")
        else:
            print("⚠️ No content extracted - trying fallback analysis")
            
        # Show first 500 chars of analysis
        print("\n📊 Analysis preview:")
        print(result['analysis'][:500] + "...")
        
        # Optionally save
        save = input("\n💾 Save this report? (y/n): ")
        if save.lower() == 'y':
            filepath = agent.save_enhanced_report(result)
            print(f"Saved to: {filepath}")
    
    print("\n" + "=" * 50)
    print("✅ Research agent test complete!")

if __name__ == "__main__":
    test_research_with_better_topics()