# save_portfolio.py
from src.agents.improved_research_agent import ImprovedResearchAgent

agent = ImprovedResearchAgent()

portfolio_topics = [
    "AI startup funding trends 2025",
    "Remote work technology market analysis",
    "Sustainable technology business opportunities",
    "E-commerce automation trends 2025",
    "Healthcare AI market analysis"
]

for topic in portfolio_topics:
    print(f"\n🔬 Researching: {topic}")
    result = agent.conduct_enhanced_research(topic, "medium")
    filepath = agent.save_enhanced_report(result)
    print(f"✅ Saved: {filepath}")

print("\n🎉 Portfolio complete! You now have 5 professional research reports.")