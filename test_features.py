"""
Test all new EdgeMind features
"""

def test_streaming():
    print("\n📺 Testing Streaming...")
    from src.core.streaming_demo import stream_demo
    stream_demo()

def test_web_search():
    print("\n🔍 Testing Web Search...")
    from src.tools.web_search import search_enhanced_query
    search_enhanced_query("What is the weather today?")

def test_voice():
    print("\n🎤 Testing Voice (TTS only)...")
    import pyttsx3
    engine = pyttsx3.init()
    engine.say("EdgeMind voice test successful")
    engine.runAndWait()
    print("✅ Voice test complete")

if __name__ == "__main__":
    print("🧪 EdgeMind Feature Tests")
    print("=" * 60)
    
    # Test each feature
    try:
        test_streaming()
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
    
    try:
        test_web_search()
    except Exception as e:
        print(f"❌ Web search failed: {e}")
    
    try:
        test_voice()
    except Exception as e:
        print(f"❌ Voice failed: {e}")
    
    print("\n✅ Feature testing complete!")