"""
Voice-enabled EdgeMind
Speak to your AI and hear responses
"""

import pyttsx3
import speech_recognition as sr
from src.core.edgemind import EdgeMind
import time

class VoiceEdgeMind:
    def __init__(self):
        # Initialize EdgeMind
        self.em = EdgeMind(verbose=False)
        
        # Initialize text-to-speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)  # Speed of speech
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
    def speak(self, text: str):
        """Convert text to speech"""
        print(f"🔊 Speaking: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen(self) -> str:
        """Listen for voice input"""
        with self.microphone as source:
            print("🎤 Listening... (speak now)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=5)
                print("🎧 Processing...")
                
                # Use Google's free speech recognition
                text = self.recognizer.recognize_google(audio)
                print(f"📝 You said: {text}")
                return text
                
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                print("❌ Could not understand audio")
                return ""
            except Exception as e:
                print(f"❌ Error: {e}")
                return ""
    
    def voice_chat(self):
        """Interactive voice chat"""
        self.speak("Hello! I'm EdgeMind. Say 'exit' to stop.")
        
        while True:
            # Listen for input
            user_input = self.listen()
            
            if not user_input:
                continue
                
            if 'exit' in user_input.lower():
                self.speak("Goodbye!")
                break
            
            # Generate response
            response = self.em.generate(user_input, max_tokens=50)
            
            # Speak response
            self.speak(response)
            
            time.sleep(0.5)  # Brief pause

if __name__ == "__main__":
    # Install requirements first:
    # pip install pyttsx3 SpeechRecognition pyaudio
    
    print("🎙️ EdgeMind Voice Assistant")
    print("=" * 50)
    
    assistant = VoiceEdgeMind()
    
    # Test TTS
    assistant.speak("Voice system initialized")
    
    # Start voice chat
    assistant.voice_chat()