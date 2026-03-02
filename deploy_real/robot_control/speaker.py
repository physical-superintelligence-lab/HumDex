import pyttsx3

class Speaker:
    def __init__(self):
        # Initialize speech engine
        self.engine = pyttsx3.init()
        # Use English voice by default
        self.engine.setProperty('voice', self.get_english_voice())
        # Set speech rate, 1.0 is normal speed
        self.engine.setProperty('rate', 150)
    
    def get_english_voice(self):
        """Get English voice"""
        voices = self.engine.getProperty('voices')
        # Try to find English voice
        for voice in voices:
            # Usually English voice ID contains 'en'
            if 'en' in voice.id.lower():
                return voice.id
        # If not found, return default voice
        return voices[0].id if voices else None
    
    def speak(self, text):
        """Read out text"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def change_voice(self, gender='male'):
        """Change voice gender"""
        voices = self.engine.getProperty('voices')
        for voice in voices:
            # Determine gender by ID, this is a simple method, may not be 100% accurate
            if gender.lower() == 'male' and 'male' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                return True
            elif gender.lower() == 'female' and 'female' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                return True
        return False
    
    def adjust_speed(self, rate=150):
        """Adjust speech rate (default 150)"""
        self.engine.setProperty('rate', rate)
    
    def adjust_volume(self, volume=1.0):
        """Adjust volume (0.0 to 1.0)"""
        self.engine.setProperty('volume', volume)

# Usage example
if __name__ == "__main__":
    speaker = Speaker()
    
    # Basic usage
    speaker.speak("Get into teleop mode!")
    
    # Non-English text example
    speaker.speak("hello")
    
    # Adjust speed
    speaker.adjust_speed(120)  # Slower
    speaker.speak("This is speaking at a slower rate.")
    
    # Adjust volume
    speaker.adjust_volume(0.8)
    speaker.speak("This is speaking at a lower volume.")
    
    # Try to change voice gender
    if speaker.change_voice('female'):
        speaker.speak("Now I'm speaking with a female voice.")