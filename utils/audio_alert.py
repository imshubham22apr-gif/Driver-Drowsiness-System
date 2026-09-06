import threading
import numpy as np
import pygame
import os

class AudioAlert:
    def __init__(self, sound_path="alarm.wav"):
        self.sound_path = sound_path
        self.is_playing = False
        
        pygame.mixer.init()
        
        if os.path.exists(self.sound_path):
            self.sound = pygame.mixer.Sound(self.sound_path)
        else:
            # Synthesize a beep sound using numpy
            sample_rate = 44100
            duration = 1.0
            frequency = 1000.0 # 1000 Hz beep
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(frequency * t * 2 * np.pi)
            
            # Apply fade in/out to avoid popping sounds
            fade_samples = int(sample_rate * 0.05)
            fade = np.linspace(0, 1, fade_samples)
            tone[:fade_samples] *= fade
            tone[-fade_samples:] *= fade[::-1]
            
            # Convert to 16-bit integer format for pygame
            audio = np.zeros((len(t), 2), dtype=np.int16)
            max_amp = 32767
            audio[:, 0] = tone * max_amp
            audio[:, 1] = tone * max_amp
            
            self.sound = pygame.sndarray.make_sound(audio)

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self):
        self.sound.play(-1) # Play on loop

    def stop(self):
        if self.is_playing:
            self.sound.stop()
            self.is_playing = False
