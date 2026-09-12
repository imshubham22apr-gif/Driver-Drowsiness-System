import threading
import numpy as np
import pygame
import time
import os

class AudioAlert:
    """
    Automotive Graduated Acoustic Warning Engine.
    Synthesizes discrete auditory warning signatures for each Euro NCAP alert stage:
      - Level 1 Advisory: Soft pleasant chime (subtle reminder)
      - Level 2 Caution: Intermittent warning tone (attention grabber)
      - Level 3 Critical: Urgent repeating alarm (immediate emergency intervention)
    """
    def __init__(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except Exception:
            pass
            
        self.is_playing_critical = False
        self.last_advisory_time = 0.0
        self.last_caution_time = 0.0
        
        # Synthesize audio signatures
        self.sound_advisory = self._synthesize_chime(freq1=660, freq2=880, duration=0.25, volume=0.4)
        self.sound_caution = self._synthesize_tone(freq=750, duration=0.35, pulse=True, volume=0.7)
        self.sound_critical = self._synthesize_tone(freq=1200, duration=0.60, pulse=True, volume=1.0)
        
    def _synthesize_tone(self, freq=1000.0, duration=0.5, pulse=False, volume=0.8):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        
        if pulse:
            # Amplitude modulation for pulsing urgency
            mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * 8.0 * t))
            tone = np.sin(2.0 * np.pi * freq * t) * mod
        else:
            tone = np.sin(2.0 * np.pi * freq * t)
            
        # Envelope shaping (fade in and fade out to avoid clicks)
        fade_len = int(sample_rate * 0.02)
        if len(tone) > 2 * fade_len:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            tone[:fade_len] *= fade_in
            tone[-fade_len:] *= fade_out
            
        audio = np.zeros((n_samples, 2), dtype=np.int16)
        max_amp = int(32767 * volume)
        audio[:, 0] = (tone * max_amp).astype(np.int16)
        audio[:, 1] = (tone * max_amp).astype(np.int16)
        
        try:
            return pygame.sndarray.make_sound(audio)
        except Exception:
            return None

    def _synthesize_chime(self, freq1=660, freq2=880, duration=0.25, volume=0.5):
        sample_rate = 44100
        n_half = int(sample_rate * (duration / 2.0))
        t1 = np.linspace(0, duration / 2.0, n_half, False)
        t2 = np.linspace(0, duration / 2.0, n_half, False)
        
        tone1 = np.sin(2.0 * np.pi * freq1 * t1)
        tone2 = np.sin(2.0 * np.pi * freq2 * t2)
        tone = np.concatenate([tone1, tone2])
        
        fade_len = int(sample_rate * 0.02)
        fade = np.linspace(1, 0, fade_len)
        tone[-fade_len:] *= fade
        
        audio = np.zeros((len(tone), 2), dtype=np.int16)
        max_amp = int(32767 * volume)
        audio[:, 0] = (tone * max_amp).astype(np.int16)
        audio[:, 1] = (tone * max_amp).astype(np.int16)
        
        try:
            return pygame.sndarray.make_sound(audio)
        except Exception:
            return None

    def trigger(self, alert_level, is_muted=False):
        """Dispatches graduated acoustic feedback based on current alert level."""
        if is_muted:
            self.stop()
            return
            
        now = time.monotonic()
        
        if alert_level == 3: # LEVEL 3 CRITICAL
            if not self.is_playing_critical:
                self.is_playing_critical = True
                if self.sound_critical:
                    self.sound_critical.play(-1) # Loop continuously
        elif alert_level == 2: # LEVEL 2 CAUTION
            self.stop_critical()
            # Pulse warning chime every 1.5 seconds
            if now - self.last_caution_time >= 1.5:
                self.last_caution_time = now
                if self.sound_caution:
                    self.sound_caution.play()
        elif alert_level == 1: # LEVEL 1 ADVISORY
            self.stop_critical()
            # Play soft chime every 4 seconds
            if now - self.last_advisory_time >= 4.0:
                self.last_advisory_time = now
                if self.sound_advisory:
                    self.sound_advisory.play()
        else: # LEVEL 0 NOMINAL
            self.stop()

    def play(self):
        """Backward compatible critical alert."""
        if not self.is_playing_critical:
            self.is_playing_critical = True
            if self.sound_critical:
                self.sound_critical.play(-1)

    def stop_critical(self):
        if self.is_playing_critical:
            if self.sound_critical:
                self.sound_critical.stop()
            self.is_playing_critical = False

    def stop(self):
        self.stop_critical()
        if self.sound_caution:
            self.sound_caution.stop()
        if self.sound_advisory:
            self.sound_advisory.stop()

