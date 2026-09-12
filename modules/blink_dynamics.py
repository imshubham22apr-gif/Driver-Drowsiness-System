import time
import numpy as np
from collections import deque
import config

class BlinkDynamicsProfiler:
    """
    Analyzes dynamic eyelid velocity profiles and Amplitude-to-Velocity Ratio (AVR).
    Differentiates between healthy, sharp reflex blinks (100-200ms) and sluggish
    neuro-physiological fatigue blinks (>350ms, slow eyelid drooping).
    """
    def __init__(self, buffer_size=config.EAR_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.ear_history = deque(maxlen=buffer_size)      # Stores (timestamp, ear)
        self.velocity_history = deque(maxlen=buffer_size) # Stores (timestamp, dEAR/dt)
        
        # Blink state machine: 'OPEN', 'CLOSING', 'OPENING'
        self.state = 'OPEN'
        self.blink_start_time = None
        self.blink_trough_ear = 1.0
        self.blink_baseline_ear = 0.30
        self.max_closing_velocity = 0.0
        self.max_opening_velocity = 0.0
        
        # Telemetry metrics for HUD
        self.last_blink_duration_ms = 0.0
        self.last_avr = 0.0
        self.last_blink_type = "NOMINAL"
        self.blink_count = 0
        self.drowsy_blink_count = 0
        self.last_blink_time = 0.0
        
        # Baseline EAR calibration
        self.open_ear_baseline = config.EAR_THRESHOLD + 0.08
        self.close_ear_threshold = config.EAR_THRESHOLD

    def calibrate(self, neutral_ear):
        """Calibrate open eye baseline EAR."""
        self.open_ear_baseline = float(neutral_ear)
        self.close_ear_threshold = max(0.16, neutral_ear - 0.06)

    def update(self, ear):
        """
        Processes real-time EAR sample, calculates derivative dEAR/dt,
        and tracks blink kinematic metrics.
        
        Returns:
            dict containing:
                duration_ms: Duration of last completed blink in ms
                avr: Amplitude-to-Velocity Ratio of last blink
                closing_velocity: Current / peak closing speed
                blink_type: 'NORMAL_REFLEX' or 'DROWSY_DROOP'
                ear_history: Deque of normalized EAR values for wave graph
        """
        current_time = time.monotonic()
        
        # Calculate instantaneous derivative (dEAR / dt)
        derivative = 0.0
        if self.ear_history:
            prev_time, prev_ear = self.ear_history[-1]
            dt = current_time - prev_time
            if dt > 1e-4:
                derivative = (ear - prev_ear) / dt
                
        self.ear_history.append((current_time, ear))
        self.velocity_history.append((current_time, derivative))
        
        # State machine for blink kinematic tracking
        if self.state == 'OPEN':
            # Blink initiates when EAR drops below threshold
            if ear < self.close_ear_threshold:
                self.state = 'CLOSING'
                self.blink_start_time = current_time
                self.blink_baseline_ear = prev_ear if self.ear_history else self.open_ear_baseline
                self.blink_trough_ear = ear
                self.max_closing_velocity = max(0.1, -derivative)
                self.max_opening_velocity = 0.0
                
        elif self.state in ('CLOSING', 'OPENING'):
            # Track deepest eyelid trough and velocities
            if ear < self.blink_trough_ear:
                self.blink_trough_ear = ear
            if -derivative > self.max_closing_velocity:
                self.max_closing_velocity = -derivative
            if derivative > self.max_opening_velocity:
                self.max_opening_velocity = derivative
                
            if derivative > 0.2:
                self.state = 'OPENING'
                
            # Blink completion: EAR recovers to open threshold
            if ear >= (self.close_ear_threshold + 0.01):
                self.state = 'OPEN'
                if self.blink_start_time is not None:
                    duration_sec = current_time - self.blink_start_time
                    self.last_blink_duration_ms = duration_sec * 1000.0
                    self.last_blink_time = current_time
                    self.blink_count += 1
                    
                    # Amplitude-to-Velocity Ratio (AVR)
                    delta_ear = max(0.01, self.blink_baseline_ear - self.blink_trough_ear)
                    # Normalize AVR: amplitude divided by peak closing velocity (scaled)
                    self.last_avr = (delta_ear / max(0.05, self.max_closing_velocity)) * 100.0
                    
                    # Classification of blink kinematics
                    if self.last_blink_duration_ms >= config.BLINK_DROWSY_DURATION_MS or self.last_avr >= config.AVR_FATIGUE_THRESHOLD:
                        self.last_blink_type = "DROWSY_DROOP"
                        self.drowsy_blink_count += 1
                    else:
                        self.last_blink_type = "NORMAL_REFLEX"
                        
                self.blink_start_time = None
                
        # Safety reset if stuck in closing/opening (e.g., prolonged eye closure / microsleep)
        if self.blink_start_time is not None and (current_time - self.blink_start_time) > 4.0:
            self.state = 'OPEN'
            self.blink_start_time = None
            
        return {
            'duration_ms': self.last_blink_duration_ms,
            'avr': self.last_avr,
            'closing_velocity': self.max_closing_velocity,
            'opening_velocity': self.max_opening_velocity,
            'blink_type': self.last_blink_type,
            'blink_count': self.blink_count,
            'drowsy_blink_count': self.drowsy_blink_count,
            'recent_ears': [e for _, e in self.ear_history]
        }
