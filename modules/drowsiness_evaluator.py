import time
from collections import deque
import config

class DrowsinessEvaluator:
    def __init__(self):
        self.ear_thresh = config.EAR_THRESHOLD
        self.mar_thresh = config.MAR_THRESHOLD
        
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.distraction_start_time = None
        
        self.perclos_history = deque()
        self.yawn_history = deque()
        
    def calibrate(self, ear_val, mar_val):
        """Calibrate thresholds based on driver's natural face shape."""
        self.ear_thresh = max(0.18, ear_val - 0.05)
        self.mar_thresh = min(0.8, mar_val + 0.1)
        
    def evaluate(self, ear, mar, pitch, yaw):
        current_time = time.monotonic()
        
        # PERCLOS Tracking
        is_closed = ear < self.ear_thresh
        self.perclos_history.append((current_time, is_closed))
        
        while self.perclos_history and (current_time - self.perclos_history[0][0]) > config.PERCLOS_WINDOW_SEC:
            self.perclos_history.popleft()
            
        perclos_percentage = sum(1 for _, c in self.perclos_history if c) / max(1, len(self.perclos_history))
        
        # Clean up old yawns (2 minutes window)
        while self.yawn_history and (current_time - self.yawn_history[0]) > 120.0:
            self.yawn_history.popleft()
            
        status = "NORMAL"
        
        # Eyes closure tracking
        if is_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
        else:
            self.eye_closed_start_time = None
            
        # Yawning tracking
        is_yawning = mar > self.mar_thresh
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
        else:
            if self.yawn_start_time is not None:
                duration = current_time - self.yawn_start_time
                if duration > config.MAR_CONSEC_SEC:
                    self.yawn_history.append(current_time)
                self.yawn_start_time = None

        # Distraction tracking (Look away or pitch down heavily)
        is_distracted = (abs(yaw) > config.HEAD_YAW_THRESHOLD) or (pitch < config.HEAD_PITCH_DOWN_THRESHOLD)
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
        else:
            self.distraction_start_time = None
            
        # Determine current status hierarchically (overrides from lowest to highest priority)
        
        if self.distraction_start_time is not None and (current_time - self.distraction_start_time) > 2.5:
            status = "DISTRACTED"
            
        if (self.yawn_start_time is not None and (current_time - self.yawn_start_time) > config.MAR_CONSEC_SEC) or len(self.yawn_history) >= 3:
            status = "DROWSY_YAWN"
            
        if self.eye_closed_start_time is not None and (current_time - self.eye_closed_start_time) > config.EAR_CONSEC_SEC:
            status = "CRITICAL_DROWSY"
        
        if perclos_percentage > config.PERCLOS_THRESHOLD:
            status = "CRITICAL_DROWSY"
            
        # Microsleep
        if is_closed and (pitch < config.HEAD_PITCH_DOWN_THRESHOLD):
            if self.eye_closed_start_time is not None and (current_time - self.eye_closed_start_time) > 1.0:
                status = "CRITICAL_DROWSY"
                
        return status, perclos_percentage
