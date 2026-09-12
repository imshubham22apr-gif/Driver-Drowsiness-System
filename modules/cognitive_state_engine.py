import time
from collections import deque
import config

class CognitiveStateEngine:
    """
    Automotive Driver Monitoring System (DMS) Cognitive State Engine.
    Fully compliant with Euro NCAP 2023+ DMS Assessment Protocols:
      - Microsleep Evaluation (>= 1.0s complete closure)
      - Long Distraction Evaluation (> 3.0s continuous gaze away)
      - Cumulative Distraction Evaluation (>= 10.0s in 30s sliding window)
      - Graduated Alert HMI Management (Level 0 - Nominal to Level 3 - Critical)
    """
    def __init__(self):
        self.ear_thresh = config.EAR_THRESHOLD
        self.mar_thresh = config.MAR_THRESHOLD
        
        # Temporal event timers
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        
        # Sliding history windows
        self.perclos_history = deque()
        self.yawn_history = deque()
        
        # Current graduated state
        self.alert_level = config.LEVEL_0_NOMINAL
        self.status_title = "ATTENTIVE & FOCUSED"
        self.status_detail = "Euro NCAP Nominal"
        self.active_violation = None
        
    def calibrate(self, neutral_ear, neutral_mar):
        """Calibrate dynamic thresholds based on individual facial ergonomics."""
        self.ear_thresh = max(0.17, neutral_ear - 0.05)
        self.mar_thresh = min(0.80, neutral_mar + 0.12)

    def evaluate(self, ear, mar, pitch, yaw, gaze_info, blink_info, rppg_info):
        """
        Multimodal fusion and state evaluation.
        
        Args:
            ear (float): Eye Aspect Ratio
            mar (float): Mouth Aspect Ratio
            pitch (float): Head pose pitch (degrees)
            yaw (float): Head pose yaw (degrees)
            gaze_info (dict): Output from GazeEstimator
            blink_info (dict): Output from BlinkDynamicsProfiler
            rppg_info (dict): Output from RPPGEstimator
            
        Returns:
            alert_level (int): 0 (Nominal), 1 (Advisory), 2 (Caution), 3 (Critical)
            status_title (str): High-level HUD message
            status_detail (str): Specific protocol compliance rationale
            perclos (float): Real-time PERCLOS proportion (0.0 to 1.0)
        """
        current_time = time.monotonic()
        
        # 1. PERCLOS Tracking (Proportion of closure in 60s sliding window)
        is_closed = ear < self.ear_thresh
        self.perclos_history.append((current_time, is_closed))
        
        while self.perclos_history and (current_time - self.perclos_history[0][0]) > config.PERCLOS_WINDOW_SEC:
            self.perclos_history.popleft()
            
        perclos = sum(1 for _, c in self.perclos_history if c) / max(1, len(self.perclos_history))
        
        # 2. Eye Closure & Microsleep Timer
        eye_closed_sec = 0.0
        if is_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            eye_closed_sec = current_time - self.eye_closed_start_time
        else:
            self.eye_closed_start_time = None
            
        # 3. Yawn Tracking
        is_yawning = mar > self.mar_thresh
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
        else:
            if self.yawn_start_time is not None:
                duration = current_time - self.yawn_start_time
                if duration >= config.MAR_CONSEC_SEC:
                    self.yawn_history.append(current_time)
                self.yawn_start_time = None
                
        # Clean yawns older than 2 minutes
        while self.yawn_history and (current_time - self.yawn_history[0]) > 120.0:
            self.yawn_history.popleft()
            
        # 4. Gaze Distraction Metrics
        continuous_off_road_sec = gaze_info.get('continuous_off_road_sec', 0.0)
        cumulative_off_road_sec = gaze_info.get('cumulative_off_road_sec', 0.0)
        cabin_zone = gaze_info.get('cabin_zone', 'ROAD_FORWARD')
        
        # 5. Blink Dynamics Metrics
        blink_type = blink_info.get('blink_type', 'NOMINAL')
        last_blink_ms = blink_info.get('duration_ms', 0.0)
        last_avr = blink_info.get('avr', 0.0)
        
        # ----------------------------------------------------------------------
        # STATE MACHINE RESOLUTION (Hierarchical: Level 3 > Level 2 > Level 1 > 0)
        # ----------------------------------------------------------------------
        
        # LEVEL 3: CRITICAL INTERVENTION (Euro NCAP Hard Violations)
        # Condition A: Microsleep (Continuous closure >= 1.0s)
        if eye_closed_sec >= config.MICROSLEEP_THRESHOLD_SEC:
            self.alert_level = config.LEVEL_3_CRITICAL
            self.status_title = "CRITICAL: MICROSLEEP DETECTED!"
            self.status_detail = f"Euro NCAP Eyelid Closure: {eye_closed_sec:.1f}s >= 1.0s"
            self.active_violation = "MICROSLEEP"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition B: Long Distraction (Continuous Off-Road Gaze > 3.0s)
        if continuous_off_road_sec >= config.LONG_DISTRACTION_THRESHOLD_SEC:
            self.alert_level = config.LEVEL_3_CRITICAL
            self.status_title = "CRITICAL: LONG DISTRACTION!"
            zone_desc = "Looking Down (Phone)" if cabin_zone == "PHONE_DOWN" else f"Zone: {cabin_zone}"
            self.status_detail = f"Off-Road {continuous_off_road_sec:.1f}s > 3.0s ({zone_desc})"
            self.active_violation = "LONG_DISTRACTION"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition C: Severe PERCLOS Impairment
        if perclos >= config.PERCLOS_THRESHOLD_CRITICAL:
            self.alert_level = config.LEVEL_3_CRITICAL
            self.status_title = "CRITICAL: SEVERE DROWSINESS!"
            self.status_detail = f"PERCLOS-80: {perclos*100:.0f}% >= 30%"
            self.active_violation = "SEVERE_PERCLOS"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition D: Head Nodding with Eyes Drooping
        if is_closed and (pitch < config.HEAD_PITCH_DOWN_THRESHOLD) and (eye_closed_sec >= 0.7):
            self.alert_level = config.LEVEL_3_CRITICAL
            self.status_title = "CRITICAL: HEAD NOD DROWSINESS!"
            self.status_detail = f"Head Pitch {pitch:.0f} deg + Eye Closure"
            self.active_violation = "HEAD_NOD"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # LEVEL 2: CAUTION (Early Distraction / Moderate Fatigue)
        # Condition A: Approaching Long Distraction (Off-Road > 2.0s)
        if continuous_off_road_sec >= config.CAUTION_DISTRACTION_THRESHOLD_SEC:
            self.alert_level = config.LEVEL_2_CAUTION
            self.status_title = "CAUTION: DISTRACTED - LOOK AT ROAD"
            self.status_detail = f"Off-Road Dwell: {continuous_off_road_sec:.1f}s ({cabin_zone})"
            self.active_violation = "EARLY_DISTRACTION"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition B: Cumulative Distraction in 30s Window (Euro NCAP >= 10.0s)
        if cumulative_off_road_sec >= config.CUMULATIVE_DISTRACTION_THRESHOLD_SEC:
            self.alert_level = config.LEVEL_2_CAUTION
            self.status_title = "CAUTION: FREQUENT GLANCES AWAY"
            self.status_detail = f"Cumulative Off-Road: {cumulative_off_road_sec:.1f}s / 30s"
            self.active_violation = "CUMULATIVE_DISTRACTION"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition C: Elevated PERCLOS
        if perclos >= config.PERCLOS_THRESHOLD_MILD:
            self.alert_level = config.LEVEL_2_CAUTION
            self.status_title = "CAUTION: FATIGUE PROGRESSING"
            self.status_detail = f"PERCLOS: {perclos*100:.0f}% (Threshold: 15%)"
            self.active_violation = "MILD_PERCLOS"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # LEVEL 1: ADVISORY (Neuro-Fatigue Onset / Early Warning)
        # Condition A: Sluggish Eyelid Droop (Blink Duration > 350ms or high AVR)
        if blink_type == "DROWSY_DROOP":
            self.alert_level = config.LEVEL_1_ADVISORY
            self.status_title = "ADVISORY: FATIGUE ONSET DETECTED"
            self.status_detail = f"Blink Droop: {last_blink_ms:.0f}ms (AVR: {last_avr:.1f})"
            self.active_violation = "DROWSY_BLINK"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # Condition B: Yawning detected
        if (self.yawn_start_time is not None and (current_time - self.yawn_start_time) >= config.MAR_CONSEC_SEC) or len(self.yawn_history) >= config.YAWN_FATIGUE_COUNT:
            self.alert_level = config.LEVEL_1_ADVISORY
            self.status_title = "ADVISORY: FREQUENT YAWNING"
            self.status_detail = f"Yawns in 2 min: {len(self.yawn_history)}"
            self.active_violation = "YAWNING"
            return self.alert_level, self.status_title, self.status_detail, perclos
            
        # LEVEL 0: NOMINAL
        self.alert_level = config.LEVEL_0_NOMINAL
        self.status_title = "NOMINAL: ATTENTIVE & FOCUSED"
        self.status_detail = f"Zone: {cabin_zone} | Road Forward"
        self.active_violation = None
        
        return self.alert_level, self.status_title, self.status_detail, perclos
