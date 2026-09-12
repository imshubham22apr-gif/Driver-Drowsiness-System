import time
import numpy as np
from collections import deque
import config

class GazeEstimator:
    """
    Appearance-Based 3D Gaze Estimator with Cabin Attention Zoning.
    Fuses Head Pose (Euler angles) with Iris Eyeball Orientation to determine
    the driver's true 3D Line-of-Sight (LOS) vector and cabin attention zone.
    """
    def __init__(self):
        # Filtered gaze angles
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.is_initialized = False
        
        # Neutral gaze baseline calibration offsets
        self.baseline_iris_x = 0.5
        self.baseline_iris_y = 0.5
        
        # Off-road tracking
        self.off_road_start_time = None
        self.continuous_off_road_sec = 0.0
        
        # 30-second sliding window for Euro NCAP cumulative distraction
        # Stores tuples of (timestamp, is_off_road)
        self.gaze_history = deque()
        self.cumulative_off_road_sec = 0.0
        
        self.last_update_time = time.monotonic()

    def calibrate(self, avg_iris_x, avg_iris_y):
        """Calibrate driver-specific resting neutral gaze."""
        self.baseline_iris_x = float(avg_iris_x)
        self.baseline_iris_y = float(avg_iris_y)

    def estimate(self, gaze_anchors, right_iris_center, left_iris_center, head_pitch, head_yaw, smooth=True):
        """
        Calculates 3D gaze vector and evaluates visual attention zone.
        
        Returns:
            gaze_pitch (float): Pitch of the gaze vector in degrees
            gaze_yaw (float): Yaw of the gaze vector in degrees
            cabin_zone (str): Visual attention zone ('ROAD_FORWARD', 'PHONE_DOWN', etc.)
            is_off_road (bool): True if gaze is outside roadway FOV
            continuous_off_road_sec (float): Continuous off-road dwell time
            cumulative_off_road_sec (float): Cumulative off-road seconds in 30s window
        """
        current_time = time.monotonic()
        dt = max(1e-4, current_time - self.last_update_time)
        self.last_update_time = current_time
        
        eye_yaw_deg = 0.0
        eye_pitch_deg = 0.0
        
        # Compute eye-in-head gaze if anchors are available
        if gaze_anchors and right_iris_center is not None and left_iris_center is not None:
            r_outer = np.array(gaze_anchors['r_outer'], dtype=float)
            r_inner = np.array(gaze_anchors['r_inner'], dtype=float)
            r_top = np.array(gaze_anchors['r_top'], dtype=float)
            r_bottom = np.array(gaze_anchors['r_bottom'], dtype=float)
            r_center = np.array(right_iris_center, dtype=float)
            
            l_inner = np.array(gaze_anchors['l_inner'], dtype=float)
            l_outer = np.array(gaze_anchors['l_outer'], dtype=float)
            l_top = np.array(gaze_anchors['l_top'], dtype=float)
            l_bottom = np.array(gaze_anchors['l_bottom'], dtype=float)
            l_center = np.array(left_iris_center, dtype=float)
            
            # Horizontal iris ratio (0.0: looking left, 1.0: looking right)
            r_width = np.linalg.norm(r_inner - r_outer)
            l_width = np.linalg.norm(l_outer - l_inner)
            
            r_ratio_x = (r_center[0] - r_outer[0]) / max(1.0, r_width) if r_width > 0 else 0.5
            l_ratio_x = (l_center[0] - l_inner[0]) / max(1.0, l_width) if l_width > 0 else 0.5
            avg_ratio_x = (r_ratio_x + l_ratio_x) / 2.0
            
            # Vertical iris ratio (0.0: looking up, 1.0: looking down)
            r_height = np.linalg.norm(r_bottom - r_top)
            l_height = np.linalg.norm(l_bottom - l_top)
            
            r_ratio_y = (r_center[1] - r_top[1]) / max(1.0, r_height) if r_height > 0 else 0.5
            l_ratio_y = (l_center[1] - l_top[1]) / max(1.0, l_height) if l_height > 0 else 0.5
            avg_ratio_y = (r_ratio_y + l_ratio_y) / 2.0
            
            # Deviation from calibrated resting baseline
            delta_x = avg_ratio_x - self.baseline_iris_x
            delta_y = avg_ratio_y - self.baseline_iris_y
            
            eye_yaw_deg = delta_x * config.GAZE_IRIS_YAW_WEIGHT
            # Negative y is looking downward
            eye_pitch_deg = -delta_y * config.GAZE_IRIS_PITCH_WEIGHT
            
        # Composite 3D Gaze Vector = Head Pose + Eye Vector
        raw_gaze_yaw = (head_yaw * config.GAZE_HEAD_WEIGHT) + eye_yaw_deg
        raw_gaze_pitch = (head_pitch * config.GAZE_HEAD_WEIGHT) + eye_pitch_deg
        
        # Exponential Moving Average (EMA) smoothing
        if not smooth or not self.is_initialized:
            self.smooth_yaw = raw_gaze_yaw
            self.smooth_pitch = raw_gaze_pitch
            self.is_initialized = True
        else:
            alpha = config.GAZE_SMOOTHING_ALPHA
            self.smooth_yaw = alpha * raw_gaze_yaw + (1.0 - alpha) * self.smooth_yaw
            self.smooth_pitch = alpha * raw_gaze_pitch + (1.0 - alpha) * self.smooth_pitch
            
        gaze_yaw = self.smooth_yaw
        gaze_pitch = self.smooth_pitch
        
        # Cabin Attention Zoning
        yaw_min, yaw_max = config.ROAD_CENTER_YAW_RANGE
        pitch_min, pitch_max = config.ROAD_CENTER_PITCH_RANGE
        
        is_in_road_cone = (yaw_min <= gaze_yaw <= yaw_max) and (pitch_min <= gaze_pitch <= pitch_max)
        
        if is_in_road_cone:
            cabin_zone = "ROAD_FORWARD"
            is_off_road = False
        elif gaze_pitch < config.PHONE_PITCH_THRESHOLD:
            cabin_zone = "PHONE_DOWN"
            is_off_road = True
        elif abs(gaze_yaw) > config.SIDE_MIRROR_YAW_THRESHOLD:
            cabin_zone = "SIDE_MIRRORS"
            is_off_road = True
        elif gaze_yaw > config.CENTER_CONSOLE_YAW_THRESHOLD:
            cabin_zone = "CENTER_CONSOLE"
            is_off_road = True
        else:
            cabin_zone = "OFF_ROAD"
            is_off_road = True
            
        # Continuous off-road dwell time tracking
        if is_off_road:
            if self.off_road_start_time is None:
                self.off_road_start_time = current_time
            self.continuous_off_road_sec = current_time - self.off_road_start_time
        else:
            self.off_road_start_time = None
            self.continuous_off_road_sec = 0.0
            
        # 30s Sliding window for cumulative distraction
        self.gaze_history.append((current_time, dt, is_off_road))
        
        # Prune entries older than 30s
        while self.gaze_history and (current_time - self.gaze_history[0][0]) > config.CUMULATIVE_WINDOW_SEC:
            self.gaze_history.popleft()
            
        # Cumulative duration spent off-road in current 30s window
        self.cumulative_off_road_sec = sum(duration for _, duration, off in self.gaze_history if off)
        
        return gaze_pitch, gaze_yaw, cabin_zone, is_off_road, self.continuous_off_road_sec, self.cumulative_off_road_sec
