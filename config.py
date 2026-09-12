import cv2

# ==============================================================================
# EURO NCAP 2023+ DRIVER MONITORING SYSTEM (DMS) COMPLIANCE STANDARDS
# ==============================================================================
# Euro NCAP Technical Bulletin: Visual distraction & drowsiness evaluation
MICROSLEEP_THRESHOLD_SEC = 1.0           # Eyelid closure >= 1000ms triggers Level 3 critical warning
LONG_DISTRACTION_THRESHOLD_SEC = 3.0     # Continuous off-road gaze > 3.0s triggers Level 3 critical warning
CAUTION_DISTRACTION_THRESHOLD_SEC = 2.0  # Continuous off-road gaze > 2.0s triggers Level 2 caution
CUMULATIVE_DISTRACTION_THRESHOLD_SEC = 10.0 # Cumulative off-road gaze >= 10s within 30s sliding window
CUMULATIVE_WINDOW_SEC = 30.0             # Sliding evaluation window for cumulative distraction

# PERCLOS Standards (Proportion of time eyes are >= 80% closed)
PERCLOS_WINDOW_SEC = 60.0
PERCLOS_THRESHOLD_MILD = 0.15            # Mild fatigue onset (Level 1 / 2)
PERCLOS_THRESHOLD_CRITICAL = 0.30        # Severe impairment (Level 3)

# ==============================================================================
# GEOMETRIC & APPEARANCE-BASED THRESHOLDS
# ==============================================================================
# Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)
EAR_THRESHOLD = 0.22
EAR_CONSEC_SEC = 1.0                     # Aligned with Euro NCAP 1.0s microsleep definition

MAR_THRESHOLD = 0.65
MAR_CONSEC_SEC = 2.5
YAWN_FATIGUE_COUNT = 2                   # Multiple yawns in 2-min window indicate fatigue

HEAD_PITCH_DOWN_THRESHOLD = -15.0
HEAD_YAW_THRESHOLD = 25.0

# ==============================================================================
# 3D GAZE ESTIMATION & CABIN VISUAL ATTENTION ZONING
# ==============================================================================
# Roadway Forward Field of View (FOV) angular boundaries (degrees)
ROAD_CENTER_YAW_RANGE = (-18.0, 18.0)
ROAD_CENTER_PITCH_RANGE = (-12.0, 15.0)

# In-Cabin Distraction Target Angles
PHONE_PITCH_THRESHOLD = -14.0            # Downward gaze towards smartphone / lap
CENTER_CONSOLE_YAW_THRESHOLD = 20.0      # Infotainment / center display glance
SIDE_MIRROR_YAW_THRESHOLD = 28.0         # Side view mirrors glance

# Weighting factors for eye-in-head gaze fusion: Gaze = Head_Pose + K * Eye_Vector
GAZE_HEAD_WEIGHT = 1.0
GAZE_IRIS_YAW_WEIGHT = 45.0              # Degrees multiplier for horizontal iris deviation
GAZE_IRIS_PITCH_WEIGHT = 40.0            # Degrees multiplier for vertical iris deviation
GAZE_SMOOTHING_ALPHA = 0.35              # Exponential moving average filter coefficient

# ==============================================================================
# BLINK DYNAMICS & AMPLITUDE-TO-VELOCITY RATIO (AVR)
# ==============================================================================
BLINK_NORMAL_MAX_DURATION_MS = 250.0     # Typical reflex blink: 100-200ms
BLINK_DROWSY_DURATION_MS = 350.0         # Slow, drowsy drooping blink: >350ms
AVR_FATIGUE_THRESHOLD = 8.0              # Elevated Amplitude-to-Velocity Ratio (Normal: 2-5, Drowsy: >15)
EAR_BUFFER_SIZE = 90                     # ~3.0 seconds buffer at 30 FPS for waveform & derivatives

# ==============================================================================
# CONTACTLESS rPPG (REMOTE PHOTOPLETHYSMOGRAPHY)
# ==============================================================================
RPPG_BUFFER_SIZE = 150                   # 5.0 seconds buffer at 30 FPS
RPPG_MIN_HR_BPM = 45.0                   # Physiological human cardiac minimum
RPPG_MAX_HR_BPM = 180.0                  # Physiological human cardiac maximum
RPPG_FPS_DEFAULT = 30.0

# ==============================================================================
# GRADUATED COGNITIVE ALERT SYSTEM (LEVEL 0 - 3)
# ==============================================================================
LEVEL_0_NOMINAL = 0                      # Driver focused & attentive
LEVEL_1_ADVISORY = 1                     # Attentive fatigue (elevated AVR / yawning / slow blinks)
LEVEL_2_CAUTION = 2                      # Distraction / moderate drowsiness (off-road 2s, cumulative >10s)
LEVEL_3_CRITICAL = 3                     # Immediate danger (Microsleep >= 1.0s or Long Distraction > 3.0s)

# ==============================================================================
# UI & HUD VISUAL STYLING
# ==============================================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
THICKNESS = 1

# Automotive HUD Color Palette (BGR)
COLOR_GREEN = (0, 230, 0)
COLOR_CYAN = (230, 215, 0)
COLOR_YELLOW = (0, 200, 255)             # Amber
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (40, 40, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK_GRAY = (35, 35, 35)
COLOR_PANEL_BG = (20, 20, 20)

# Hardware Settings
CAMERA_INDEX = 0

