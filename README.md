# Driver Drowsiness and Distraction Detection System

A production-grade, real-time driver drowsiness and distraction detection system built in Python using MediaPipe and OpenCV.

## Features
- **MediaPipe Face Mesh:** Robust 468-point 3D facial landmark tracking.
- **Eye Aspect Ratio (EAR) & PERCLOS:** Tracks eye closure and fatigue rate over time.
- **Mouth Aspect Ratio (MAR):** Detects yawning.
- **Head Pose Estimation:** Identifies nodding (microsleep) and looking away (distraction).
- **Time-Based Tracking:** Independent of camera FPS.
- **Synthetic Audio Alert:** Generates a beep dynamically if an audio file is missing.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

### Controls
- `q`: Quit the application
- `c`: Calibrate EAR and MAR thresholds for the driver (3 seconds)
- `m`: Toggle audio mute
