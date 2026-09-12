import cv2
import time
import numpy as np

import config
from utils.geometric_metrics import eye_aspect_ratio, mouth_aspect_ratio
from utils.audio_alert import AudioAlert
from modules.face_mesh_detector import FaceMeshDetector
from modules.head_pose_estimator import HeadPoseEstimator
from modules.gaze_estimator import GazeEstimator
from modules.blink_dynamics import BlinkDynamicsProfiler
from modules.rppg_estimator import RPPGEstimator
from modules.cognitive_state_engine import CognitiveStateEngine
from ui.dashboard import Dashboard

def main():
    print("=" * 70)
    print("AUTOMOTIVE GRADE DRIVER MONITORING SYSTEM (DMS)")
    print("Euro NCAP 2023+ Protocol Compliance | ISO 26262 Tier-1 Architecture")
    print("=" * 70)
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 0 or height == 0:
        print("[ERROR] Failed to open camera stream. Check CAMERA_INDEX in config.py.")
        return

    # Initialize Automotive Modules
    face_detector = FaceMeshDetector(refine_landmarks=True)
    pose_estimator = HeadPoseEstimator((height, width))
    gaze_estimator = GazeEstimator()
    blink_profiler = BlinkDynamicsProfiler()
    rppg_estimator = RPPGEstimator()
    state_engine = CognitiveStateEngine()
    dashboard = Dashboard()
    alert_engine = AudioAlert()
    
    is_muted = False
    calibrating = False
    calibration_start_time = 0
    calib_ear_vals = []
    calib_mar_vals = []
    calib_iris_x_vals = []
    calib_iris_y_vals = []
    
    prev_frame_time = time.time()
    fps = 30.0

    print("\nControls:")
    print("  'q' -> Quit DMS")
    print("  'c' -> Calibrate Driver Neutral Baseline (Resting Gaze & EAR)")
    print("  'm' -> Toggle Audio Warning Mute")
    print("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab video frame.")
            break
            
        current_frame_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, current_frame_time - prev_frame_time))
        prev_frame_time = current_frame_time
        
        frame = cv2.flip(frame, 1)
        landmarks = face_detector.process(frame)
        
        ear = 0.0
        mar = 0.0
        pitch, yaw, roll = 0.0, 0.0, 0.0
        perclos = 0.0
        
        gaze_info = {
            'gaze_pitch': 0.0, 'gaze_yaw': 0.0,
            'cabin_zone': 'UNKNOWN', 'is_off_road': False,
            'continuous_off_road_sec': 0.0, 'cumulative_off_road_sec': 0.0
        }
        blink_info = {
            'duration_ms': 0.0, 'avr': 0.0,
            'blink_type': 'NOMINAL', 'closing_velocity': 0.0,
            'drowsy_blink_count': 0, 'recent_ears': []
        }
        rppg_info = {
            'bpm': 72.0, 'hrv_index': 50.0,
            'pulse_waveform': [], 'is_ready': False
        }
        
        alert_level = config.LEVEL_0_NOMINAL
        status_title = "ATTENTIVE & FOCUSED"
        status_detail = "Euro NCAP Nominal"
        
        if landmarks:
            # 1. Geometric Eye & Mouth Metrics
            left_ear = eye_aspect_ratio(landmarks['left_eye'])
            right_ear = eye_aspect_ratio(landmarks['right_eye'])
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks['mouth'])
            
            # 2. Head Pose Estimation (PnP 3D Model)
            pitch, yaw, roll = pose_estimator.estimate(landmarks['head_pose'])
            
            # 3. 3D Gaze Estimation & Cabin Attention Zoning
            gaze_pitch, gaze_yaw, cabin_zone, is_off_road, cont_off, cumul_off = gaze_estimator.estimate(
                landmarks.get('gaze_anchors'),
                landmarks.get('right_iris_center'),
                landmarks.get('left_iris_center'),
                pitch, yaw
            )
            gaze_info = {
                'gaze_pitch': gaze_pitch, 'gaze_yaw': gaze_yaw,
                'cabin_zone': cabin_zone, 'is_off_road': is_off_road,
                'continuous_off_road_sec': cont_off, 'cumulative_off_road_sec': cumul_off
            }
            
            # 4. Blink Velocity Profiling & Amplitude-Velocity Ratio (AVR)
            blink_info = blink_profiler.update(ear)
            
            # 5. Contactless Facial rPPG Heart Rate Monitoring
            bpm, hrv, pulse_wave, rppg_ready = rppg_estimator.update(frame, landmarks.get('forehead_roi'))
            rppg_info = {
                'bpm': bpm, 'hrv_index': hrv,
                'pulse_waveform': pulse_wave, 'is_ready': rppg_ready
            }
            
            # 6. Driver Neutral Baseline Calibration Routine
            if calibrating:
                curr_time = time.time()
                if curr_time - calibration_start_time <= 3.0:
                    calib_ear_vals.append(ear)
                    calib_mar_vals.append(mar)
                    
                    # Track iris baseline
                    anchors = landmarks.get('gaze_anchors', {})
                    r_c = landmarks.get('right_iris_center')
                    l_c = landmarks.get('left_iris_center')
                    if anchors and r_c and l_c:
                        r_w = np.linalg.norm(np.array(anchors['r_inner']) - np.array(anchors['r_outer']))
                        l_w = np.linalg.norm(np.array(anchors['l_outer']) - np.array(anchors['l_inner']))
                        rx = (r_c[0] - anchors['r_outer'][0]) / max(1.0, r_w)
                        lx = (l_c[0] - anchors['l_inner'][0]) / max(1.0, l_w)
                        calib_iris_x_vals.append((rx + lx) / 2.0)
                        
                        r_h = np.linalg.norm(np.array(anchors['r_bottom']) - np.array(anchors['r_top']))
                        l_h = np.linalg.norm(np.array(anchors['l_bottom']) - np.array(anchors['l_top']))
                        ry = (r_c[1] - anchors['r_top'][1]) / max(1.0, r_h)
                        ly = (l_c[1] - anchors['l_top'][1]) / max(1.0, l_h)
                        calib_iris_y_vals.append((ry + ly) / 2.0)
                        
                    cv2.putText(frame, "CALIBRATING RESTING NEUTRAL POSE...", 
                                (width // 2 - 240, height // 2), 
                                config.FONT, 0.8, config.COLOR_YELLOW, 2)
                else:
                    calibrating = False
                    if calib_ear_vals and calib_mar_vals:
                        avg_ear = np.mean(calib_ear_vals)
                        avg_mar = np.mean(calib_mar_vals)
                        state_engine.calibrate(avg_ear, avg_mar)
                        blink_profiler.calibrate(avg_ear)
                    if calib_iris_x_vals and calib_iris_y_vals:
                        gaze_estimator.calibrate(np.mean(calib_iris_x_vals), np.mean(calib_iris_y_vals))
                    print(f"[INFO] Calibrated! EAR Thresh: {state_engine.ear_thresh:.2f}, MAR Thresh: {state_engine.mar_thresh:.2f}")
            else:
                # 7. Euro NCAP Graduated Cognitive State Evaluation
                alert_level, status_title, status_detail, perclos = state_engine.evaluate(
                    ear, mar, pitch, yaw, gaze_info, blink_info, rppg_info
                )
                
                # 8. Graduated Acoustic Feedback
                alert_engine.trigger(alert_level, is_muted=is_muted)
        else:
            alert_level = config.LEVEL_1_ADVISORY
            status_title = "SYSTEM: NO DRIVER FACE DETECTED"
            status_detail = "Camera Occlusion or Driver Out of Frame"
            alert_engine.stop()

        # Render Automotive Cockpit HUD
        frame = dashboard.draw(
            frame, ear, mar, pitch, yaw, roll, perclos,
            alert_level, status_title, status_detail,
            gaze_info, blink_info, rppg_info, landmarks, fps=fps
        )
        
        if is_muted:
            cv2.putText(frame, "[MUTED]", (width - 120, height - 35), config.FONT, 0.55, config.COLOR_RED, 2)
            
        cv2.imshow("Automotive Driver Monitoring System (Euro NCAP DMS)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrating = True
            calibration_start_time = time.time()
            calib_ear_vals = []
            calib_mar_vals = []
            calib_iris_x_vals = []
            calib_iris_y_vals = []
            print("[INFO] Calibration initiated. Please look forward at the road with a neutral expression for 3 seconds...")
        elif key == ord('m'):
            is_muted = not is_muted
            if is_muted:
                alert_engine.stop()
            print(f"[INFO] Audio Alerts: {'MUTED' if is_muted else 'ACTIVE'}")

    cap.release()
    cv2.destroyAllWindows()
    alert_engine.stop()
    print("\n[INFO] DMS system safely terminated.")

if __name__ == "__main__":
    main()

