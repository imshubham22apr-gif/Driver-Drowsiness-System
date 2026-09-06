import cv2
import time
import numpy as np

import config
from utils.geometric_metrics import eye_aspect_ratio, mouth_aspect_ratio
from utils.audio_alert import AudioAlert
from modules.face_mesh_detector import FaceMeshDetector
from modules.head_pose_estimator import HeadPoseEstimator
from modules.drowsiness_evaluator import DrowsinessEvaluator
from ui.dashboard import Dashboard

def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 0 or height == 0:
        print("Failed to open camera.")
        return

    face_detector = FaceMeshDetector()
    pose_estimator = HeadPoseEstimator((height, width))
    evaluator = DrowsinessEvaluator()
    dashboard = Dashboard()
    alert = AudioAlert()
    
    is_muted = False
    calibrating = False
    calibration_start_time = 0
    calib_ear_vals = []
    calib_mar_vals = []

    print("Driver Drowsiness System started.")
    print("Press 'q' to quit")
    print("Press 'c' to calibrate")
    print("Press 'm' to toggle mute")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frame = cv2.flip(frame, 1)
        landmarks = face_detector.process(frame)
        
        ear = 0.0
        mar = 0.0
        pitch, yaw, roll = 0.0, 0.0, 0.0
        perclos = 0.0
        status = "NORMAL"
        
        if landmarks:
            left_ear = eye_aspect_ratio(landmarks['left_eye'])
            right_ear = eye_aspect_ratio(landmarks['right_eye'])
            ear = (left_ear + right_ear) / 2.0
            
            mar = mouth_aspect_ratio(landmarks['mouth'])
            pitch, yaw, roll = pose_estimator.estimate(landmarks['head_pose'])
            
            if calibrating:
                curr_time = time.time()
                if curr_time - calibration_start_time <= 3.0:
                    calib_ear_vals.append(ear)
                    calib_mar_vals.append(mar)
                    cv2.putText(frame, "CALIBRATING... Keep face neutral", (50, int(height/2)), 
                                config.FONT, 1, config.COLOR_YELLOW, 2)
                else:
                    calibrating = False
                    if calib_ear_vals and calib_mar_vals:
                        avg_ear = np.mean(calib_ear_vals)
                        avg_mar = np.mean(calib_mar_vals)
                        evaluator.calibrate(avg_ear, avg_mar)
                        print(f"Calibrated! New EAR Thresh: {evaluator.ear_thresh:.2f}, MAR Thresh: {evaluator.mar_thresh:.2f}")
            else:
                status, perclos = evaluator.evaluate(ear, mar, pitch, yaw)
                
                if status == "CRITICAL_DROWSY" and not is_muted:
                    alert.play()
                else:
                    alert.stop()
        else:
            status = "NO_FACE"
            alert.stop()
            
        frame = dashboard.draw(frame, ear, mar, pitch, yaw, perclos, status, landmarks)
        
        if is_muted:
            cv2.putText(frame, "MUTED", (width - 100, height - 20), config.FONT, 0.7, config.COLOR_RED, 2)
            
        cv2.imshow("Driver Drowsiness & Distraction Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrating = True
            calibration_start_time = time.time()
            calib_ear_vals = []
            calib_mar_vals = []
            print("Calibration started for 3 seconds...")
        elif key == ord('m'):
            is_muted = not is_muted
            if is_muted:
                alert.stop()
            print("Audio Muted" if is_muted else "Audio Unmuted")

    cap.release()
    cv2.destroyAllWindows()
    alert.stop()

if __name__ == "__main__":
    main()
