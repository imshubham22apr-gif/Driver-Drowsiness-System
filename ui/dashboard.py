import cv2
import config
import time

class Dashboard:
    def __init__(self):
        pass

    def draw(self, frame, ear, mar, pitch, yaw, perclos, status, landmarks):
        h, w, _ = frame.shape
        
        # Draw landmarks if available
        if landmarks:
            for eye_name in ['left_eye', 'right_eye']:
                pts = landmarks.get(eye_name, [])
                for p in pts:
                    cv2.circle(frame, p, 2, config.COLOR_GREEN, -1)
            
            mouth = landmarks.get('mouth', {})
            for name, p in mouth.items():
                cv2.circle(frame, p, 2, config.COLOR_YELLOW, -1)
        
        # Telemetry Bar (bottom)
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        
        telemetry_text_1 = f"EAR: {ear:.2f} | MAR: {mar:.2f} | PERCLOS: {perclos*100:.0f}%"
        cv2.putText(frame, telemetry_text_1, (10, h - 50), config.FONT, config.FONT_SCALE, config.COLOR_WHITE, 1)
        
        telemetry_text_2 = f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f}"
        cv2.putText(frame, telemetry_text_2, (10, h - 20), config.FONT, config.FONT_SCALE, config.COLOR_WHITE, 1)
        
        # Status Banner (top)
        color = config.COLOR_GREEN
        banner_text = "ACTIVE & FOCUSED"
        
        if status == "CRITICAL_DROWSY":
            color = config.COLOR_RED
            banner_text = "DANGER: DROWSINESS DETECTED - WAKE UP!"
        elif status == "DROWSY_YAWN":
            color = config.COLOR_YELLOW
            banner_text = "WARNING: FREQUENT YAWNING"
        elif status == "DISTRACTED":
            color = config.COLOR_YELLOW
            banner_text = "WARNING: DISTRACTED / LOOKING AWAY"
            
        # Flashing effect for critical
        is_flashing = False
        if status == "CRITICAL_DROWSY":
            if int(time.time() * 5) % 2 == 0:
                is_flashing = True
                
        bg_color = (0, 0, 0) if is_flashing else color
        cv2.rectangle(frame, (0, 0), (w, 50), bg_color, -1)
        
        text_size = cv2.getTextSize(banner_text, config.FONT, config.FONT_SCALE, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 35
        
        text_color = config.COLOR_WHITE if (color == config.COLOR_RED or color == config.COLOR_GREEN or is_flashing) else (0, 0, 0)
        if is_flashing and color == config.COLOR_RED:
            text_color = config.COLOR_RED
            
        cv2.putText(frame, banner_text, (text_x, text_y), config.FONT, config.FONT_SCALE, text_color, 2)
        
        return frame
