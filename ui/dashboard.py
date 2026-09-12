import cv2
import numpy as np
import time
import config

class Dashboard:
    """
    Automotive Cockpit Head-Up Display (HUD) Dashboard.
    Visualizes real-time Euro NCAP compliance telemetry, 3D Gaze target reticle,
    dynamic EAR waveform oscilloscope, rPPG Heart Rate monitor, and graduated alerts.
    """
    def __init__(self):
        self.sparkline_w = 260
        self.sparkline_h = 50

    def draw(self, frame, ear, mar, pitch, yaw, roll, perclos, 
             alert_level, status_title, status_detail, 
             gaze_info, blink_info, rppg_info, landmarks, fps=30.0):
        h, w, _ = frame.shape
        
        # 1. Subtle Facial Annotations
        if landmarks:
            # Forehead rPPG ROI
            forehead_pts = landmarks.get('forehead_roi', [])
            if len(forehead_pts) >= 4:
                cv2.polylines(frame, [np.array(forehead_pts, dtype=np.int32)], True, (120, 100, 40), 1)
                
            # Eyelid contours
            eye_color = config.COLOR_RED if ear < config.EAR_THRESHOLD else config.COLOR_GREEN
            for eye_name in ['left_eye', 'right_eye']:
                pts = landmarks.get(eye_name, [])
                for p in pts:
                    cv2.circle(frame, p, 1, eye_color, -1)
                    
            # Iris Centers & Line-of-sight Reticles
            r_center = landmarks.get('right_iris_center')
            l_center = landmarks.get('left_iris_center')
            if r_center and l_center:
                cv2.circle(frame, r_center, 3, config.COLOR_CYAN, -1)
                cv2.circle(frame, l_center, 3, config.COLOR_CYAN, -1)
                
                # Draw subtle gaze vector ray from iris centers
                gaze_yaw = gaze_info.get('gaze_yaw', 0.0)
                gaze_pitch = gaze_info.get('gaze_pitch', 0.0)
                dx = int(gaze_yaw * 1.5)
                dy = int(-gaze_pitch * 1.5)
                cv2.line(frame, r_center, (r_center[0] + dx, r_center[1] + dy), config.COLOR_CYAN, 1)
                cv2.line(frame, l_center, (l_center[0] + dx, l_center[1] + dy), config.COLOR_CYAN, 1)

            # Mouth contour
            mouth = landmarks.get('mouth', {})
            for name, p in mouth.items():
                cv2.circle(frame, p, 1, config.COLOR_YELLOW, -1)
                
        # ----------------------------------------------------------------------
        # 2. TOP EURO NCAP GRADUATED STATUS BANNER
        # ----------------------------------------------------------------------
        banner_h = 55
        banner_color = config.COLOR_GREEN
        text_color = config.COLOR_WHITE
        flashing = False
        
        if alert_level == config.LEVEL_3_CRITICAL:
            banner_color = config.COLOR_RED
            if int(time.time() * 6) % 2 == 0:
                flashing = True
                banner_color = (0, 0, 0)
                text_color = config.COLOR_RED
        elif alert_level == config.LEVEL_2_CAUTION:
            banner_color = config.COLOR_YELLOW
            text_color = (0, 0, 0)
        elif alert_level == config.LEVEL_1_ADVISORY:
            banner_color = config.COLOR_CYAN
            text_color = (0, 0, 0)
        else:
            banner_color = (20, 110, 20) # Muted cockpit dark green
            text_color = config.COLOR_WHITE
            
        cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)
        
        # Primary Title
        cv2.putText(frame, status_title, (20, 28), config.FONT, 0.75, text_color, 2)
        
        # Secondary Euro NCAP Technical Detail
        cv2.putText(frame, status_detail, (20, 48), config.FONT, 0.48, text_color, 1)
        
        # FPS and Euro NCAP Badge (Right top)
        badge_text = f"FPS: {fps:.1f} | Euro NCAP 2023+"
        badge_sz = cv2.getTextSize(badge_text, config.FONT, 0.48, 1)[0]
        cv2.putText(frame, badge_text, (w - badge_sz[0] - 20, 32), config.FONT, 0.48, text_color, 1)
        
        # ----------------------------------------------------------------------
        # 3. 3D GAZE TARGET RETICLE HUD (Top-Right Overlay)
        # ----------------------------------------------------------------------
        reticle_box_sz = 140
        rx1 = w - reticle_box_sz - 15
        ry1 = banner_h + 15
        rx2 = rx1 + reticle_box_sz
        ry2 = ry1 + reticle_box_sz
        
        # HUD Panel Background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), config.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), config.COLOR_GRAY, 1)
        
        # HUD Heading
        cv2.putText(frame, "3D GAZE RETICLE", (rx1 + 10, ry1 + 16), config.FONT, 0.40, config.COLOR_GRAY, 1)
        
        # Center Windshield Roadway Box
        cx = rx1 + reticle_box_sz // 2
        cy = ry1 + reticle_box_sz // 2
        
        road_w = int((config.ROAD_CENTER_YAW_RANGE[1] - config.ROAD_CENTER_YAW_RANGE[0]) * 1.5)
        road_h = int((config.ROAD_CENTER_PITCH_RANGE[1] - config.ROAD_CENTER_PITCH_RANGE[0]) * 1.5)
        cv2.rectangle(frame, (cx - road_w//2, cy - road_h//2), (cx + road_w//2, cy + road_h//2), (60, 60, 60), 1)
        
        # Gaze Point Position
        gaze_yaw = gaze_info.get('gaze_yaw', 0.0)
        gaze_pitch = gaze_info.get('gaze_pitch', 0.0)
        cabin_zone = gaze_info.get('cabin_zone', 'ROAD_FORWARD')
        is_off_road = gaze_info.get('is_off_road', False)
        
        gx = int(cx + np.clip(gaze_yaw * 1.8, -reticle_box_sz//2 + 8, reticle_box_sz//2 - 8))
        gy = int(cy - np.clip(gaze_pitch * 1.8, -reticle_box_sz//2 + 8, reticle_box_sz//2 - 8))
        
        dot_color = config.COLOR_RED if is_off_road else config.COLOR_GREEN
        cv2.circle(frame, (gx, gy), 5, dot_color, -1)
        cv2.drawMarker(frame, (gx, gy), dot_color, cv2.MARKER_CROSS, 14, 1)
        
        # Cabin Zone Label below reticle
        zone_color = config.COLOR_GREEN if not is_off_road else config.COLOR_YELLOW
        cv2.putText(frame, f"Zone: {cabin_zone}", (rx1 + 8, ry2 - 8), config.FONT, 0.40, zone_color, 1)
        
        # ----------------------------------------------------------------------
        # 4. EURO NCAP DISTRACTION METERS (Right Side Overlay)
        # ----------------------------------------------------------------------
        meter_y = ry2 + 15
        meter_w = reticle_box_sz
        
        # Panel Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, meter_y), (rx2, meter_y + 110), config.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (rx1, meter_y), (rx2, meter_y + 110), config.COLOR_GRAY, 1)
        
        cv2.putText(frame, "EURO NCAP METERS", (rx1 + 8, meter_y + 16), config.FONT, 0.40, config.COLOR_GRAY, 1)
        
        # Meter 1: Continuous Off-Road
        off_sec = gaze_info.get('continuous_off_road_sec', 0.0)
        max_off = config.LONG_DISTRACTION_THRESHOLD_SEC # 3.0s
        ratio_off = min(1.0, off_sec / max_off)
        cv2.putText(frame, f"Off-Road: {off_sec:.1f}s / {max_off:.0f}s", (rx1 + 8, meter_y + 36), config.FONT, 0.40, config.COLOR_WHITE, 1)
        cv2.rectangle(frame, (rx1 + 8, meter_y + 42), (rx2 - 8, meter_y + 50), (40, 40, 40), -1)
        bar_col = config.COLOR_RED if off_sec >= max_off else (config.COLOR_YELLOW if off_sec >= 2.0 else config.COLOR_GREEN)
        cv2.rectangle(frame, (rx1 + 8, meter_y + 42), (rx1 + 8 + int(ratio_off * (meter_w - 16)), meter_y + 50), bar_col, -1)
        
        # Meter 2: 30s Cumulative Distraction
        cumul_sec = gaze_info.get('cumulative_off_road_sec', 0.0)
        max_cumul = config.CUMULATIVE_DISTRACTION_THRESHOLD_SEC # 10.0s
        ratio_cumul = min(1.0, cumul_sec / max_cumul)
        cv2.putText(frame, f"30s Cumul: {cumul_sec:.1f}s / {max_cumul:.0f}s", (rx1 + 8, meter_y + 70), config.FONT, 0.40, config.COLOR_WHITE, 1)
        cv2.rectangle(frame, (rx1 + 8, meter_y + 76), (rx2 - 8, meter_y + 84), (40, 40, 40), -1)
        cumul_col = config.COLOR_RED if cumul_sec >= max_cumul else config.COLOR_YELLOW
        cv2.rectangle(frame, (rx1 + 8, meter_y + 76), (rx1 + 8 + int(ratio_cumul * (meter_w - 16)), meter_y + 84), cumul_col, -1)
        
        # PERCLOS-80 Percentage
        cv2.putText(frame, f"PERCLOS-80: {perclos*100:.0f}% (max 30%)", (rx1 + 8, meter_y + 102), config.FONT, 0.40, config.COLOR_WHITE, 1)

        # ----------------------------------------------------------------------
        # 5. BIOMETRICS & BLINK DYNAMICS TELEMETRY (Left Top Overlay)
        # ----------------------------------------------------------------------
        bio_w = 230
        bio_h = 130
        bx1 = 15
        by1 = banner_h + 15
        bx2 = bx1 + bio_w
        by2 = by1 + bio_h
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), config.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), config.COLOR_GRAY, 1)
        
        cv2.putText(frame, "BIOMETRIC TELEMETRY", (bx1 + 10, by1 + 18), config.FONT, 0.42, config.COLOR_GRAY, 1)
        
        # Contactless rPPG Heart Rate
        bpm = rppg_info.get('bpm', 72.0)
        hrv = rppg_info.get('hrv_index', 50.0)
        rppg_ready = rppg_info.get('is_ready', False)
        
        hr_str = f"Heart Rate: {bpm:.0f} BPM" if rppg_ready else "Heart Rate: Acquiring..."
        cv2.putText(frame, hr_str, (bx1 + 10, by1 + 42), config.FONT, 0.48, config.COLOR_RED, 1)
        cv2.putText(frame, f"Autonomic HRV: {hrv:.0f} ms", (bx1 + 10, by1 + 60), config.FONT, 0.40, config.COLOR_WHITE, 1)
        
        # Blink Kinematics (Duration & AVR)
        last_ms = blink_info.get('duration_ms', 0.0)
        last_avr = blink_info.get('avr', 0.0)
        b_type = blink_info.get('blink_type', 'NOMINAL')
        drowsy_count = blink_info.get('drowsy_blink_count', 0)
        
        b_color = config.COLOR_RED if b_type == "DROWSY_DROOP" else config.COLOR_GREEN
        cv2.putText(frame, f"Last Blink: {last_ms:.0f} ms ({b_type})", (bx1 + 10, by1 + 84), config.FONT, 0.42, b_color, 1)
        cv2.putText(frame, f"AVR: {last_avr:.1f} | Droop Blinks: {drowsy_count}", (bx1 + 10, by1 + 104), config.FONT, 0.40, config.COLOR_WHITE, 1)
        cv2.putText(frame, f"MAR (Yawn): {mar:.2f}", (bx1 + 10, by1 + 122), config.FONT, 0.40, config.COLOR_WHITE, 1)

        # ----------------------------------------------------------------------
        # 6. DYNAMIC EAR WAVEFORM OSCILLOSCOPE (Bottom-Left HUD)
        # ----------------------------------------------------------------------
        recent_ears = blink_info.get('recent_ears', [])
        spark_x = 15
        spark_y = h - 65
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (spark_x, spark_y - 20), (spark_x + self.sparkline_w, spark_y + self.sparkline_h), config.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (spark_x, spark_y - 20), (spark_x + self.sparkline_w, spark_y + self.sparkline_h), config.COLOR_GRAY, 1)
        
        cv2.putText(frame, f"EAR WAVEFORM (Cur: {ear:.2f})", (spark_x + 8, spark_y - 6), config.FONT, 0.38, config.COLOR_GRAY, 1)
        
        # Draw threshold reference line
        norm_thresh_y = spark_y + self.sparkline_h - int((config.EAR_THRESHOLD / 0.45) * self.sparkline_h)
        norm_thresh_y = np.clip(norm_thresh_y, spark_y, spark_y + self.sparkline_h)
        cv2.line(frame, (spark_x + 5, norm_thresh_y), (spark_x + self.sparkline_w - 5, norm_thresh_y), (0, 0, 180), 1)
        
        # Plot points
        if len(recent_ears) >= 2:
            step = (self.sparkline_w - 10) / max(1, config.EAR_BUFFER_SIZE)
            pts = []
            for i, val in enumerate(recent_ears):
                px = int(spark_x + 5 + i * step)
                # Map EAR 0.0 - 0.45 into sparkline height
                py = int(spark_y + self.sparkline_h - np.clip(val / 0.45, 0.0, 1.0) * self.sparkline_h)
                pts.append((px, py))
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, config.COLOR_GREEN, 1)
            
        # ----------------------------------------------------------------------
        # 7. TELEMETRY BAR (Bottom Full Width)
        # ----------------------------------------------------------------------
        bar_h = 24
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (10, 10, 10), -1)
        telemetry_txt = (f"PITCH: {pitch:+.1f} | YAW: {yaw:+.1f} | ROLL: {roll:+.1f} | "
                         f"GAZE PITCH: {gaze_pitch:+.1f} | GAZE YAW: {gaze_yaw:+.1f} | "
                         f"HOTKEYS: [C] Calibrate Neutral | [M] Mute | [Q] Quit")
        cv2.putText(frame, telemetry_txt, (15, h - 7), config.FONT, 0.38, config.COLOR_WHITE, 1)

        return frame

