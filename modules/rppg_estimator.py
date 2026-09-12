import cv2
import numpy as np
import scipy.signal
from collections import deque
import config

class RPPGEstimator:
    """
    Contactless Facial Remote Photoplethysmography (rPPG).
    Extracts blood volume pulse (BVP) and Heart Rate (BPM) from subtle 
    diffuse reflectance color changes in the driver's forehead skin ROI 
    using the Plane-Orthogonal-to-Skin (POS) chrominance algorithm.
    """
    def __init__(self, buffer_size=config.RPPG_BUFFER_SIZE, fps=config.RPPG_FPS_DEFAULT):
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Color buffers for R, G, B channels
        self.r_buffer = deque(maxlen=buffer_size)
        self.g_buffer = deque(maxlen=buffer_size)
        self.b_buffer = deque(maxlen=buffer_size)
        
        # Filtered pulse waveform buffer for UI visualization
        self.pulse_waveform = deque(maxlen=60)
        
        self.current_bpm = 72.0
        self.smooth_bpm = 72.0
        self.hrv_index = 50.0 # Heart rate variability indicator (ms proxy)
        self.is_ready = False

    def update(self, frame, forehead_roi_pts):
        """
        Extracts skin patch, runs POS algorithm, and estimates Heart Rate.
        
        Args:
            frame: OpenCV BGR image
            forehead_roi_pts: List of 2D coordinates [p1, p2, p3, p4]
            
        Returns:
            bpm (float): Estimated heart rate in beats per minute
            hrv_index (float): Autonomic HRV indicator
            pulse_waveform (list): Recent filtered pulse points for UI
            is_ready (bool): True if signal has stabilized
        """
        if not forehead_roi_pts or len(forehead_roi_pts) < 3:
            return self.smooth_bpm, self.hrv_index, list(self.pulse_waveform), self.is_ready

        # Calculate bounding box of the forehead ROI
        pts = np.array(forehead_roi_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0 or w < 8 or h < 8:
            return self.smooth_bpm, self.hrv_index, list(self.pulse_waveform), self.is_ready
            
        # Calculate mean color in BGR channels (convert to RGB)
        mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
        
        self.r_buffer.append(mean_r)
        self.g_buffer.append(mean_g)
        self.b_buffer.append(mean_b)
        
        # We need at least ~60 frames (~2.0 seconds) to compute pulse
        if len(self.r_buffer) >= 60:
            self.is_ready = True
            
            r_arr = np.array(self.r_buffer, dtype=np.float64)
            g_arr = np.array(self.g_buffer, dtype=np.float64)
            b_arr = np.array(self.b_buffer, dtype=np.float64)
            
            # Plane-Orthogonal-to-Skin (POS) Algorithm
            # 1. Temporal normalization: divide by temporal mean
            r_norm = r_arr / (np.mean(r_arr) + 1e-6)
            g_norm = g_arr / (np.mean(g_arr) + 1e-6)
            b_norm = b_arr / (np.mean(b_arr) + 1e-6)
            
            # 2. Chrominance orthogonal projections
            s1 = g_norm - b_norm
            s2 = g_norm + b_norm - 2.0 * r_norm
            
            std_s1 = np.std(s1)
            std_s2 = np.std(s2)
            alpha = (std_s1 / (std_s2 + 1e-6))
            
            # 3. Combined pulse signal
            pulse = s1 + alpha * s2
            
            # 4. Zero-phase Butterworth Bandpass filter (0.75 - 3.0 Hz / 45 - 180 BPM)
            try:
                low = config.RPPG_MIN_HR_BPM / 60.0
                high = config.RPPG_MAX_HR_BPM / 60.0
                nyq = 0.5 * self.fps
                low_cut = max(0.01, min(0.99, low / nyq))
                high_cut = max(0.02, min(0.99, high / nyq))
                
                if low_cut < high_cut:
                    b, a = scipy.signal.butter(2, [low_cut, high_cut], btype='bandpass')
                    filtered_pulse = scipy.signal.filtfilt(b, a, pulse)
                else:
                    filtered_pulse = pulse - np.mean(pulse)
            except Exception:
                filtered_pulse = pulse - np.mean(pulse)
                
            # Update live pulse wave for HUD
            if len(filtered_pulse) > 0:
                norm_val = float(filtered_pulse[-1])
                self.pulse_waveform.append(norm_val)
                
            # 5. Spectral Peak Analysis (FFT)
            n = len(filtered_pulse)
            fft_vals = np.abs(np.fft.rfft(filtered_pulse))
            fft_freqs = np.fft.rfftfreq(n, 1.0 / self.fps)
            
            # Restrict search within physiological cardiac frequency band [0.75, 3.0] Hz
            valid_mask = (fft_freqs >= (config.RPPG_MIN_HR_BPM / 60.0)) & (fft_freqs <= (config.RPPG_MAX_HR_BPM / 60.0))
            
            if np.any(valid_mask):
                masked_freqs = fft_freqs[valid_mask]
                masked_vals = fft_vals[valid_mask]
                peak_idx = np.argmax(masked_vals)
                best_freq = masked_freqs[peak_idx]
                raw_bpm = best_freq * 60.0
                
                # Smooth BPM using exponential filter
                self.current_bpm = raw_bpm
                self.smooth_bpm = 0.15 * raw_bpm + 0.85 * self.smooth_bpm
                
                # Proxy HRV indicator from spectral entropy / dominant ratio
                dominant_power = masked_vals[peak_idx]
                total_power = np.sum(masked_vals) + 1e-6
                self.hrv_index = float(np.clip((1.0 - (dominant_power / total_power)) * 100.0, 20.0, 95.0))
                
        return self.smooth_bpm, self.hrv_index, list(self.pulse_waveform), self.is_ready
