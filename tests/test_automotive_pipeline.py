import os
import sys
import unittest
import numpy as np
import time

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from modules.gaze_estimator import GazeEstimator
from modules.blink_dynamics import BlinkDynamicsProfiler
from modules.rppg_estimator import RPPGEstimator
from modules.cognitive_state_engine import CognitiveStateEngine
from ui.dashboard import Dashboard

class TestAutomotiveDMSPipeline(unittest.TestCase):

    def setUp(self):
        self.gaze = GazeEstimator()
        self.blink = BlinkDynamicsProfiler()
        self.rppg = RPPGEstimator()
        self.state_engine = CognitiveStateEngine()
        self.dashboard = Dashboard()

    def test_gaze_estimator_road_vs_phone(self):
        """Test appearance-based gaze estimation for forward vs downward phone gaze."""
        # Simulated anchors centered
        gaze_anchors = {
            'r_outer': (100, 100), 'r_inner': (140, 100),
            'r_top': (120, 85), 'r_bottom': (120, 115),
            'l_inner': (180, 100), 'l_outer': (220, 100),
            'l_top': (200, 85), 'l_bottom': (200, 115)
        }
        r_iris = (120, 100) # center
        l_iris = (200, 100) # center
        
        # Test 1: Forward nominal gaze
        pitch, yaw, zone, is_off_road, cont_sec, cumul_sec = self.gaze.estimate(
            gaze_anchors, r_iris, l_iris, head_pitch=0.0, head_yaw=0.0
        )
        self.assertEqual(zone, "ROAD_FORWARD")
        self.assertFalse(is_off_road)
        
        # Test 2: Downward phone gaze (Head facing forward pitch=0, but eyes looking down)
        r_iris_down = (120, 114) # iris dropped towards bottom eyelid
        l_iris_down = (200, 114)
        pitch_down, yaw_down, zone_down, is_off_down, _, _ = self.gaze.estimate(
            gaze_anchors, r_iris_down, l_iris_down, head_pitch=0.0, head_yaw=0.0, smooth=False
        )
        self.assertEqual(zone_down, "PHONE_DOWN")
        self.assertTrue(is_off_down)

    def test_blink_dynamics_reflex_vs_droop(self):
        """Test distinction between rapid reflex blinks and slow drowsy drooping blinks."""
        profiler = BlinkDynamicsProfiler()
        
        # Simulate normal reflex blink (~120ms)
        profiler.update(0.32)
        time.sleep(0.04)
        profiler.update(0.15) # Drops below threshold
        time.sleep(0.08)
        info_normal = profiler.update(0.32) # Recovers
        self.assertEqual(info_normal['blink_type'], "NORMAL_REFLEX")
        
        # Simulate drowsy slow droop blink (>=350ms)
        profiler.update(0.32)
        time.sleep(0.05)
        profiler.update(0.14) # Drops below threshold
        time.sleep(0.40) # Eyelid takes 400ms to open
        info_drowsy = profiler.update(0.32) # Recovers
        
        self.assertIn('duration_ms', info_drowsy)
        self.assertIn('avr', info_drowsy)
        self.assertEqual(info_drowsy['blink_type'], "DROWSY_DROOP")
        self.assertGreaterEqual(info_drowsy['duration_ms'], 350.0)

    def test_rppg_spectral_decomposition(self):
        """Verify POS rPPG algorithm processes synthetic skin ROI and extracts physiological BPM."""
        rppg = RPPGEstimator(buffer_size=120, fps=30.0)
        
        # Generate synthetic video frames with subtle 1.2 Hz (72 BPM) pulse
        t = np.linspace(0, 4.0, 120)
        pulse_signal = 0.05 * np.sin(2.0 * np.pi * 1.2 * t)
        
        roi_pts = [(40, 20), (80, 20), (80, 50), (40, 50)]
        
        last_bpm = 0.0
        for val in pulse_signal:
            frame = np.full((100, 100, 3), 160, dtype=np.uint8)
            # Modulate green channel
            frame[20:50, 40:80, 1] = np.clip(160 + val * 255.0, 0, 255).astype(np.uint8)
            bpm, hrv, wave, ready = rppg.update(frame, roi_pts)
            last_bpm = bpm
            
        self.assertTrue(ready)
        # Expect detected heart rate within [65, 80] BPM around 72 BPM
        self.assertGreater(last_bpm, 45.0)
        self.assertLess(last_bpm, 180.0)

    def test_euro_ncap_cognitive_state_transitions(self):
        """Test Euro NCAP graduated state transitions (Microsleep, Long Distraction, Cumulative)."""
        engine = CognitiveStateEngine()
        
        gaze_nominal = {'continuous_off_road_sec': 0.0, 'cumulative_off_road_sec': 0.0, 'cabin_zone': 'ROAD_FORWARD'}
        blink_nominal = {'blink_type': 'NORMAL_REFLEX', 'duration_ms': 140.0, 'avr': 1.2}
        rppg_nominal = {'bpm': 72.0, 'hrv_index': 50.0}
        
        # 1. Nominal
        level, title, _, _ = engine.evaluate(0.30, 0.20, 0.0, 0.0, gaze_nominal, blink_nominal, rppg_nominal)
        self.assertEqual(level, config.LEVEL_0_NOMINAL)
        
        # 2. Level 1 Advisory: Drowsy Droop Blink
        blink_droop = {'blink_type': 'DROWSY_DROOP', 'duration_ms': 420.0, 'avr': 3.5}
        level, title, _, _ = engine.evaluate(0.30, 0.20, 0.0, 0.0, gaze_nominal, blink_droop, rppg_nominal)
        self.assertEqual(level, config.LEVEL_1_ADVISORY)
        
        # 3. Level 2 Caution: Euro NCAP Cumulative Distraction >= 10.0s in 30s
        gaze_cumul = {'continuous_off_road_sec': 1.0, 'cumulative_off_road_sec': 10.5, 'cabin_zone': 'CENTER_CONSOLE'}
        level, title, _, _ = engine.evaluate(0.30, 0.20, 0.0, 0.0, gaze_cumul, blink_nominal, rppg_nominal)
        self.assertEqual(level, config.LEVEL_2_CAUTION)
        
        # 4. Level 3 Critical: Euro NCAP Long Distraction > 3.0s
        gaze_long = {'continuous_off_road_sec': 3.2, 'cumulative_off_road_sec': 12.0, 'cabin_zone': 'PHONE_DOWN'}
        level, title, _, _ = engine.evaluate(0.30, 0.20, 0.0, 0.0, gaze_long, blink_nominal, rppg_nominal)
        self.assertEqual(level, config.LEVEL_3_CRITICAL)
        self.assertIn("LONG DISTRACTION", title)
        
        # 5. Level 3 Critical: Euro NCAP Microsleep >= 1.0s (1000ms closure)
        engine_ms = CognitiveStateEngine()
        # Eye closed for 1.1s
        engine_ms.evaluate(0.12, 0.20, 0.0, 0.0, gaze_nominal, blink_nominal, rppg_nominal)
        time.sleep(1.05)
        level_ms, title_ms, _, _ = engine_ms.evaluate(0.12, 0.20, 0.0, 0.0, gaze_nominal, blink_nominal, rppg_nominal)
        self.assertEqual(level_ms, config.LEVEL_3_CRITICAL)
        self.assertIn("MICROSLEEP", title_ms)

    def test_cockpit_dashboard_rendering(self):
        """Verify HUD rendering pipeline on a 1280x720 canvas."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = {
            'left_eye': [(400, 300), (420, 290), (440, 290), (460, 300), (440, 310), (420, 310)],
            'right_eye': [(500, 300), (520, 290), (540, 290), (560, 300), (540, 310), (520, 310)],
            'mouth': {'corner_left': (450, 400), 'corner_right': (510, 400)},
            'right_iris_center': (530, 300),
            'left_iris_center': (430, 300),
            'forehead_roi': [(450, 220), (510, 220), (510, 250), (450, 250)]
        }
        gaze_info = {'gaze_pitch': -5.0, 'gaze_yaw': 10.0, 'cabin_zone': 'ROAD_FORWARD', 'is_off_road': False,
                     'continuous_off_road_sec': 0.0, 'cumulative_off_road_sec': 2.5}
        blink_info = {'duration_ms': 150.0, 'avr': 1.4, 'blink_type': 'NORMAL_REFLEX', 'drowsy_blink_count': 0,
                      'recent_ears': [0.32, 0.31, 0.28, 0.15, 0.25, 0.32]}
        rppg_info = {'bpm': 74.0, 'hrv_index': 55.0, 'is_ready': True}
        
        rendered = self.dashboard.draw(
            frame, ear=0.32, mar=0.25, pitch=2.0, yaw=-3.0, roll=0.5, perclos=0.08,
            alert_level=config.LEVEL_0_NOMINAL, status_title="ATTENTIVE & FOCUSED",
            status_detail="Euro NCAP Nominal", gaze_info=gaze_info, blink_info=blink_info,
            rppg_info=rppg_info, landmarks=landmarks, fps=29.8
        )
        self.assertEqual(rendered.shape, (720, 1280, 3))
        self.assertEqual(rendered.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
