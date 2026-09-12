import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Left and right eye 6-point contours for EAR calculation
        # Format: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom2, p6_bottom1]
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Iris Landmark indices (MediaPipe 468-477)
        # Eye 1 (Right eye on camera / person's left): 468 is center
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]
        # Eye 2 (Left eye on camera / person's right): 473 is center
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        
        # Key reference landmarks for eye corners and eyelid centers (for Gaze vector)
        self.RIGHT_EYE_CORNER_OUTER = 33
        self.RIGHT_EYE_CORNER_INNER = 133
        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145
        
        self.LEFT_EYE_CORNER_INNER = 362
        self.LEFT_EYE_CORNER_OUTER = 263
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374
        
        # Mouth landmarks
        self.MOUTH_CORNERS = [78, 308]
        self.LIPS_INNER = [13, 14]
        self.LIPS_OUTER_L = [82, 87]
        self.LIPS_OUTER_R = [312, 317]
        
        # Head pose landmarks (PnP 3D Model Correspondence)
        self.HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]
        
        # Forehead Skin ROI for contactless rPPG pulse extraction
        self.FOREHEAD_ROI_INDICES = [109, 10, 338, 9]

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        landmarks_dict = {}
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            num_landmarks = len(face_landmarks.landmark)
            
            def get_coords(indices):
                return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
                
            landmarks_dict['left_eye'] = get_coords(self.LEFT_EYE)
            landmarks_dict['right_eye'] = get_coords(self.RIGHT_EYE)
            
            mouth_dict = {
                'corner_left': get_coords([self.MOUTH_CORNERS[0]])[0],
                'corner_right': get_coords([self.MOUTH_CORNERS[1]])[0],
                'top_inner': get_coords([self.LIPS_INNER[0]])[0],
                'bottom_inner': get_coords([self.LIPS_INNER[1]])[0],
                'top_outer_left': get_coords([self.LIPS_OUTER_L[0]])[0],
                'bottom_outer_left': get_coords([self.LIPS_OUTER_L[1]])[0],
                'top_outer_right': get_coords([self.LIPS_OUTER_R[0]])[0],
                'bottom_outer_right': get_coords([self.LIPS_OUTER_R[1]])[0]
            }
            landmarks_dict['mouth'] = mouth_dict
            
            # Head pose points (float precision for cv2.solvePnP)
            head_pts = []
            for idx in self.HEAD_POSE_INDICES:
                lm = face_landmarks.landmark[idx]
                head_pts.append((lm.x * w, lm.y * h))
            landmarks_dict['head_pose'] = head_pts
            
            # Iris landmarks (available if refine_landmarks=True, num_landmarks == 478)
            if num_landmarks >= 478:
                landmarks_dict['right_iris'] = get_coords(self.RIGHT_IRIS)
                landmarks_dict['left_iris'] = get_coords(self.LEFT_IRIS)
                landmarks_dict['right_iris_center'] = get_coords([self.RIGHT_IRIS[0]])[0]
                landmarks_dict['left_iris_center'] = get_coords([self.LEFT_IRIS[0]])[0]
                
                # Gaze reference points
                landmarks_dict['gaze_anchors'] = {
                    'r_outer': get_coords([self.RIGHT_EYE_CORNER_OUTER])[0],
                    'r_inner': get_coords([self.RIGHT_EYE_CORNER_INNER])[0],
                    'r_top': get_coords([self.RIGHT_EYE_TOP])[0],
                    'r_bottom': get_coords([self.RIGHT_EYE_BOTTOM])[0],
                    'l_inner': get_coords([self.LEFT_EYE_CORNER_INNER])[0],
                    'l_outer': get_coords([self.LEFT_EYE_CORNER_OUTER])[0],
                    'l_top': get_coords([self.LEFT_EYE_TOP])[0],
                    'l_bottom': get_coords([self.LEFT_EYE_BOTTOM])[0],
                }
            else:
                landmarks_dict['right_iris'] = []
                landmarks_dict['left_iris'] = []
                landmarks_dict['right_iris_center'] = None
                landmarks_dict['left_iris_center'] = None
                landmarks_dict['gaze_anchors'] = {}
                
            # Forehead skin ROI for rPPG
            landmarks_dict['forehead_roi'] = get_coords(self.FOREHEAD_ROI_INDICES)
            
        return landmarks_dict

