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
        
        # Left and right eye landmarks from MediaPipe
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Mouth landmarks
        self.MOUTH_CORNERS = [78, 308]
        self.LIPS_INNER = [13, 14]
        self.LIPS_OUTER_L = [82, 87]
        self.LIPS_OUTER_R = [312, 317]
        
        # Head pose landmarks
        self.HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        landmarks_dict = {}
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
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
            
            # Head pose uses float coordinates for accuracy
            head_pts = []
            for idx in self.HEAD_POSE_INDICES:
                lm = face_landmarks.landmark[idx]
                head_pts.append((lm.x * w, lm.y * h))
            landmarks_dict['head_pose'] = head_pts
            
        return landmarks_dict
