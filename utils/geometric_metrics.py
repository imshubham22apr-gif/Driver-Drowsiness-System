import numpy as np

def eye_aspect_ratio(eye_landmarks):
    """
    Computes the Eye Aspect Ratio (EAR).
    Uses 6 landmarks: [p1, p2, p3, p4, p5, p6]
    """
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_landmarks]
    
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    
    # Compute the euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(p1 - p4)
    
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """
    Computes the Mouth Aspect Ratio (MAR).
    """
    top_inner = np.array(mouth_landmarks['top_inner'])
    bottom_inner = np.array(mouth_landmarks['bottom_inner'])
    
    top_outer_l = np.array(mouth_landmarks['top_outer_left'])
    bottom_outer_l = np.array(mouth_landmarks['bottom_outer_left'])
    
    top_outer_r = np.array(mouth_landmarks['top_outer_right'])
    bottom_outer_r = np.array(mouth_landmarks['bottom_outer_right'])
    
    corner_l = np.array(mouth_landmarks['corner_left'])
    corner_r = np.array(mouth_landmarks['corner_right'])
    
    # Vertical distances
    A = np.linalg.norm(top_inner - bottom_inner)
    B = np.linalg.norm(top_outer_l - bottom_outer_l)
    C = np.linalg.norm(top_outer_r - bottom_outer_r)
    
    # Horizontal distance
    width = np.linalg.norm(corner_l - corner_r)
    
    # MAR
    mar = (A + B + C) / (3.0 * width)
    return mar
