import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Selected 62 anthropometric landmark indices
selected_landmark_indices = [
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 288,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    133, 159, 145, 153, 154, 155, 246,
    362, 386, 374, 380, 381, 382, 466,
    1, 2, 98, 327, 168, 195, 5,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 0, 13, 14,
    152, 19, 94
]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    cosine_angle = np.dot(a, b) / norm_product
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def area_of_triangle(p1, p2, p3):
    a = euclidean_distance(p1, p2)
    b = euclidean_distance(p2, p3)
    c = euclidean_distance(p3, p1)
    s = (a + b + c) / 2
    return math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

def extract_geometric_features(landmarks_468: np.ndarray) -> np.ndarray:
    assert landmarks_468.shape == (468, 2), "Input must be a (468, 2) array"
    
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]
    MOUTH = [61, 291]
    MOUTH_TOP_BOTTOM = [13, 14]
    EYEBROWS = [70, 300]
    BROWS_R = [63, 70, 66]
    BROWS_L = [296, 300, 293]
    NOSE_TIP = 1
    NOSE_BRIDGE = 168
    CHIN = 152
    LEFT_EYE_HEIGHT = [159, 145]
    RIGHT_EYE_HEIGHT = [386, 374]
    LEFT_NOSTRIL = 98
    RIGHT_NOSTRIL = 327
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    dist_left_eye = euclidean_distance(landmarks_468[33], landmarks_468[133])
    dist_right_eye = euclidean_distance(landmarks_468[362], landmarks_468[263])
    dist_mouth = euclidean_distance(landmarks_468[61], landmarks_468[291])
    mouth_height = euclidean_distance(landmarks_468[13], landmarks_468[14])
    eyebrow_dist = euclidean_distance(landmarks_468[70], landmarks_468[300])
    nose_length = euclidean_distance(landmarks_468[168], landmarks_468[1])
    face_height = euclidean_distance(landmarks_468[1], landmarks_468[152])
    interocular = euclidean_distance(landmarks_468[33], landmarks_468[263])
    nostril_width = euclidean_distance(landmarks_468[98], landmarks_468[327])
    brow_eye_dist_R = euclidean_distance(landmarks_468[70], landmarks_468[159])
    brow_eye_dist_L = euclidean_distance(landmarks_468[300], landmarks_468[386])

    ratio_mouth_to_eyes = dist_mouth / interocular if interocular != 0 else 0
    ratio_eyebrow_to_eyes = eyebrow_dist / interocular if interocular != 0 else 0
    ratio_mouth_height_to_width = mouth_height / dist_mouth if dist_mouth != 0 else 0
    ratio_nose_to_face = nose_length / face_height if face_height != 0 else 0
    ratio_broweye_to_face = (brow_eye_dist_R + brow_eye_dist_L) / (2 * face_height) if face_height != 0 else 0

    angle_mouth_nose_chin = angle_between(landmarks_468[61], landmarks_468[1], landmarks_468[152])
    angle_eyebrow_eye = angle_between(landmarks_468[70], landmarks_468[300], landmarks_468[263])
    angle_brow_R = angle_between(landmarks_468[BROWS_R[0]], landmarks_468[BROWS_R[1]], landmarks_468[BROWS_R[2]])
    angle_brow_L = angle_between(landmarks_468[BROWS_L[0]], landmarks_468[BROWS_L[1]], landmarks_468[BROWS_L[2]])
    face_tilt = angle_between(landmarks_468[LEFT_CHEEK], landmarks_468[NOSE_TIP], landmarks_468[RIGHT_CHEEK])

    area_mouth_nose = area_of_triangle(landmarks_468[61], landmarks_468[291], landmarks_468[1])
    area_nostril_triangle = area_of_triangle(landmarks_468[98], landmarks_468[327], landmarks_468[1])

    left_eye_height = abs(landmarks_468[159][1] - landmarks_468[145][1])
    right_eye_height = abs(landmarks_468[386][1] - landmarks_468[374][1])
    eye_height_diff = abs(left_eye_height - right_eye_height)

    face_center_indices = [33, 263, 61, 291, 1, 152]
    face_center = np.mean(landmarks_468[face_center_indices], axis=0)
    centroid_distances = [euclidean_distance(landmarks_468[i], face_center) for i in face_center_indices]
    centroid_spread_mean = np.mean(centroid_distances)
    centroid_spread_range = max(centroid_distances) - min(centroid_distances)

    return np.array([
        dist_left_eye, dist_right_eye, dist_mouth, mouth_height, eyebrow_dist, nose_length, face_height,
        interocular, nostril_width, brow_eye_dist_R, brow_eye_dist_L,
        ratio_mouth_to_eyes, ratio_eyebrow_to_eyes, ratio_mouth_height_to_width,
        ratio_nose_to_face, ratio_broweye_to_face,
        angle_mouth_nose_chin, angle_eyebrow_eye, angle_brow_R, angle_brow_L, face_tilt,
        area_mouth_nose, area_nostril_triangle,
        eye_height_diff, centroid_spread_mean, centroid_spread_range
    ], dtype=np.float32)

def extract_features_from_image(file_path: str):
    """
    Extracts 62 landmarks and 25 geometric features from an input image and saves them as a NumPy array.
    Args:
        file_path (str): Path to the input image file.
    Returns:
        np.ndarray: Flattened feature vector (shape: (149,))
    """

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image at path: {file_path}")

    img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image.")

    landmarks_468 = np.zeros((468, 2), dtype=np.float32)
    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks_468[idx] = (landmark.x, landmark.y)

    selected_landmarks = landmarks_468[selected_landmark_indices].flatten()
    geometric_features = extract_geometric_features(landmarks_468)
    full_feature_vector = np.concatenate([selected_landmarks, geometric_features])

    return full_feature_vector

# features = extract_features_from_image("examples\\eugene.jpg")
# print(features)