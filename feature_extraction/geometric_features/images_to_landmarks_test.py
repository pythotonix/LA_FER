import numpy as np
import os
import math

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

    mouth_width = euclidean_distance(landmarks_468[61], landmarks_468[291])
    mar_numerator = (
        euclidean_distance(landmarks_468[13], landmarks_468[14]) +
        euclidean_distance(landmarks_468[78], landmarks_468[308]) +
        euclidean_distance(landmarks_468[82], landmarks_468[312])
    )
    mar = mar_numerator / (3 * mouth_width) if mouth_width != 0 else 0

    jawline_chin_angle = angle_between(landmarks_468[234], landmarks_468[152], landmarks_468[454])

    features = np.array([
        dist_left_eye, dist_right_eye, dist_mouth, mouth_height, eyebrow_dist, nose_length, face_height,
        interocular, nostril_width, brow_eye_dist_R, brow_eye_dist_L,
        ratio_mouth_to_eyes, ratio_eyebrow_to_eyes, ratio_mouth_height_to_width,
        ratio_nose_to_face, ratio_broweye_to_face,
        angle_mouth_nose_chin, angle_eyebrow_eye,
        area_mouth_nose, area_nostril_triangle,
        eye_height_diff, centroid_spread_mean, centroid_spread_range, mar, jawline_chin_angle
    ], dtype=np.float32)

    for i, feature in enumerate(features):
        if np.isnan(feature) or np.isinf(feature):
            raise ValueError(f"Feature {i} is NaN or Inf")
    

    return np.array([
        dist_left_eye, dist_right_eye, dist_mouth, mouth_height, eyebrow_dist, nose_length, face_height,
        interocular, nostril_width, brow_eye_dist_R, brow_eye_dist_L,
        ratio_mouth_to_eyes, ratio_eyebrow_to_eyes, ratio_mouth_height_to_width,
        ratio_nose_to_face, ratio_broweye_to_face,
        angle_mouth_nose_chin, angle_eyebrow_eye,
        area_mouth_nose, area_nostril_triangle,
        eye_height_diff, centroid_spread_mean, centroid_spread_range, mar, jawline_chin_angle
    ], dtype=np.float32)

load_dir = "data/landmarks_test"
save_dir = "feature_extraction/geometric_features/data_test"
os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(load_dir):
    if filename.endswith("_landmarks.npy"):
        emotion = filename.replace("_landmarks.npy", "")
        print(f"Processing {emotion}...")

        full_path = os.path.join(load_dir, filename)
        landmarks_data = np.load(full_path)

        feature_vectors = []

        for i, face_landmarks in enumerate(landmarks_data):
            try:
                features = extract_geometric_features(face_landmarks)
                feature_vectors.append(features)
            except Exception as e:
                print(f"Error on {emotion} sample {i}: {e}")
                continue

        feature_array = np.array(feature_vectors, dtype=np.float32)
        save_path = os.path.join(save_dir, f"{emotion}_features.npy")
        np.save(save_path, feature_array)
        print(f"Saved {save_path} with shape {feature_array.shape}")

print("All features extracted successfully!")
