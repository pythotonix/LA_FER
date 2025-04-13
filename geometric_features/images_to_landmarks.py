import numpy as np
import os
import math

# === UTILITY FUNCTIONS ===

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def area_of_triangle(p1, p2, p3):
    a = euclidean_distance(p1, p2)
    b = euclidean_distance(p2, p3)
    c = euclidean_distance(p3, p1)
    s = (a + b + c) / 2
    return math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

def extract_geometric_features(landmarks_468: np.ndarray) -> np.ndarray:
    assert landmarks_468.shape == (468, 2), "Input must be a (468, 2) array"

    # Define indices
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]
    MOUTH_CORNERS = [61, 291]
    EYEBROWS = [70, 300]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_HEIGHT = [159, 145]
    RIGHT_EYE_HEIGHT = [386, 374]

    # Distances
    dist_left_eye = euclidean_distance(landmarks_468[33], landmarks_468[133])
    dist_right_eye = euclidean_distance(landmarks_468[362], landmarks_468[263])
    dist_mouth = euclidean_distance(landmarks_468[61], landmarks_468[291])
    dist_eyebrows = euclidean_distance(landmarks_468[70], landmarks_468[300])
    interocular = euclidean_distance(landmarks_468[33], landmarks_468[263])

    # Ratios
    ratio_mouth_to_eyes = dist_mouth / interocular if interocular != 0 else 0
    ratio_eyebrow_to_eyes = dist_eyebrows / interocular if interocular != 0 else 0

    # Angles
    angle_mouth_nose_chin = angle_between(landmarks_468[61], landmarks_468[1], landmarks_468[152])
    angle_eyebrow_eye = angle_between(landmarks_468[70], landmarks_468[300], landmarks_468[263])

    # Areas
    area_mouth_nose = area_of_triangle(landmarks_468[61], landmarks_468[291], landmarks_468[1])

    # Symmetry
    left_eye_height = abs(landmarks_468[159][1] - landmarks_468[145][1])
    right_eye_height = abs(landmarks_468[386][1] - landmarks_468[374][1])
    eye_height_diff = abs(left_eye_height - right_eye_height)

    return np.array([
        dist_left_eye,
        dist_right_eye,
        dist_mouth,
        dist_eyebrows,
        ratio_mouth_to_eyes,
        ratio_eyebrow_to_eyes,
        angle_mouth_nose_chin,
        angle_eyebrow_eye,
        area_mouth_nose,
        eye_height_diff
    ], dtype=np.float32)

# === PROCESS ALL FILES ===

load_dir = "data/landmarks_train"
save_dir = "geometric_features/data_train"
os.makedirs(save_dir, exist_ok=True)

# Process each .npy file in the input directory
for filename in os.listdir(load_dir):
    if filename.endswith("_landmarks.npy"):
        emotion = filename.replace("_landmarks.npy", "")
        print(f"Processing {emotion}...")

        full_path = os.path.join(load_dir, filename)
        landmarks_data = np.load(full_path)  # shape: (N, 468, 2)

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
