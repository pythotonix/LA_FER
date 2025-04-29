import numpy as np
import os

landmark_dir = "feature_extraction/anthropometric_landmarks/data_train"
geometric_dir = "feature_extraction/geometric_features/data_train"
save_dir = "feature_extraction/hybrid/data_train"
os.makedirs(save_dir, exist_ok=True)

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

for emotion in emotions:
    landmark_path = os.path.join(landmark_dir, f"{emotion}_landmarks.npy")
    geometric_path = os.path.join(geometric_dir, f"{emotion}_features.npy")

    landmarks = np.load(landmark_path)
    geometric = np.load(geometric_path)

    assert landmarks.shape[0] == geometric.shape[0], f"Sample count mismatch for {emotion}"

    landmarks_flat = landmarks.reshape(landmarks.shape[0], -1)

    hybrid_features = np.concatenate([landmarks_flat, geometric], axis=1)

    save_path = os.path.join(save_dir, f"{emotion}_hybrid_features.npy")
    np.save(save_path, hybrid_features)
    print(f"Saved {save_path} with shape {hybrid_features.shape}")

print("All hybrid features created.")
