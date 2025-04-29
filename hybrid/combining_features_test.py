import numpy as np
import os

# === Paths ===
landmark_dir = "anthropometric_landmarks/data_test"
geometric_dir = "geometric_features/data_test"
save_dir = "hybrid/data_test"
os.makedirs(save_dir, exist_ok=True)

# === Emotions (filenames must follow *_landmarks.npy and *_features.npy) ===
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

for emotion in emotions:
    # Load files
    landmark_path = os.path.join(landmark_dir, f"{emotion}_landmarks.npy")
    geometric_path = os.path.join(geometric_dir, f"{emotion}_features.npy")

    landmarks = np.load(landmark_path)   # shape: (n, 62, 2)
    geometric = np.load(geometric_path)  # shape: (n, 25)

    # Sanity check
    assert landmarks.shape[0] == geometric.shape[0], f"Sample count mismatch for {emotion}"

    # Flatten landmarks: (n, 62, 2) → (n, 124)
    landmarks_flat = landmarks.reshape(landmarks.shape[0], -1)

    # Concatenate: (n, 124 + 25) → (n, 149)
    hybrid_features = np.concatenate([landmarks_flat, geometric], axis=1)

    # Save result
    save_path = os.path.join(save_dir, f"{emotion}_hybrid_features.npy")
    np.save(save_path, hybrid_features)
    print(f"Saved {save_path} with shape {hybrid_features.shape}")

print("All hybrid features created.")
