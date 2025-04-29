import cv2
import mediapipe as mp
import numpy as np
import os

# Load MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Path to dataset
train_dir = "test"  # Parent directory containing emotion folders
save_dir = "anthropometric_landmarks\\data_test"  # Directory to save landmarks

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Define 90 landmark indices (eyes, mouth, and jawline)
selected_landmark_indices = [
    # Jawline (simplified)
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 288,

    # Eyebrows (inner, middle, outer for both sides)
    70, 63, 105, 66, 107,   # Right eyebrow
    336, 296, 334, 293, 300,  # Left eyebrow

    # Eyes (corners + centers)
    133, 159, 145, 153, 154, 155, 246,  # Right eye
    362, 386, 374, 380, 381, 382, 466,  # Left eye

    # Nose (bridge + tip + nostrils)
    1, 2, 98, 327, 168, 195, 5,

    # Mouth outer + inner (simplified)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 0, 13, 14,

    # Chin, mid-lip, and mid-forehead
    152, 19, 94
]

# Get list of emotion categories (subfolders)
emotion_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

# Dictionary to store landmarks for each emotion
landmarks_dict = {emotion: [] for emotion in emotion_folders}

# Iterate over each emotion folder
for emotion in emotion_folders:
    emotion_path = os.path.join(train_dir, emotion)
    print(f"Processing {emotion} images...")

    # Iterate over images in the emotion folder
    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)

        # Load 48x48 grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {img_path}, failed to load.")
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Process face
        results = face_mesh.process(img_rgb)

        # Check if face landmarks detected
        if not results.multi_face_landmarks:
            print(f"No face detected in {img_path}!")
            continue

        # Extract only the selected 90 landmarks
        for face_landmarks in results.multi_face_landmarks:
            selected_landmarks = [
                (face_landmarks.landmark[idx].x,
                face_landmarks.landmark[idx].y)
                for idx in selected_landmark_indices
            ]
            landmarks_dict[emotion].append(selected_landmarks)  # Append to emotion category

# Convert lists to NumPy arrays and save them
for emotion, landmarks_list in landmarks_dict.items():
    if landmarks_list:  # Ensure we have data before saving
        landmarks_array = np.array(landmarks_list, dtype=np.float32) # Convert to NumPy array
        np.save(os.path.join(save_dir, f"{emotion}_landmarks.npy"), landmarks_array)  # Save as .npy file
        print(f"Saved {emotion}_landmarks.npy with shape {landmarks_array.shape}")

print("Landmark extraction completed successfully!")
