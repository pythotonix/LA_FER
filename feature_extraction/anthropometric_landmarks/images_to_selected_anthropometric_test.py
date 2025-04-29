import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

train_dir = "test"  # Parent directory containing emotion folders
save_dir = "feature_extraction\\anthropometric_landmarks\\data_test"  # Directory to save landmarks
os.makedirs(save_dir, exist_ok=True)

selected_landmark_indices = [
    # Jawline
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 288,
    # Eyebrows
    70, 63, 105, 66, 107,   # Right 
    336, 296, 334, 293, 300,  # Left 
    # Eyes
    133, 159, 145, 153, 154, 155, 246,  # Right eye
    362, 386, 374, 380, 381, 382, 466,  # Left eye

    # Nose
    1, 2, 98, 327, 168, 195, 5,

    # Mouth outer + inner
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 0, 13, 14,

    # Chin, mid-lip, and mid-forehead
    152, 19, 94
]
emotion_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
landmarks_dict = {emotion: [] for emotion in emotion_folders}
for emotion in emotion_folders:
    emotion_path = os.path.join(train_dir, emotion)
    print(f"Processing {emotion} images...")
    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {img_path}, failed to load.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            print(f"No face detected in {img_path}!")
            continue
        for face_landmarks in results.multi_face_landmarks:
            selected_landmarks = [
                (face_landmarks.landmark[idx].x,
                face_landmarks.landmark[idx].y)
                for idx in selected_landmark_indices
            ]
            landmarks_dict[emotion].append(selected_landmarks)

for emotion, landmarks_list in landmarks_dict.items():
    if landmarks_list:
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        np.save(os.path.join(save_dir, f"{emotion}_landmarks.npy"), landmarks_array)
        print(f"Saved {emotion}_landmarks.npy with shape {landmarks_array.shape}")

print("Landmark extraction completed successfully!")
