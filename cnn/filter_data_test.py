import cv2
import mediapipe as mp
import os
import shutil

# Load MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Paths
input_dir = "test"  # or "test"
output_dir = "filtered_test"  # Save filtered images here

# Create output directory structure
os.makedirs(output_dir, exist_ok=True)

# Get list of emotion categories
emotion_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for emotion in emotion_folders:
    input_emotion_path = os.path.join(input_dir, emotion)
    output_emotion_path = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_path, exist_ok=True)

    print(f"Processing {emotion} images...")

    # Iterate over images
    for img_name in os.listdir(input_emotion_path):
        img_path = os.path.join(input_emotion_path, img_name)

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {img_path}, failed to load.")
            continue

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Process face
        results = face_mesh.process(img_rgb)

        # Save only if face detected
        if results.multi_face_landmarks:
            # Save image into filtered directory (keeping the same filename)
            save_path = os.path.join(output_emotion_path, img_name)
            cv2.imwrite(save_path, img)
        else:
            print(f"No face detected in {img_path}, skipped.")

print("✅ Image filtering completed!")
