import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Define landmark indices for eye, mouth, and full jawline
eye_mouth_face_outline_indices = [
    # Left Eye (Upper & Lower)
    246, 161, 160, 159, 158, 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362,
    # Right Eye (Upper & Lower)
    466, 388, 387, 386, 385, 384, 398, 33, 7, 163, 144, 145, 153, 154, 155, 133,
    # Mouth Outer Border
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # Mouth Inner Border
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82,
    # Full Face Silhouette (Jawline - Ear to Ear)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454
]

# Load 48x48 grayscale image
image_path = "train\\happy\\Training_1206.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Process face
results = face_mesh.process(img_rgb)

# Check if face landmarks detected
if not results.multi_face_landmarks:
    print("No face detected!")
else:
    print(f"Detected {len(results.multi_face_landmarks)} face(s).")

    # Store selected landmark points
    landmarks = []

    for face_landmarks in results.multi_face_landmarks:
        for idx in eye_mouth_face_outline_indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            landmarks.append((x, y))
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)  # Draw landmark

    # Convert to NumPy array
    landmarks_array = np.array(landmarks)

    # Save landmarks array (optional)
    # np.save("landmarks_eyes_mouth_jawline.npy", landmarks_array)

    # Save image with landmarks
    cv2.imwrite('landmarks_eyes_mouth_jawline.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Check the stored points
print("Landmarks shape:", landmarks_array.shape)
