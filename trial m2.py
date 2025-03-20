import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load 48x48 grayscale image
image_path = "train\\angry\\Training_99982465.jpg"
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

    # Store landmark points
    landmarks = []

    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            landmarks.append((x, y))
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)  # Draw landmark

    # Convert to NumPy array
    landmarks_array = np.array(landmarks)

    # Save landmarks array (optional)
    # np.save("landmarks.npy", landmarks_array)

    # Show image with landmarks
    cv2.imwrite('landmarks.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Check the stored points
print("Landmarks shape:", landmarks_array.shape)
