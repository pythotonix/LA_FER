import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
selected_landmark_indices = [
    # Jawline
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 288,
    # Eyebrows
    70, 63, 105, 66, 107,   # Right eyebrow
    336, 296, 334, 293, 300,  # Left eyebrow
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

image_path = "examples\\images\\input\\image.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
results = face_mesh.process(img_rgb)
if not results.multi_face_landmarks:
    print("No face detected!")
else:
    print(f"Detected {len(results.multi_face_landmarks)} face(s).")
    landmarks = []

    for face_landmarks in results.multi_face_landmarks:
        for idx in selected_landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            landmarks.append((x, y))
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

    landmarks_array = np.array(landmarks)
    cv2.imwrite('selected_landmark_indices.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Landmarks shape:", landmarks_array.shape)
