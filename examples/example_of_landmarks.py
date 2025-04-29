import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
image_path = "examples\\images\\input\\tetianka_1.jpg"
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
selected_landmark_indices = [
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 288,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    133, 159, 145, 153, 154, 155, 246,
    362, 386, 374, 380, 381, 382, 466,
    1, 2, 98, 327, 168, 195, 5,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 0, 13, 14,
    152, 19, 94
]
results = face_mesh.process(img_rgb)
img_draw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for idx in selected_landmark_indices:
            pt = face_landmarks.landmark[idx]
            x = int(pt.x * img_resized.shape[1])
            y = int(pt.y * img_resized.shape[0])
            cv2.circle(img_draw, (x, y), 1, (255, 255, 255), -1)

cv2.imwrite("examples\\images\\output\\tetianka_1_landmarks.jpg", img_draw)
print("Saved image with landmarks!")
