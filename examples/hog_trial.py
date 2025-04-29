from skimage.feature import hog
from skimage import io
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# === HOG + PCA on One Image ===
def extract_hog_features(img_path: str) -> np.ndarray:
    image = io.imread(img_path, as_gray=True)
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    features = hog(
        image_resized,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features.astype(np.float32)

# Path to image
img_path = "examples\\image.png"

# Extract HOG features
hog_raw = extract_hog_features(img_path)
print(f"Original HOG shape: {hog_raw.shape}")

# === Simulate PCA process ===
# Normally you'd fit PCA on all training data first. For now, simulate with dummy batch:
hog_dummy_batch = np.array([hog_raw])  # shape (1, 1568)

# Standardize (simulate with one-sample fit)
scaler = StandardScaler()
hog_scaled = scaler.fit_transform(hog_dummy_batch)

# PCA (simulate with one-sample fit)
pca = PCA(n_components=0.95)
hog_pca = pca.fit_transform(hog_scaled)

# Output result
print(f"PCA-reduced shape: {hog_pca.shape}")
print(f"Reduced HOG vector: {hog_pca[0]}")
