# Facial Emotion Recognition (FER)

![Python](https://img.shields.io/badge/python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Used-brightgreen)
![License](https://img.shields.io/github/license/pythotonix/LA_FER)

This repository contains a complete pipeline for **facial emotion recognition** using **machine learning (KNN, SVM)** and **deep learning (CNN)** models, developed as a Linear Algebra course project at the Ukrainian Catholic University.

> 🎓 Authors: Shvets Tetiana, Ostafiichuk Oleksandra, Kravchuk Yevhenii  
> 📅 Date: April 2025

---

## 🚀 Project Overview

The goal of this project is to classify facial expressions into one of 7 emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😀 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

We compare traditional machine learning approaches (KNN, SVM with hybrid landmark-based features) with a baseline **Convolutional Neural Network (CNN)** trained on raw images.

---

## 📦 Folder Structure

```
LA_FER/
├── classifiers/              # Contains KNN, SVM, CNN models
├── data/                     # Preprocessed landmarks and hybrid feature datasets
├── examples/                 # Example images and landmark visualizations
├── feature_extraction/       # Landmark selection + geometric feature code
├── notebooks/                # Experiments and visualizations
├── extract_features_one.py   # Single-image feature extractor
├── run.py                    # CLI script to run prediction on a photo
├── README.md                 # Project documentation
```

---

## 🧠 Key Features

- 🧩 **149-D hybrid feature vector** combining:
  - 62 facial landmarks (MediaPipe)
  - 25 handcrafted geometric features (distances, angles, ratios)
- ⚙️ KNN and SVM classification based on extracted features
- 🧠 CNN model for comparison (using grayscale FER2013 images)
- 📊 Performance Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- 🧪 Easy CLI interface to test on custom images

---

## 🖼️ Example Usage (CLI)

To predict emotion for a single image using KNN or SVM:

```bash
python run.py path_to_image.jpg knn
python run.py path_to_image.jpg svm
```

Ensure the image contains a visible human face. MediaPipe FaceMesh must detect 468 landmarks.

## 📊 Evaluation Results

| Model | Accuracy | Macro F1 | Weighted F1 |
| :---: | :------: | :------: | :---------: |
|  CNN  |  56.1%   |  0.4900  |   0.5480    |
|  SVM  |  51.5%   |  0.4637  |   0.5000    |
|  KNN  |  47.1%   |  0.4128  |   0.4336    |

CNN performs best overall, but KNN and SVM with engineered features are lightweight and interpretable.

---

## 📚 Dataset

We used the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) available on Kaggle.

---

## 📖 References

- [KNN-based Emotion Recognition](https://www.researchgate.net/publication/368884295_K-nearest_neighbor_based_facial_emotion_recognition_using_effective_features)
- [Facial Expression Recognition using HOG & CNN](https://www.jomcom.org/index.php/1/article/view/77/41)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🛠 Requirements

- Python 3.10+
- `mediapipe`
- `opencv-python`
- `scikit-learn`
- `tensorflow` (for CNN model)

### Install dependencies:

```bash
pip install -r requirements.txt
```

⚠️ **Note:** CNN training code is provided separately in the `notebooks/`.

---

## 📎 License

This project is released under the MIT License. See `LICENSE` for details.
