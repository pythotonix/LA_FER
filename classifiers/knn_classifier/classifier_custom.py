import numpy as np
from sklearn.preprocessing import StandardScaler
from knn import KNN_Weighted
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

angry_train = np.load('../data/selected_landmarks_train/angry_landmarks.npy')
disgust_train = np.load('../data/selected_landmarks_train/disgust_landmarks.npy')
fear_train = np.load('../data/selected_landmarks_train/fear_landmarks.npy') 
happy_train = np.load('../data/selected_landmarks_train/happy_landmarks.npy')
neutral_train = np.load('../data/selected_landmarks_train/neutral_landmarks.npy')
sad_train = np.load('../data/selected_landmarks_train/sad_landmarks.npy')
suprise_train = np.load('../data/selected_landmarks_train/surprise_landmarks.npy')

angry_test = np.load('../data/selected_landmarks_test/angry_landmarks.npy')
disgust_test = np.load('../data/selected_landmarks_test/disgust_landmarks.npy')
fear_test = np.load('../data/selected_landmarks_test/fear_landmarks.npy') 
happy_test = np.load('../data/selected_landmarks_test/happy_landmarks.npy')
neutral_test = np.load('../data/selected_landmarks_test/neutral_landmarks.npy')
sad_test = np.load('../data/selected_landmarks_test/sad_landmarks.npy')
suprise_test = np.load('../data/selected_landmarks_test/surprise_landmarks.npy')

train_data = {
    'angry': angry_train,
    'disgust': disgust_train,
    'fear': fear_train,
    'happy': happy_train,
    'neutral': neutral_train,
    'sad': sad_train,
    'suprise': suprise_train
}

test_data = {
    'angry': angry_test,
    'disgust': disgust_test,
    'fear': fear_test,
    'happy': happy_test,
    'neutral': neutral_test,
    'sad': sad_test,
    'suprise': suprise_test
}

train_X_list = []
train_y_list = []

for emotion, data in train_data.items():
    train_X_list.append(data.reshape(data.shape[0], -1))
    train_y_list.append(np.full((data.shape[0],), emotion))

X_train = np.concatenate(train_X_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)

test_X_list = []
test_y_list = []

for emotion, data in test_data.items():
    test_X_list.append(data.reshape(data.shape[0], -1))
    test_y_list.append(np.full((data.shape[0],), emotion))

X_test = np.concatenate(test_X_list, axis=0)
y_test = np.concatenate(test_y_list, axis=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNN_Weighted(k=97)
knn.fit(X_train_scaled, y_train)

knn_predictions = knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, knn_predictions, labels=knn.classes_)

print("Classification Report:\n")
report = classification_report(y_test, knn_predictions, digits=4, zero_division=0)
print(report)

print("Overall Metrics:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Macro F1-score:", f1_score(y_test, knn_predictions, average='macro'))
print("Weighted F1-score:", f1_score(y_test, knn_predictions, average='weighted'))
print("Macro Precision:", precision_score(y_test, knn_predictions, average='macro'))
print("Macro Recall:", recall_score(y_test, knn_predictions, average='macro'))

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
    