import numpy as np
from svm import CustomSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

angry_train = np.load('../../data/hybrid_data_train/angry_hybrid_features.npy')
disgust_train = np.load('../../data/hybrid_data_train/disgust_hybrid_features.npy')
fear_train = np.load('../../data/hybrid_data_train/fear_hybrid_features.npy') 
happy_train = np.load('../../data/hybrid_data_train/happy_hybrid_features.npy')
neutral_train = np.load('../../data/hybrid_data_train/neutral_hybrid_features.npy')
sad_train = np.load('../../data/hybrid_data_train/sad_hybrid_features.npy')
suprise_train = np.load('../../data/hybrid_data_train/surprise_hybrid_features.npy')

angry_test = np.load('../../data/hybrid_data_test/angry_hybrid_features.npy')
disgust_test = np.load('../../data/hybrid_data_test/disgust_hybrid_features.npy')
fear_test = np.load('../../data/hybrid_data_test/fear_hybrid_features.npy') 
happy_test = np.load('../../data/hybrid_data_test/happy_hybrid_features.npy')
neutral_test = np.load('../../data/hybrid_data_test/neutral_hybrid_features.npy')
sad_test = np.load('../../data/hybrid_data_test/sad_hybrid_features.npy')
suprise_test = np.load('../../data/hybrid_data_test/surprise_hybrid_features.npy')

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

svm = CustomSVM(C=1.0, gamma=0.05, max_iter=500, lr=0.001)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)
print("\nTest Accuracy:", accuracy)

y_pred = svm.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')

print("\nMacro F1-score:", f1_macro)
print("Weighted F1-score:", f1_weighted)
print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
