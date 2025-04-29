import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def classify_photo_knn(numpy_arr):
    angry_train = np.load('./data/hybrid_data_train/angry_hybrid_features.npy')
    disgust_train = np.load('./data/hybrid_data_train/disgust_hybrid_features.npy')
    fear_train = np.load('./data/hybrid_data_train/fear_hybrid_features.npy') 
    happy_train = np.load('./data/hybrid_data_train/happy_hybrid_features.npy')
    neutral_train = np.load('./data/hybrid_data_train/neutral_hybrid_features.npy')
    sad_train = np.load('./data/hybrid_data_train/sad_hybrid_features.npy')
    suprise_train = np.load('./data/hybrid_data_train/surprise_hybrid_features.npy')

    train_data = {
        'angry': angry_train,
        'disgust': disgust_train,
        'fear': fear_train,
        'happy': happy_train,
        'neutral': neutral_train,
        'sad': sad_train,
        'suprise': suprise_train
    }

    train_X_list = []
    train_y_list = []

    for emotion, data in train_data.items():
        train_X_list.append(data.reshape(data.shape[0], -1))
        train_y_list.append(np.full((data.shape[0],), emotion))

    X_train = np.concatenate(train_X_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=57, weights="distance")
    knn.fit(X_train_scaled, y_train)

    numpy_arr = numpy_arr.reshape(1, -1)

    photo_features_scaled = scaler.transform(numpy_arr)

    predicted_label = knn.predict(photo_features_scaled)

    return predicted_label[0]
