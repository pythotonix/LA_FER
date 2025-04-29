import argparse
from extract_features_one import extract_features_from_image
from classifiers.knn_classifier.classify_one import classify_photo_knn
# from classifiers.svm_classifier.classify_one import classify_photo_svm  # Assuming you have it

def main():
    parser = argparse.ArgumentParser(description="Emotion Classification from Image Features")
    parser.add_argument("file", type=str, help="Path to the input image file")
    parser.add_argument("classifier", type=str, choices=["knn", "svm"], help="Classifier to use (knn or svm)")
    
    args = parser.parse_args()

    features = extract_features_from_image(args.file)
    if features is None:
        print("❌ Could not extract features. Please check the image path or content.")
        return

    print(f"Feature vector size: {features.size}")

    if args.classifier == "knn":
        result = classify_photo_knn(features)
        print(f"✅ Predicted emotion using KNN: {result}")
    elif args.classifier == "svm":
        pass
        # result = classify_photo_svm(features)
        # print(f"✅ Predicted emotion using SVM: {result}")

if __name__ == "__main__":
    main()
