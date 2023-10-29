import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC


def main():
    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784")
    X, y = mnist.data, mnist.target

    # Convert labels to binary (0 or 1)
    y = [1 if label == '1' else 0 for label in y]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the SVM model
    model = SVC(probability=True)  # Use probability=True to enable ROC curve later
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("SVM Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # Save the model and test data using Pickle
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('test_data_svm.pkl', 'wb') as test_data_file:
        pickle.dump((X_test, y_test), test_data_file)


if __name__ == '__main__':
    main()
