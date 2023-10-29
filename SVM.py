import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from cuml import SVC  # Import GPU-accelerated SVM from cuml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to NumPy arrays (required by cuml)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Train the GPU-accelerated SVM classifier
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Save the model using pickle
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Save the test data for later use
    with open('test_data.pkl', 'wb') as test_data_file:
        pickle.dump((X_test, y_test), test_data_file)

    # Make Predictions and Evaluate the Model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


if __name__ == '__main__':
    main()
