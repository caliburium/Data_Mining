import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


def main():
    # Load the Logistic Regression model and test data
    with open('pickle/logistic_regression_model.pkl', 'rb') as model_file:
        logistic_model = pickle.load(model_file)

    with open('pickle/test_data_logistic_regression.pkl', 'rb') as test_data_file:
        X_test_logistic, y_test_logistic = pickle.load(test_data_file)

    # Load the Decision Tree model and test data
    with open('pickle/decision_tree_model.pkl', 'rb') as model_file:
        decision_tree_model = pickle.load(model_file)

    with open('pickle/test_data_decision_tree.pkl', 'rb') as test_data_file:
        X_test_decision_tree, y_test_decision_tree = pickle.load(test_data_file)

    # Load the SVM model and test data
    with open('pickle/svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)

    with open('pickle/test_data_svm.pkl', 'rb') as test_data_file:
        X_test_svm, y_test_svm = pickle.load(test_data_file)

    # Load the KNN model and test data
    with open('pickle/knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)

    with open('pickle/test_data_knn.pkl', 'rb') as test_data_file:
        X_test_knn, y_test_knn = pickle.load(test_data_file)

    # Initialize empty dictionaries to store AUC values for each model
    auc_values = {}

    # Plot ROC curves and calculate AUC for each model
    plt.figure()
    models = [("Logistic Regression", logistic_model, X_test_logistic, y_test_logistic),
              ("Decision Tree", decision_tree_model, X_test_decision_tree, y_test_decision_tree),
              ("SVM", svm_model, X_test_svm, y_test_svm),
              ("K-Nearest Neighbors", knn_model, X_test_knn, y_test_knn)]

    for model_name, model, X_test, y_test in models:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_values[model_name] = roc_auc

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Rank models by AUC values
    sorted_models = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
    print("Model Rankings (AUC values):")
    for i, (model_name, auc_value) in enumerate(sorted_models, start=1):
        print(f"{i}. {model_name}: {auc_value:.3f}")

    # Calculate and print other metrics for each model
    for model_name, model, X_test, y_test in models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{model_name} Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

    # Discuss any surprising results or observations
    print("\nDiscussion:")
    # You can add your observations and analysis here.

    plt.show()


if __name__ == '__main__':
    main()
