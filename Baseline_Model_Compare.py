import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore') # Ignore warnings, especially from GridSearchCV or metrics calculation

# --- Assume data is loaded and split from previous steps (Part 1 & 2.1) ---
# For demonstration, let's re-create the dummy data and train basic SVM/KNN models.
# In your actual implementation, you would use the X_train, X_test, y_train, y_test
# and the trained 'final_svm_model' and 'final_knn_model' from steps 2.1 and 2.2.

# Dummy data generation (replace with your actual loaded data)
num_samples = 1000
num_features = 100
num_classes = 8 # Anthracnose, Bacterial Canker, etc., and Healthy

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, num_classes, num_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Dummy training for demonstration if you haven't run 2.1 and 2.2 yet ---
# In a real scenario, these would be the 'best_estimator_' from your GridSearchCV results.

# Train a dummy SVM model
print("Training dummy SVM model for evaluation demonstration...")
# Using a common parameter for quick training, not the result of GridSearchCV
dummy_svm_model = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
dummy_svm_model.fit(X_train, y_train)
final_svm_model = dummy_svm_model # Assign to the variable expected by this step

# Train a dummy KNN model
print("Training dummy KNN model for evaluation demonstration...")
# Using a common parameter for quick training, not the result of GridSearchCV
dummy_knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
dummy_knn_model.fit(X_train, y_train)
final_knn_model = dummy_knn_model # Assign to the variable expected by this step

print("\n--- Starting Evaluation of Baseline Models ---")

# --- Custom FNR calculation function (reiterated for completeness in this block) ---
def calculate_fnr(y_true, y_pred, healthy_class_label=0):
    """
    Calculates the False Negative Rate (FNR) for overall disease detection.
    FNR = FN / (TP + FN) * 100%
    This version treats 'healthy_class_label' as the negative class and all others as positive (diseased).
    """
    # Convert multi-class labels to binary: 0 for healthy, 1 for diseased
    y_true_binary = (y_true != healthy_class_label).astype(int)
    y_pred_binary = (y_pred != healthy_class_label).astype(int)

    # Compute binary confusion matrix
    # cm_binary structure:
    # [[True Negatives (TN), False Positives (FP)],  <- Actual Healthy
    #  [False Negatives (FN), True Positives (TP)]] <- Actual Diseased
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)

    # Extract FN and TP from the binary confusion matrix
    # Ensure indices are within bounds
    FN = cm_binary[1, 0] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 0 else 0
    TP = cm_binary[1, 1] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 1 else 0

    # Calculate FNR
    if (TP + FN) == 0:
        return 0.0 # Avoid division by zero
    fnr = (FN / (TP + FN)) * 100
    return fnr

# --- Evaluate SVM Model ---
print("\nEvaluating SVM Baseline Model...")
y_pred_svm = final_svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
fnr_svm = calculate_fnr(y_test, y_pred_svm, healthy_class_label=0) # Assuming 0 is the healthy class

print(f"SVM Test Accuracy: {accuracy_svm:.2f}%")
print(f"SVM Test FNR (considering all diseases as positive): {fnr_svm:.2f}%")

# --- Evaluate KNN Model ---
print("\nEvaluating KNN Baseline Model...")
y_pred_knn = final_knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
fnr_knn = calculate_fnr(y_test, y_pred_knn, healthy_class_label=0) # Assuming 0 is the healthy class

print(f"KNN Test Accuracy: {accuracy_knn:.2f}%")
print(f"KNN Test FNR (considering all diseases as positive): {fnr_knn:.2f}%")

print("\n--- Baseline Model Evaluation Completed ---")

# You can store these results for later comparison with the KD model
baseline_results = {
    "SVM_Accuracy": accuracy_svm,
    "SVM_FNR": fnr_svm,
    "KNN_Accuracy": accuracy_knn,
    "KNN_FNR": fnr_knn
}
print("\nBaseline Results Summary:")
print(baseline_results)