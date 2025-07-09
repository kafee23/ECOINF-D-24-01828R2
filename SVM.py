import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore') # Ignore warnings, especially from GridSearchCV

# --- Assume data is loaded from Part 1 ---
# For demonstration, let's create dummy data similar to what might come from feature extraction
# In a real scenario, you would load your 'combined_features.csv' or similar.

# Example: 1000 samples, 100 features, 8 classes (7 diseases + 1 healthy)
num_samples = 1000
num_features = 100
num_classes = 8 # Anthracnose, Bacterial Canker, etc., and Healthy

# Generate random features
X = np.random.rand(num_samples, num_features)
# Generate random labels
y = np.random.randint(0, num_classes, num_samples)

# Split the dataset into training and testing sets
# Using a 80/20 split as commonly done for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}\n")


# --- SVM Implementation with Parameter Tuning ---

print("--- Starting SVM Training and Parameter Tuning ---")

# Define the parameter grid for GridSearchCV
# These ranges cover various possibilities for C and gamma
# You can expand or narrow these ranges based on preliminary results or computational resources.
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    'kernel': ['rbf'] # As specified in the paper, RBF kernel is preferred for large feature sets
}

# Initialize the SVM classifier
svm = SVC()

# Initialize GridSearchCV
# cv=5 for 5-fold cross-validation as mentioned in the paper 
# verbose=3 to see the progress
# n_jobs=-1 to use all available CPU cores for faster computation
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=3, n_jobs=-1, scoring='accuracy')

# Fit GridSearchCV to the training data
print("Performing Grid Search for SVM hyperparameters...")
grid_search_svm.fit(X_train, y_train)

# Get the best parameters and best score
best_svm_params = grid_search_svm.best_params_
best_svm_score = grid_search_svm.best_score_

print("\n--- SVM Tuning Results ---")
print(f"Best parameters found for SVM: {best_svm_params}")
print(f"Best cross-validation accuracy for SVM: {best_svm_score:.4f}")

# Train the final SVM model with the best parameters
final_svm_model = grid_search_svm.best_estimator_

# --- Evaluation on the Test Set ---
print("\n--- Evaluating Final SVM Model on Test Set ---")

y_pred_svm = final_svm_model.predict(X_test)

# Custom function to calculate False Negative Rate (FNR)
def calculate_fnr(y_true, y_pred):
    """
    Calculates the False Negative Rate (FNR).
    FNR = FN / (TP + FN) * 100% [cite: 609]
    A low FNR is crucial for disease detection systems as it means fewer diseased
    leaves are incorrectly classified as healthy[cite: 607].
    """
    # Compute confusion matrix
    # The rows correspond to true labels and columns to predicted labels
    # If dealing with multi-class, sum values to get overall TP, TN, FP, FN
    # For FNR in multi-class, we consider each class's FN and TP.
    # A simplified approach for overall FNR might be needed, or class-wise FNR.
    # Let's provide a general FNR for binary case, and discuss multi-class later.
    # For now, we'll calculate FNR per class and average, or sum for overall 'diseased' vs 'healthy' if applicable.

    # For a general multi-class scenario, FNR is usually calculated per class
    # or as an aggregate if there's a specific 'positive' class (e.g., any disease vs. healthy).
    # Since the paper talks about 'diseased part of the image' and 'healthy leaves',
    # we'll consider all disease classes as 'positive' and 'healthy' as 'negative'.

    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    # Assuming class 0 is 'healthy' and classes 1 to N-1 are 'diseased'
    # This might need adjustment based on your actual label encoding.
    # For simplicity, if we want overall FNR, we treat all non-healthy as 'positive'.
    # If 0 is healthy, then:
    TP_overall = 0
    FN_overall = 0

    for i in range(1, num_classes): # Iterate through disease classes (positive classes)
        TP_overall += cm[i, i] # True Positives for class i
        FN_overall += np.sum(cm[i, :]) - cm[i, i] # False Negatives for class i

    if (TP_overall + FN_overall) == 0:
        return 0.0 # Avoid division by zero
    fnr = (FN_overall / (TP_overall + FN_overall)) * 100
    return fnr

# Accuracy: (TP + TN) / (TP + TN + FP + FN) * 100% [cite: 601]
accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
# FNR: FN / (TP + FN) * 100% [cite: 609]
fnr_svm = calculate_fnr(y_test, y_pred_svm) # Assuming healthy is class 0, others are diseases

print(f"SVM Test Accuracy: {accuracy_svm:.2f}%")
print(f"SVM Test FNR (considering all diseases as positive): {fnr_svm:.2f}%")

# To provide more granular FNR values as in the paper's Table 3,
# you would iterate through each class and calculate its FNR.
# For example, to get FNR for 'Anthracnose' specifically:
# You would need to know which label corresponds to Anthracnose.
# Let's assume Anthracnose is class 1 for example.

# def calculate_class_wise_fnr(y_true, y_pred, target_class_label):
#     tp_class = np.sum((y_true == target_class_label) & (y_pred == target_class_label))
#     fn_class = np.sum((y_true == target_class_label) & (y_pred != target_class_label))
#     if (tp_class + fn_class) == 0:
#         return 0.0
#     return (fn_class / (tp_class + fn_class)) * 100

# Example if you knew Anthracnose was label 1:
# fnr_anthracnose_svm = calculate_class_wise_fnr(y_test, y_pred_svm, 1)
# print(f"SVM FNR for Anthracnose: {fnr_anthracnose_svm:.2f}%")

print("\n--- SVM Training and Evaluation Completed ---")