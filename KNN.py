import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore') # Ignore warnings, especially from GridSearchCV

# --- Assume data is loaded from Part 1 ---
# This part is a placeholder for actual data loading.
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}\n")


# --- KNN Implementation with Parameter Tuning ---

print("--- Starting KNN Training and Parameter Tuning ---")

# Define the parameter grid for GridSearchCV
# 'n_neighbors': The 'k' value, typically an odd number to avoid ties in binary classification
#                For multi-class, it's still good practice but not strictly required.
#                We'll explore a range of values.
# 'metric': The distance metric to use.
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], # Common range for k, adjust as needed
    'metric': ['euclidean', 'manhattan', 'minkowski'] # Common distance metrics. Minkowski with p=2 is Euclidean, p=1 is Manhattan.
    # 'p': [1, 2] # If 'minkowski' is chosen, 'p' parameter defines the power.
                  # p=1 is Manhattan (L1), p=2 is Euclidean (L2)
}

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV for KNN
# cv=5 for 5-fold cross-validation
# verbose=3 for detailed output
# n_jobs=-1 to use all available CPU cores
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, verbose=3, n_jobs=-1, scoring='accuracy')

# Fit GridSearchCV to the training data
print("Performing Grid Search for KNN hyperparameters...")
grid_search_knn.fit(X_train, y_train)

# Get the best parameters and best score
best_knn_params = grid_search_knn.best_params_
best_knn_score = grid_search_knn.best_best_score_

print("\n--- KNN Tuning Results ---")
print(f"Best parameters found for KNN: {best_knn_params}")
print(f"Best cross-validation accuracy for KNN: {best_knn_score:.4f}")

# Train the final KNN model with the best parameters
final_knn_model = grid_search_knn.best_estimator_

# --- Evaluation on the Test Set ---
print("\n--- Evaluating Final KNN Model on Test Set ---")

y_pred_knn = final_knn_model.predict(X_test)

# Custom function to calculate False Negative Rate (FNR)
# Reusing the FNR function from the SVM section
def calculate_fnr(y_true, y_pred):
    """
    Calculates the False Negative Rate (FNR).
    FNR = FN / (TP + FN) * 100%
    This version aggregates FNR for all 'diseased' classes vs 'healthy'.
    Assumes class 0 is 'healthy' and other classes (1 to N-1) are 'diseased'.
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    TP_overall = 0
    FN_overall = 0

    # Sum TP and FN for all disease classes
    for i in range(1, num_classes): # Iterate through disease classes (positive classes)
        TP_overall += cm[i, i] # True Positives for class i
        FN_overall += np.sum(cm[i, :]) - cm[i, i] # False Negatives for class i (actual for class i but predicted otherwise)

    # For overall FNR where 'positive' means *any* disease
    # We need to make sure 'healthy' (class 0) is treated as 'negative'.
    # A true negative (TN) for disease classes is correctly predicting healthy.
    # A false positive (FP) for disease classes is predicting disease when it's healthy.
    # To be precise on FNR=FN/(TP+FN), where Positive is *being diseased*:
    # TP = Sum of correctly classified diseased samples
    # FN = Sum of diseased samples classified as healthy OR as a different disease
    # This requires summing correctly classified diseased instances and instances that *were* diseased but were misclassified.

    # Let's adjust for a more direct interpretation of FN for 'diseased' vs 'healthy'
    # If healthy is class 0, and any other class is diseased:
    y_true_binary = (y_true != 0).astype(int) # 1 for diseased, 0 for healthy
    y_pred_binary = (y_pred != 0).astype(int) # 1 for predicted diseased, 0 for predicted healthy

    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    # cm_binary structure:
    # [[TN, FP],
    #  [FN, TP]]
    
    TN = cm_binary[0, 0]
    FP = cm_binary[0, 1]
    FN = cm_binary[1, 0]
    TP = cm_binary[1, 1]

    if (TP + FN) == 0:
        return 0.0 # Avoid division by zero
    fnr = (FN / (TP + FN)) * 100
    return fnr


# Accuracy: (TP + TN) / (TP + TN + FP + FN) * 100%
accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
# FNR: FN / (TP + FN) * 100% (using the adjusted binary FNR for overall disease detection)
fnr_knn = calculate_fnr(y_test, y_pred_knn) # Assuming healthy is class 0, others are diseases

print(f"KNN Test Accuracy: {accuracy_knn:.2f}%")
print(f"KNN Test FNR (considering all diseases as positive): {fnr_knn:.2f}%")

print("\n--- KNN Training and Evaluation Completed ---")