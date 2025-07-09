import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# Assume the `calculate_accuracy`, `calculate_false_negative_rate`,
# `calculate_sensitivity`, `calculate_specificity`, `calculate_class_fnr` functions
# from Step 4.1 are already defined and available.

# --- Important: Placeholder for actual data and predictions ---
# In your full pipeline, these variables would be populated by the preceding steps:
# X_test, y_test: Your actual test dataset features and true labels.
# y_pred_svm: Predicted labels from the best SVM model (from Step 2.3).
# y_pred_knn: Predicted labels from the best KNN model (from Step 2.3).
# y_pred_kd_student: Predicted labels from the KD-enhanced student model (from Step 3.4).
# healthy_class_label: The actual numerical label for the 'healthy' class in your dataset.
# num_classes: Total number of classes (e.g., 8 for MangoLeafBD dataset).
# class_names: A list of human-readable names for your classes (e.g., ['Healthy', 'Anthracnose', ...]).

# --- Example Sample (Bring from original model's saved value) Data and Predictions (REMOVE IN FINAL INTEGRATION) ---
# This block is for demonstration if you run this script in isolation.
# In a real scenario, model will bring actual data loading and model predictions.
if 'y_test' not in locals() or 'y_pred_svm' not in locals():
    print("Warning: Using Sample (Bring from original model's saved value) data for demonstration. Replace with actual data from previous steps.")
    num_samples_test = 200 # Example number of test samples
    num_classes = 8 # From the paper's dataset description
    y_test = np.random.randint(0, num_classes, num_samples_test) # Sample (Bring from original model's saved value) true labels
    y_pred_svm = np.random.randint(0, num_classes, num_samples_test) #  replace with actual value
    y_pred_knn = np.random.randint(0, num_classes, num_samples_test) # Sample (Bring from original model's saved value) KNN predictions
    y_pred_kd_student = np.random.randint(0, num_classes, num_samples_test) # Sample (Bring from original model's saved value) KD predictions
    healthy_class_label = 0 # Assuming 'healthy' is class 0, adjust if different in your data.
    # Example class names, replace with your actual class names (e.g., from encoder.categories_[0] if using OneHotEncoder)
    class_names = ['Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                   'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mold']
# --- END OF Sample (Bring from original model's saved value) DATA ---


# --- Define Evaluation Functions (Re-included for self-contained execution) ---
def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy."""
    return accuracy_score(y_true, y_pred) * 100

def calculate_false_negative_rate(y_true, y_pred, healthy_class_label):
    """Calculates the False Negative Rate (FNR) for overall disease detection."""
    y_true_binary = (y_true != healthy_class_label).astype(int)
    y_pred_binary = (y_pred != healthy_class_label).astype(int)
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    FN = cm_binary[1, 0] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 0 else 0
    TP = cm_binary[1, 1] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 1 else 0
    return (FN / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0

def calculate_sensitivity(y_true, y_pred, target_class_label):
    """
    Calculates Sensitivity (Recall) for a specific class.
    Sensitivity = TP / (TP + FN)
    """
    TP = np.sum((y_true == target_class_label) & (y_pred == target_class_label))
    FN = np.sum((y_true == target_class_label) & (y_pred != target_class_label))
    return (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0

def calculate_specificity(y_true, y_pred, target_class_label):
    """
    Calculates Specificity for a specific class.
    Specificity = TN / (TN + FP)
    """
    TN = np.sum((y_true != target_class_label) & (y_pred != target_class_label))
    FP = np.sum((y_true != target_class_label) & (y_pred == target_class_label))
    return (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0.0

def calculate_class_fnr(y_true, y_pred, target_class_label):
    """
    Calculates FNR for a specific class.
    FNR = FN / (TP + FN)
    """
    TP = np.sum((y_true == target_class_label) & (y_pred == target_class_label))
    FN = np.sum((y_true == target_class_label) & (y_pred != target_class_label))
    return (FN / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0


# --- Calculate Class-wise Metrics for Each Model ---
print("\n--- Calculating Class-wise Metrics for Detailed Table ---")

# Initialize dictionaries to store class-wise metrics
# The structure will be {Class: [list of metrics for SVM], ...}
# This prepares the data for a wide format table where each row is a class
# and columns are metrics for each model.

svm_metrics_list = []
knn_metrics_list = []
kd_metrics_list = []

for i, class_name in enumerate(class_names):
    # Calculate metrics for SVM
    acc_svm_class = calculate_accuracy(y_test, y_pred_svm) # Overall accuracy for SVM, not class-specific
    sens_svm_class = calculate_sensitivity(y_test, y_pred_svm, i)
    spec_svm_class = calculate_specificity(y_test, y_pred_svm, i)
    fnr_svm_class = calculate_class_fnr(y_test, y_pred_svm, i)
    svm_metrics_list.append([
        class_name,
        f"{acc_svm_class:.2f}%",
        f"{sens_svm_class:.2f}%",
        f"{spec_svm_class:.2f}%",
        f"{fnr_svm_class:.2f}%"
    ])

    # Calculate metrics for KNN
    acc_knn_class = calculate_accuracy(y_test, y_pred_knn) # Overall accuracy for KNN
    sens_knn_class = calculate_sensitivity(y_test, y_pred_knn, i)
    spec_knn_class = calculate_specificity(y_test, y_pred_knn, i)
    fnr_knn_class = calculate_class_fnr(y_test, y_pred_knn, i)
    knn_metrics_list.append([
        class_name,
        f"{acc_knn_class:.2f}%",
        f"{sens_knn_class:.2f}%",
        f"{spec_knn_class:.2f}%",
        f"{fnr_knn_class:.2f}%"
    ])

    # Calculate metrics for KD-Enhanced Student
    acc_kd_class = calculate_accuracy(y_test, y_pred_kd_student) # Overall accuracy for KD
    sens_kd_class = calculate_sensitivity(y_test, y_pred_kd_student, i)
    spec_kd_class = calculate_specificity(y_test, y_pred_kd_student, i)
    fnr_kd_class = calculate_class_fnr(y_test, y_pred_kd_student, i)
    kd_metrics_list.append([
        class_name,
        f"{acc_kd_class:.2f}%",
        f"{sens_kd_class:.2f}%",
        f"{spec_kd_class:.2f}%",
        f"{fnr_kd_class:.2f}%"
    ])

# Create DataFrames for each model's class-wise metrics
columns = ['Class', 'Accuracy', 'Sensitivity', 'Specificity', 'FNR']
svm_class_df = pd.DataFrame(svm_metrics_list, columns=columns)
knn_class_df = pd.DataFrame(knn_metrics_list, columns=columns)
kd_class_df = pd.DataFrame(kd_metrics_list, columns=columns)


# --- Combine Class-wise Metrics into a Single Table (like Paper's Table 3) ---
print("\n--- Combining Class-wise Metrics into a Single Table ---")

# Rename columns to distinguish between models before merging
svm_class_df_renamed = svm_class_df.rename(columns={
    'Accuracy': 'Accuracy_SVM',
    'Sensitivity': 'Sensitivity_SVM',
    'Specificity': 'Specificity_SVM',
    'FNR': 'FNR_SVM'
})

knn_class_df_renamed = knn_class_df.rename(columns={
    'Accuracy': 'Accuracy_KNN',
    'Sensitivity': 'Sensitivity_KNN',
    'Specificity': 'Specificity_KNN',
    'FNR': 'FNR_KNN'
})

kd_class_df_renamed = kd_class_df.rename(columns={
    'Accuracy': 'Accuracy_KD',
    'Sensitivity': 'Sensitivity_KD',
    'Specificity': 'Specificity_KD',
    'FNR': 'FNR_KD'
})

# Merge the DataFrames on the 'Class' column
# Start with SVM, then merge KNN, then KD
combined_class_table = pd.merge(svm_class_df_renamed, knn_class_df_renamed, on='Class')
combined_class_table = pd.merge(combined_class_table, kd_class_df_renamed, on='Class')

# Reorder columns to match the paper's Table 3 structure (optional but good for visual comparison)
# The paper's Table 3 has: Disease | Type | Accuracy (SVM, KNN, KD) | Sensitivity (SVM, KNN, KD) | ...
# We'll adapt it to: Class | Accuracy (SVM, KNN, KD) | Sensitivity (SVM, KNN, KD) | Specificity (SVM, KNN, KD) | FNR (SVM, KNN, KD)
ordered_columns = ['Class',
                   'Accuracy_SVM', 'Accuracy_KNN', 'Accuracy_KD',
                   'Sensitivity_SVM', 'Sensitivity_KNN', 'Sensitivity_KD',
                   'Specificity_SVM', 'Specificity_KNN', 'Specificity_KD',
                   'FNR_SVM', 'FNR_KNN', 'FNR_KD']

# Check if all ordered_columns exist in combined_class_table before reordering
# If a column doesn't exist (e.g., due to Sample (Bring from original model's saved value) data issues), print a warning.
final_columns = [col for col in ordered_columns if col in combined_class_table.columns]
combined_class_table = combined_class_table[final_columns]

print("\n--- Combined Class-wise Metrics Table ---")
# Use to_string() for pretty printing in console, without index
print(combined_class_table.to_string(index=False))

print("\n--- Reporting Section Completed ---")