import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# Assume the `calculate_accuracy` and `calculate_false_negative_rate` functions
# from Step 4.1 are already defined and available.

# --- Important: Placeholder for actual data and predictions ---
# In your full pipeline, these variables would be populated by the preceding steps:
# [cite_start]X_test, y_test: Your actual test dataset features and true labels. [cite: 57]
# y_pred_svm: Predicted labels from the best SVM model (from Step 2.3).
# y_pred_knn: Predicted labels from the best KNN model (from Step 2.3).
# y_pred_kd_student: Predicted labels from the KD-enhanced student model (from Step 3.4).
# [cite_start]healthy_class_label: The actual numerical label for the 'healthy' class in your dataset. [cite: 499]
# [cite_start]num_classes: Total number of classes (e.g., 8 for MangoLeafBD dataset). [cite: 499]
# class_names: A list of human-readable names for your classes (e.g., ['Healthy', 'Anthracnose', ...]).

# --- Example Dummy Data and Predictions (REMOVE IN FINAL INTEGRATION) ---
# This block is for demonstration if you run this script in isolation.
# In a real scenario, these would come from your actual data loading and model predictions.
if 'y_test' not in locals() or 'y_pred_svm' not in locals():
    print("Warning: Using dummy data for demonstration. Replace with actual data from previous steps.")
    num_samples_test = 200 # Example number of test samples
    [cite_start]num_classes = 8 # From the paper's dataset description [cite: 499]
    y_test = np.random.randint(0, num_classes, num_samples_test) # Dummy true labels
    y_pred_svm = np.random.randint(0, num_classes, num_samples_test) # Dummy SVM predictions
    y_pred_knn = np.random.randint(0, num_classes, num_samples_test) # Dummy KNN predictions
    y_pred_kd_student = np.random.randint(0, num_classes, num_samples_test) # Dummy KD predictions
    [cite_start]healthy_class_label = 0 # Assuming 'healthy' is class 0, adjust if different in your data. [cite: 499]
    # Example class names, replace with your actual class names (e.g., from encoder.categories_[0])
    class_names = ['Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                   'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mold']
# --- END OF DUMMY DATA ---


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

# --- Additional Metric Functions for Class-wise Analysis ---
def calculate_sensitivity(y_true, y_pred, target_class_label):
    """
    Calculates Sensitivity (Recall) for a specific class.
    Sensitivity = TP / (TP + FN)
    """
    # TP for target class: true label is target, predicted is target
    TP = np.sum((y_true == target_class_label) & (y_pred == target_class_label))
    # FN for target class: true label is target, predicted is NOT target
    FN = np.sum((y_true == target_class_label) & (y_pred != target_class_label))
    return (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0

def calculate_specificity(y_true, y_pred, target_class_label):
    """
    Calculates Specificity for a specific class.
    Specificity = TN / (TN + FP)
    """
    # TN for target class: true label is NOT target, predicted is NOT target
    TN = np.sum((y_true != target_class_label) & (y_pred != target_class_label))
    # FP for target class: true label is NOT target, predicted IS target
    FP = np.sum((y_true != target_class_label) & (y_pred == target_class_label))
    return (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0.0

def calculate_class_fnr(y_true, y_pred, target_class_label):
    """
    Calculates FNR for a specific class.
    FNR = FN / (TP + FN)
    """
    # TP for target class: true label is target, predicted is target
    TP = np.sum((y_true == target_class_label) & (y_pred == target_class_label))
    # FN for target class: true label is target, predicted is NOT target
    FN = np.sum((y_true == target_class_label) & (y_pred != target_class_label))
    return (FN / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0


# --- Calculate Overall Metrics for Each Model ---
print("\n--- Calculating Overall Performance Metrics ---")

# Store results for easier presentation
overall_results = []

# SVM Overall
acc_svm = calculate_accuracy(y_test, y_pred_svm)
fnr_overall_svm = calculate_false_negative_rate(y_test, y_pred_svm, healthy_class_label=healthy_class_label)
overall_results.append({'Model': 'SVM (Baseline)', 'Accuracy (%)': f"{acc_svm:.2f}", 'FNR (%)': f"{fnr_overall_svm:.2f}"})

# KNN Overall
acc_knn = calculate_accuracy(y_test, y_pred_knn)
fnr_overall_knn = calculate_false_negative_rate(y_test, y_pred_knn, healthy_class_label=healthy_class_label)
overall_results.append({'Model': 'KNN (Baseline)', 'Accuracy (%)': f"{acc_knn:.2f}", 'FNR (%)': f"{fnr_overall_knn:.2f}"})

# KD-Enhanced Student Model Overall
acc_kd = calculate_accuracy(y_test, y_pred_kd_student)
fnr_overall_kd = calculate_false_negative_rate(y_test, y_pred_kd_student, healthy_class_label=healthy_class_label)
overall_results.append({'Model': 'KD-Enhanced Student', 'Accuracy (%)': f"{acc_kd:.2f}", 'FNR (%)': f"{fnr_overall_kd:.2f}"})

# Convert to DataFrame for pretty printing
overall_results_df = pd.DataFrame(overall_results)

print("\n--- Overall Model Performance Summary ---")
print(overall_results_df.to_string(index=False))


# --- Calculate Class-wise Metrics for Each Model ---
print("\n--- Class-wise Model Performance Metrics (Detailed) ---")

# Ensure num_classes is correctly set from your data, not hardcoded.
# If `class_names` list is not available, you might generate numeric labels for printing
# For example: class_names = [f"Class {i}" for i in range(num_classes)]

# Initialize dictionaries to store class-wise metrics
svm_class_metrics = {'Class': class_names}
knn_class_metrics = {'Class': class_names}
kd_class_metrics = {'Class': class_names}

svm_class_metrics['Accuracy (%)'] = []
svm_class_metrics['Sensitivity (%)'] = []
svm_class_metrics['Specificity (%)'] = []
svm_class_metrics['FNR (%)'] = []

knn_class_metrics['Accuracy (%)'] = []
knn_class_metrics['Sensitivity (%)'] = []
knn_class_metrics['Specificity (%)'] = []
knn_class_metrics['FNR (%)'] = []

kd_class_metrics['Accuracy (%)'] = []
kd_class_metrics['Sensitivity (%)'] = []
kd_class_metrics['Specificity (%)'] = []
kd_class_metrics['FNR (%)'] = []


for i in range(num_classes):
    # SVM Class-wise
    svm_class_metrics['Accuracy (%)'].append(f"{calculate_accuracy(y_test, y_pred_svm):.2f}") # Overall accuracy, not class-specific in this context
    svm_class_metrics['Sensitivity (%)'].append(f"{calculate_sensitivity(y_test, y_pred_svm, i):.2f}")
    svm_class_metrics['Specificity (%)'].append(f"{calculate_specificity(y_test, y_pred_svm, i):.2f}")
    svm_class_metrics['FNR (%)'].append(f"{calculate_class_fnr(y_test, y_pred_svm, i):.2f}")

    # KNN Class-wise
    knn_class_metrics['Accuracy (%)'].append(f"{calculate_accuracy(y_test, y_pred_knn):.2f}") # Overall accuracy
    knn_class_metrics['Sensitivity (%)'].append(f"{calculate_sensitivity(y_test, y_pred_knn, i):.2f}")
    knn_class_metrics['Specificity (%)'].append(f"{calculate_specificity(y_test, y_pred_knn, i):.2f}")
    knn_class_metrics['FNR (%)'].append(f"{calculate_class_fnr(y_test, y_pred_knn, i):.2f}")

    # KD-Enhanced Student Class-wise
    kd_class_metrics['Accuracy (%)'].append(f"{calculate_accuracy(y_test, y_pred_kd_student):.2f}") # Overall accuracy
    kd_class_metrics['Sensitivity (%)'].append(f"{calculate_sensitivity(y_test, y_pred_kd_student, i):.2f}")
    kd_class_metrics['Specificity (%)'].append(f"{calculate_specificity(y_test, y_pred_kd_student, i):.2f}")
    kd_class_metrics['FNR (%)'].append(f"{calculate_class_fnr(y_test, y_pred_kd_student, i):.2f}")


# Create DataFrames for class-wise metrics
svm_class_df = pd.DataFrame(svm_class_metrics)
knn_class_df = pd.DataFrame(knn_class_metrics)
kd_class_df = pd.DataFrame(kd_class_metrics)

print("\n--- SVM Class-wise Metrics ---")
print(svm_class_df.to_string(index=False))

print("\n--- KNN Class-wise Metrics ---")
print(knn_class_df.to_string(index=False))

print("\n--- KD-Enhanced Student Class-wise Metrics ---")
print(kd_class_df.to_string(index=False))

# For a combined table like the paper's Table 3, you would merge these.
# This requires a more complex structure, but this provides the data.
# Example of combining (requires careful column naming and merging):
# combined_class_table = pd.merge(svm_class_df.add_suffix('_SVM'), knn_class_df.add_suffix('_KNN'), on='Class')
# combined_class_table = pd.merge(combined_class_table, kd_class_df.add_suffix('_KD'), on='Class')
# print("\n--- Combined Class-wise Metrics Table (Illustrative) ---")
# print(combined_class_table.to_string(index=False))

print("\n--- Reporting Section Completed ---")