import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- Definition of Evaluation Functions ---

def calculate_accuracy(y_true, y_pred):
    """
    [cite_start]Calculates the accuracy of the model. [cite: 599]
    [cite_start]Accuracy = (TP + TN) / (TP + TN + FP + FN) * 100% [cite: 601]

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Accuracy percentage.
    """
    # Use sklearn's accuracy_score for simplicity and robustness
    # This directly implements (TP + TN) / (Total Samples)
    [cite_start]return np.mean(y_true == y_pred) * 100 [cite: 599, 601]


def calculate_false_negative_rate(y_true, y_pred, healthy_class_label=0):
    """
    Calculates the False Negative Rate (FNR) for overall disease detection.
    [cite_start]FNR = FN / (TP + FN) * 100% [cite: 609]

    A low FNR is crucial for disease detection systems as it signifies that
    [cite_start]a limited quantity of diseased leaves are erroneously classified as healthy leaves. [cite: 606, 607]

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        healthy_class_label (int): The label corresponding to the 'healthy' class.
                                   [cite_start]All other labels are considered 'diseased'. [cite: 603]

    Returns:
        float: False Negative Rate percentage.
    """
    # Convert multi-class labels to binary: 0 for healthy, 1 for diseased
    y_true_binary = (y_true != healthy_class_label).astype(int)
    y_pred_binary = (y_pred != healthy_class_label).astype(int)

    # Compute binary confusion matrix
    # cm_binary structure:
    # [[True Negatives (TN), False Positives (FP)],  <- Actual Healthy
    #  [False Negatives (FN), True Positives (TP)]] <- Actual Diseased
    [cite_start]cm_binary = confusion_matrix(y_true_binary, y_pred_binary) [cite: 603]

    # Extract FN and TP from the binary confusion matrix
    # Safely access elements in case a class is entirely absent in predictions/truths
    # [cite_start]FN (False Negative) is the number of diseased images that are incorrectly categorized as healthy. [cite: 603]
    # [cite_start]TP (True Positive) represents correctly identified diseased images that are actually from the disease class. [cite: 603]
    [cite_start]FN = cm_binary[1, 0] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 0 else 0 [cite: 603]
    [cite_start]TP = cm_binary[1, 1] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 1 else 0 [cite: 603]

    # [cite_start]Calculate FNR [cite: 609]
    if (TP + FN) == 0:
        return 0.0 # Avoid division by zero
    [cite_start]fnr = (FN / (TP + FN)) * 100 [cite: 609]
    return fnr

print("--- Evaluation Functions Defined ---")