import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- Assume data, pre-trained teacher_model, and defined student_model are available ---
# Re-creating dummy data and models for a runnable standalone example.
# In your actual implementation, ensure X_test_scaled, y_test are available
# from Part 1/2.1, and `trained_student_model` is the output from Step 3.3.

# Dummy Data Generation (as in previous steps)
num_samples = 1000
num_features = 100
num_classes = 8 # Anthracnose, Bacterial Canker, etc., and Healthy

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, num_classes, num_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Scale features (using the same scaler fitted on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For evaluation, we typically use original integer labels for confusion matrix,
# but the model's output will be probabilities.
# y_test remains as integer labels for FNR calculation.

# --- Dummy trained_student_model ---
# In a real run, this would be the 'trained_student_model' object
# from the end of Step 3.3. Here, we create a simple one for demonstration.
trained_student_model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') # Student model outputs probabilities
])
# Simulate a trained model by compiling and fitting briefly
trained_student_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
trained_student_model.fit(X_train_scaled, y_train, epochs=5, verbose=0) # Train with hard labels for simplicity


# --- Custom FNR calculation function (reiterated for completeness) ---
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
    # Safely access elements in case a class is entirely absent in predictions/truths
    FN = cm_binary[1, 0] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 0 else 0
    TP = cm_binary[1, 1] if cm_binary.shape[0] > 1 and cm_binary.shape[1] > 1 else 0

    # Calculate FNR
    if (TP + FN) == 0:
        return 0.0 # Avoid division by zero
    fnr = (FN / (TP + FN)) * 100
    return fnr


# --- Evaluate KD-Enhanced Student Model ---
print("\n--- Evaluating KD-Enhanced Student Model on Test Set ---")

# Make predictions (probabilities) on the scaled test data
y_pred_probs_student = trained_student_model.predict(X_test_scaled, verbose=0)

# Convert probabilities to predicted class labels (hard predictions)
y_pred_labels_student = np.argmax(y_pred_probs_student, axis=1)

# Calculate Accuracy
accuracy_student = accuracy_score(y_test, y_pred_labels_student) * 100

# Calculate False Negative Rate (FNR)
# Assuming class 0 is the 'healthy' class, all others are 'diseased'.
fnr_student = calculate_fnr(y_test, y_pred_labels_student, healthy_class_label=0)

print(f"KD-Enhanced Student Model Test Accuracy: {accuracy_student:.2f}%")
print(f"KD-Enhanced Student Model Test FNR (overall diseased detection): {fnr_student:.2f}%")

print("\n--- KD-Enhanced Model Evaluation Completed ---")

# You can also store these results for a final comparison table
kd_results = {
    "KD_Accuracy": accuracy_student,
    "KD_FNR": fnr_student
}
print("\nKD Model Results Summary:")
print(kd_results)