import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# No need for train_test_split, StandardScaler, OneHotEncoder in this block,
# as they are assumed to be handled in previous steps.

# --- Assume necessary variables from previous steps are available ---
# num_features: (e.g., 100) - number of input features
# num_classes: (e.g., 8) - number of output classes

# For demonstration, re-define these if running this block independently:
num_features = 100 # Example: from your feature extraction [cite: 55, 56, 66]
num_classes = 8    # Example: from MangoLeafBD dataset [cite: 34, 499]

print("--- Defining Student Model Architecture ---")

# A 'smaller, computationally lighter' MLP model for the student
# Reduced number of layers and/or neurons compared to the teacher model
student_model = Sequential([
    # Input layer + first hidden layer: Fewer neurons than teacher's first layer
    Dense(64, activation='relu', input_shape=(num_features,), name='student_dense_1'),
    # Dropout can be reduced or removed for a smaller model, or kept if still prone to overfitting
    Dropout(0.2, name='student_dropout_1'),
    # Second hidden layer (optional, but keeping one for a bit of depth):
    Dense(32, activation='relu', name='student_dense_2'),
    # Output layer: Must match the number of classes and use 'softmax' for probabilities
    Dense(num_classes, activation='softmax', name='student_output_layer')
])

# Display student model summary
student_model.summary()

print("\n--- Student Model Architecture Defined ---")