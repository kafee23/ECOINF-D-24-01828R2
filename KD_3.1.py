import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- Assume data is loaded and split from previous steps ---
# For demonstration, let's re-create the dummy data and split it.
# In your actual implementation, you would use X_train, X_test, y_train, y_test
# that you prepared in Part 1 and used in Part 2.

num_samples = 1000
num_features = 100
num_classes = 8 # Anthracnose, Bacterial Canker, etc., and Healthy (as in MangoLeafBD dataset) [cite: 499]

# Generate random features
X = np.random.rand(num_samples, num_features)
# Generate random labels
y = np.random.randint(0, num_classes, num_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Scale features and One-Hot Encode labels
# Feature scaling is important for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels for Keras CategoricalCrossentropy loss
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of y_train_encoded: {y_train_encoded.shape}\n")


# --- Define Teacher Model Architecture ---
print("--- Defining Teacher Model Architecture ---")

# A 'larger, more accurate' MLP model for the teacher
teacher_model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,), name='dense_1'), # Input layer + first hidden layer
    Dropout(0.3, name='dropout_1'), # Dropout for regularization
    Dense(128, activation='relu', name='dense_2'), # Second hidden layer
    Dropout(0.3, name='dropout_2'),
    Dense(64, activation='relu', name='dense_3'),  # Third hidden layer
    Dense(num_classes, activation='softmax', name='output_layer') # Output layer for multi-class classification
])

# Compile the teacher model
# Using Adam optimizer, CategoricalCrossentropy for loss, and CategoricalAccuracy for metrics
teacher_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss=CategoricalCrossentropy(),
                      metrics=[CategoricalAccuracy()])

# Display model summary
teacher_model.summary()

# --- Train Teacher Model ---
print("\n--- Training Teacher Model ---")

# Train the teacher model on the scaled training data with one-hot encoded labels
history_teacher = teacher_model.fit(
    X_train_scaled, y_train_encoded,
    epochs=50, # Number of epochs can be tuned; more epochs might mean more accuracy
    batch_size=32, # Batch size can be tuned
    validation_split=0.1, # Use a small validation split from training data to monitor overfitting
    verbose=1 # Show training progress
)

# --- Evaluate Teacher Model on Test Set (for performance check) ---
print("\n--- Evaluating Teacher Model on Test Set ---")
loss_teacher, accuracy_teacher = teacher_model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f"Teacher Model Test Loss: {loss_teacher:.4f}")
print(f"Teacher Model Test Accuracy: {accuracy_teacher*100:.2f}%")


# --- Obtain Soft Predictions (Probabilities) from Teacher Model ---
print("\n--- Obtaining Soft Predictions from Teacher Model ---")

# The "soft targets" are the probabilities predicted by the teacher model on the training data.
# These predictions are typically "softened" by applying softmax with a temperature parameter.
# For now, we'll get raw probabilities, and temperature will be applied in KD loss in Step 3.3.
teacher_soft_targets = teacher_model.predict(X_train_scaled)

print(f"Shape of Teacher Soft Targets: {teacher_soft_targets.shape}")
print("Sample Teacher Soft Targets (first 5):\n", teacher_soft_targets[:5])

print("\n--- Teacher Model Defined, Trained, and Soft Targets Obtained ---")