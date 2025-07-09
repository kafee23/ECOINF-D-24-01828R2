import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- Assume data, pre-trained teacher_model, and defined student_model are available ---
# Re-creating dummy data and models for a runnable standalone example.
# In your actual implementation, use the variables from previous steps.

# Dummy Data Generation (as in previous steps)
num_samples = 1000
num_features = 100
num_classes = 8 # Anthracnose, Bacterial Canker, etc., and Healthy

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, num_classes, num_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Scale features and One-Hot Encode labels
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Dummy Teacher Model (as in Step 3.1)
teacher_model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
teacher_model.compile(optimizer=Adam(), loss=CategoricalCrossentropy())
# Train dummy teacher model (for a real run, ensure it's properly trained for accuracy)
teacher_model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, verbose=0)

# Dummy Student Model (as in Step 3.2)
student_model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# --- Get Soft Targets from the trained Teacher Model ---
# These are the teacher's predictions (probabilities) on the training data.
teacher_soft_targets = teacher_model.predict(X_train_scaled)


# --- Custom Knowledge Distillation Loss Function ---
class KnowledgeDistillationLoss(tf.keras.losses.Loss):
    def __init__(self, teacher_model, temperature, alpha, fnr_penalty_weights=None, fnr_threshold=0.5, name="kd_loss"):
        super().__init__(name=name)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha # Regularization weight to balance hard and soft loss
        self.hard_loss_fn = CategoricalCrossentropy(from_logits=False) # For student's hard labels
        self.kl_divergence_loss_fn = tf.keras.losses.KLDivergence() # For soft targets (teacher vs student)
        self.fnr_penalty_weights = fnr_penalty_weights if fnr_penalty_weights is not None else [0.0] * num_classes # Lambda_i for each class [cite: 682]
        self.fnr_threshold = fnr_threshold # Delta for FNR penalty [cite: 682]

    def call(self, y_true, y_pred_student):
        # 1. Student's loss on hard labels (categorical cross-entropy)
        hard_loss = self.hard_loss_fn(y_true, y_pred_student)

        # Get teacher's softened probabilities
        # We need the teacher's logits to apply temperature, then softmax
        # For simplicity, we assume teacher_soft_targets are already probabilities.
        # If teacher_model.predict() gives logits, adjust:
        # teacher_logits = self.teacher_model(X_input_tensor, training=False) # Needs X_input_tensor
        # teacher_soft_probs_with_temp = tf.nn.softmax(teacher_logits / self.temperature)

        # For this setup, we already have teacher_soft_targets (probabilities).
        # We need to re-softmax student predictions with temperature.
        # However, Keras loss functions typically expect raw logits or probabilities.
        # When using from_logits=True in CategoricalCrossentropy, it handles softmax internally.
        # For KD loss, we need softened *probabilities*.

        # To correctly apply temperature to both teacher and student outputs:
        # We need access to the student's *logits* to apply temperature before softmax for KL divergence.
        # The `y_pred_student` from `call` is usually after softmax if the model's last layer is softmax.
        # To get logits, we would need to modify the student_model to output logits
        # and then apply softmax with temperature.

        # Let's assume for simplicity in this loss function that `y_pred_student` are logits
        # coming directly from the layer before the final softmax in the student model.
        # And teacher_soft_targets should be teacher's logits.
        # In a real Keras custom training loop or custom Model, you'd pass logits.

        # For now, let's work with probabilities and apply temperature to them,
        # acknowledging this is a simplification for a standalone loss function.
        # A more robust way involves custom training loops or wrapping models.

        # Correct approach for KD in Keras usually involves:
        # 1. Teacher outputs logits.
        # 2. Student outputs logits.
        # 3. Apply temperature to both logits, then softmax.
        # 4. Compute KL divergence.

        # Given `y_pred_student` is usually softmax output (probabilities):
        # To simulate logits for temperature scaling, we can inverse softmax (log) or
        # directly apply temperature to log-probabilities (tf.math.log(y_pred_student)).
        # Let's adjust student's output for KD loss by taking log and applying temperature.
        # And assume teacher_soft_targets are already the soft probabilities at desired temperature from teacher.

        # Re-implementing with logits assumption for `y_pred_student` for KL div
        # A simple way for a Sequential model is to remove softmax from the last layer
        # during definition for KD, and add it back for final evaluation.
        # Or, during training, get logits and pass them.

        # Let's define the custom loss assuming `y_pred_student` are probabilities
        # (output of student_model's final softmax layer).
        # To get the effect of temperature:
        # Student's "softened" probabilities for KD
        # log_student_probs = tf.math.log(y_pred_student + K.epsilon()) # Add epsilon for numerical stability
        # student_soft_logits = log_student_probs / self.temperature
        # student_soft_probs = tf.nn.softmax(student_soft_logits)

        # For KL divergence, usually inputs are log-probabilities or probabilities.
        # `tf.keras.losses.KLDivergence` takes `y_true` (teacher's distribution) and `y_pred` (student's distribution).
        # So, we pass teacher's softened probabilities as `y_true_for_kl` and student's softened probabilities.

        # The paper's Equation 11 implies $p_T$ and $p_S$ are distributions from temperatured softmax.
        # To compute this correctly in Keras, a custom `Model` subclass is best,
        # where you get logits from both teacher and student.

        # As a workaround for a simple loss function:
        # Assume teacher_soft_targets are already "temperature-softened" probabilities.
        # Apply temperature to student logits BEFORE softmax to get student_soft_probs.
        # But `y_pred_student` is already softmaxed.
        # The standard way to implement this in Keras is to get teacher and student *logits*
        # and then apply temperature-scaled softmax.

        # Let's define a custom Keras Model for Knowledge Distillation for better control over logits.
        # This will be more robust than trying to shoehorn it into a simple Loss class `call` method.

        # The `build` method of custom Keras model subclass is the right place for this.
        # For `KnowledgeDistillationLoss` class, let's assume `y_pred_student` are *logits*
        # (i.e., the student model's last layer has no activation or 'linear' activation).
        # And `teacher_soft_targets` passed to `call` will be the teacher's *logits*.

        teacher_logits_t = tf.constant(teacher_soft_targets, dtype=tf.float32) # Assuming teacher_soft_targets are logits from teacher
        student_logits_t = y_pred_student # Assuming y_pred_student are logits from student

        # Apply temperature scaling to logits, then softmax
        teacher_soft_probs = tf.nn.softmax(teacher_logits_t / self.temperature)
        student_soft_probs = tf.nn.softmax(student_logits_t / self.temperature)

        # KL Divergence loss 
        # tf.keras.losses.KLDivergence expects probabilities, and computes D_KL(y_true || y_pred).
        # So y_true for KL is teacher_soft_probs, y_pred is student_soft_probs.
        soft_loss = self.kl_divergence_loss_fn(teacher_soft_probs, student_soft_probs) * (self.temperature ** 2)
        # Scale KL loss by T^2 as commonly done 

        # 2. Class-wise penalty function to minimize false negatives (Equation 12) 
        # L_FNR = sum(lambda_i * max(0, delta - p_S(y_i))) 
        # p_S(y_i) is the student's probability for the true class.
        # y_true is one-hot encoded. So y_true * y_pred_student gives the probability for the true class.
        true_class_probs_student = tf.reduce_sum(y_true * y_pred_student, axis=-1)

        fnr_penalty = 0.0
        # Iterate over classes to apply individual penalty weights
        for i in range(num_classes): # Loop through each class
            # Only apply penalty if y_true is for this class
            class_mask = tf.cast(tf.equal(tf.argmax(y_true, axis=1), i), tf.float32)
            
            # For each sample where true class is `i`, calculate penalty if prob < threshold
            penalty_term_per_sample = tf.maximum(0.0, self.fnr_threshold - true_class_probs_student) * class_mask

            # Sum up penalties for this class and apply lambda_i 
            fnr_penalty += self.fnr_penalty_weights[i] * tf.reduce_sum(penalty_term_per_sample)

        # 3. Combine all loss components 
        # The paper states: total_loss = alpha * Hard_Loss + (1-alpha) * Soft_Loss
        # And L_FNR is an additional component "added to the overall loss".
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + fnr_penalty
        return total_loss


# --- Create a custom Model for KD Training ---
# This approach is more robust as it allows passing both hard labels and teacher logits
# correctly within the Keras `fit` method.
class KDModel(Model):
    def __init__(self, student, teacher, temperature, alpha, fnr_penalty_weights, fnr_threshold):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.fnr_penalty_weights = fnr_penalty_weights
        self.fnr_threshold = fnr_threshold
        # Ensure teacher is not trainable during student training
        self.teacher.trainable = False

        self.hard_loss_fn = CategoricalCrossentropy(from_logits=False)
        self.kl_loss_fn = tf.keras.losses.KLDivergence()

    def compile(self, optimizer, metrics=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        # Metrics for student's hard predictions
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    def train_step(self, data):
        x, y_true_hard = data # x are features, y_true_hard are one-hot encoded labels

        with tf.GradientTape() as tape:
            # Get teacher's softened probabilities (logits from teacher model)
            teacher_logits = self.teacher(x, training=False)
            teacher_soft_probs = tf.nn.softmax(teacher_logits / self.temperature)

            # Get student's logits and softened probabilities
            student_logits = self.student(x, training=True)
            student_soft_probs = tf.nn.softmax(student_logits / self.temperature)
            student_hard_probs = tf.nn.softmax(student_logits) # For hard loss and FNR penalty

            # 1. Hard Loss: Student's predictions vs true labels
            hard_loss = self.hard_loss_fn(y_true_hard, student_hard_probs)

            # 2. KL Divergence Loss: Teacher's soft probs vs Student's soft probs 
            # Scale by T^2 as commonly done
            soft_loss = self.kl_loss_fn(teacher_soft_probs, student_soft_probs) * (self.temperature ** 2)

            # 3. Class-wise Penalty for False Negatives (Equation 12) 
            # p_S(y_i) is the student's probability for the true class.
            true_class_probs_student = tf.reduce_sum(y_true_hard * student_hard_probs, axis=-1)
            
            fnr_penalty = 0.0
            for i in range(num_classes):
                class_mask = tf.cast(tf.equal(tf.argmax(y_true_hard, axis=1), i), tf.float32)
                penalty_term_per_sample = tf.maximum(0.0, self.fnr_threshold - true_class_probs_student) * class_mask
                fnr_penalty += self.fnr_penalty_weights[i] * tf.reduce_sum(penalty_term_per_sample)

            # Combine all loss components 
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + fnr_penalty

        # Compute gradients and apply them
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update metrics (for hard predictions)
        self.accuracy_metric.update_state(y_true_hard, student_hard_probs)

        return {"loss": total_loss, "accuracy": self.accuracy_metric.result()}

    def test_step(self, data):
        # Evaluation is typically on hard predictions for accuracy/FNR
        x, y_true_hard = data
        student_hard_probs = self.student(x, training=False)
        student_logits = self.student(x, training=False) # Get logits for soft loss calculation in evaluation if needed

        # Calculate hard loss for evaluation
        hard_loss = self.hard_loss_fn(y_true_hard, student_hard_probs)

        # For evaluating the overall KD loss during testing:
        teacher_logits = self.teacher(x, training=False)
        teacher_soft_probs = tf.nn.softmax(teacher_logits / self.temperature)
        student_soft_probs = tf.nn.softmax(student_logits / self.temperature) # Use student's logits

        soft_loss = self.kl_loss_fn(teacher_soft_probs, student_soft_probs) * (self.temperature ** 2)

        true_class_probs_student = tf.reduce_sum(y_true_hard * student_hard_probs, axis=-1)
        fnr_penalty = 0.0
        for i in range(num_classes):
            class_mask = tf.cast(tf.equal(tf.argmax(y_true_hard, axis=1), i), tf.float32)
            penalty_term_per_sample = tf.maximum(0.0, self.fnr_threshold - true_class_probs_student) * class_mask
            fnr_penalty += self.fnr_penalty_weights[i] * tf.reduce_sum(penalty_term_per_sample)

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + fnr_penalty

        self.accuracy_metric.update_state(y_true_hard, student_hard_probs)

        return {"loss": total_loss, "accuracy": self.accuracy_metric.result()}


# --- Instantiate and Compile the KD Model ---
print("\n--- Compiling KD Model ---")

# Hyperparameters for Knowledge Distillation
temperature = 3.0 # Can be tuned (e.g., 2.0, 5.0, 10.0) 
alpha = 0.5       # Balance between hard and soft loss (0.0 to 1.0) 

# Class-wise penalty weights for FNR. Adjust these based on class importance.
# For example, if Anthracnose (label 1) FNR is critical, give it a higher weight.
# Here, we'll use equal weights for simplicity, but you can adjust per class.
# Example: [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] if class 1 is very important.
fnr_penalty_weights = [0.1] * num_classes # Example: equal penalty for all classes
fnr_threshold = 0.8 # Example: if student's confidence for true class is below 0.8, apply penalty [cite: 682]

# Create the KD training model
kd_model = KDModel(student=student_model, teacher=teacher_model,
                    temperature=temperature, alpha=alpha,
                    fnr_penalty_weights=fnr_penalty_weights,
                    fnr_threshold=fnr_threshold)

# Compile the KD model
# The optimizer and metrics apply to the student model's training process
kd_model.compile(optimizer=Adam(learning_rate=0.001))

# --- Train the KD Model ---
print("\n--- Training KD Model ---")

# The `fit` method of the custom `KDModel` will now use the custom `train_step`.
history_kd = kd_model.fit(
    X_train_scaled, y_train_encoded, # Pass hard labels here, KDModel handles soft labels internally
    epochs=100, # More epochs are often needed for KD to converge
    batch_size=32,
    validation_data=(X_test_scaled, y_test_encoded), # Use test data for validation check
    verbose=1
)

print("\n--- Knowledge Distillation Training Completed ---")

# The `student_model` within `kd_model` is now trained.
# You can extract it for further evaluation or deployment:
trained_student_model = kd_model.student