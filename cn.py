import tensorflow as tf
import numpy as np
import csv
import os

# --- 1. Configuration ---
DATASET_DIR = r"C:\Users\monika\Desktop\image classifier_N\datasetN"
MODEL_PATH = 'classifier_engine.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Load Model and Test Data ---
print("Loading model and test dataset...")
model = tf.keras.models.load_model(MODEL_PATH)

# Use the EXACT same seed and split as your training script
test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False # CRITICAL: Set shuffle to False to keep file paths aligned with labels
)

class_names = test_dataset.class_names
file_paths = test_dataset.file_paths

# --- 3. Generate Predictions ---
print("Predicting classes for test data...")
# predictions will be a list of probability arrays
predictions = model.predict(test_dataset)
predicted_indices = np.argmax(predictions, axis=1)

# Get the true labels from the dataset
# We concatenate all batches to get a single list of true indices
true_indices = np.concatenate([y for x, y in test_dataset], axis=0)

# --- 4. Save to CSV ---
csv_filename = 'test_results_detailed.csv'
print(f"Saving results to {csv_filename}...")

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Header row
    writer.writerow(['File Path', 'True Label', 'Predicted Label', 'Status', 'Confidence'])

    for i in range(len(true_indices)):
        true_label = class_names[true_indices[i]]
        pred_label = class_names[predicted_indices[i]]
        status = "CORRECT" if true_label == pred_label else "WRONG"
        confidence = f"{np.max(predictions[i]) * 100:.2f}%"
        
        writer.writerow([file_paths[i], true_label, pred_label, status, confidence])

print(f"Done! Processed {len(true_indices)} images.")