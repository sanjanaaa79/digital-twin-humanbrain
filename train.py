import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Import local modules
from data_loader import load_all_data
from preprocessing import apply_filter, normalize_data, map_labels, segment_data
from model import build_ann_model, compile_model

def run_experiment(X, y, task_name, num_classes):
    print(f"\n--- Starting Experiment: {task_name} ---")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Build Model
    input_shape = (X_train.shape[1],)
    model = build_ann_model(input_shape, num_classes)
    model = compile_model(model)
    
    # Train
    history = model.fit(X_train, y_train, 
                        epochs=5, 
                        batch_size=64, 
                        validation_split=0.1,
                        verbose=2)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{task_name} Test Accuracy: {acc*100:.2f}%")
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Detailed Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return acc

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = base_dir # Data is in root
    
    # 1. Load Data
    print("Loading Data...")
    try:
        # Load all subjects (s01-s32)
        # Note: If memory is an issue, reduce the list of subjects here or implementation a generator
        # For now, we try all.
        X_raw, y_raw = load_all_data(data_dir)
        print(f"Raw Data Loaded: {X_raw.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Preprocess Common Steps
    print("Preprocessing: Filtering & Normalizing...")
    X_filt = apply_filter(X_raw)
    X_norm = normalize_data(X_filt)
    
    # 3. Experiment 1: Binary Arousal
    print("\nPreparing Binary Arousal Data...")
    y_bin = map_labels(y_raw, 'binary_arousal')
    # Use only 1-second windows
    X_seg_bin, y_seg_bin = segment_data(X_norm, y_bin, window_size=128, overlap=64) 
    # Added overlap for more data
    
    run_experiment(X_seg_bin, y_seg_bin, "Binary Arousal Classification", num_classes=2)
    
    # 4. Experiment 2: Multi-class Cognitive State
    print("\nPreparing Multi-class Data...")
    y_multi = map_labels(y_raw, 'multi_class')
    X_seg_multi, y_seg_multi = segment_data(X_norm, y_multi, window_size=128, overlap=64)
    
    run_experiment(X_seg_multi, y_seg_multi, "Multi-class Cognitive State Prediction", num_classes=4)

if __name__ == "__main__":
    main()
