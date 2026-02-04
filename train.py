%%writefile /content/drive/MyDrive/NeuroTwin/project/train.py

import sys
sys.path.append('/content/drive/MyDrive/NeuroTwin/project')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_data
from preprocessing import apply_filter, normalize_data, map_labels, flatten_eeg
from model import build_ann_model, compile_model

# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    # Dataset path (Google Drive)
    data_dir = "/content/drive/MyDrive/NeuroTwin/data"

    print("Loading EEG data...")
    X_raw, y_raw = load_all_data(data_dir, subjects=['01','02','03'])

    print("Preprocessing EEG...")
    X = normalize_data(apply_filter(X_raw))

    print("Mapping Cognitive States...")
    y = map_labels(y_raw, task='multi_class')

    print("Flattening EEG...")
    X = flatten_eeg(X)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Building ANN...")
    model = build_ann_model((X_train.shape[1],), 4)
    model = compile_model(model)

    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    print("\nEvaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n==============================")
    print("FINAL TEST ACCURACY:", round(acc*100,2), "%")
    print("==============================")

if __name__ == "__main__":
    main()
