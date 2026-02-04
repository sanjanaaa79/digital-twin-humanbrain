%%writefile /content/drive/MyDrive/NeuroTwin/project/data_loader.py

import pickle
import numpy as np
import os
import glob

def load_data_file(file_path):

    with open(file_path, 'rb') as f:
        content = pickle.load(f, encoding='latin1')

    return content

def load_all_data(data_dir, subjects=None):

    data_list = []
    labels_list = []

    if subjects:
        file_paths = [os.path.join(data_dir, f"s{s}.dat") for s in subjects]
    else:
        file_paths = glob.glob(os.path.join(data_dir, "s*.dat"))

    file_paths.sort()

    for file_path in file_paths:

        if not os.path.exists(file_path):
            continue

        print(f"Loading {os.path.basename(file_path)}")

        content = load_data_file(file_path)

        eeg_data = content['data'][:, :32, :]

        data_list.append(eeg_data)
        labels_list.append(content['labels'])

    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    return X, y
