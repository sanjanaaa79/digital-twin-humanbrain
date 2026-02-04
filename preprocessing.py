%%writefile /content/drive/MyDrive/NeuroTwin/project/preprocessing.py

import numpy as np
from scipy.signal import butter, lfilter

# ----------------------------------
# Bandpass Filter
# ----------------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_filter(data, lowcut=1.0, highcut=45.0, fs=128.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data, axis=-1)

# ----------------------------------
# Normalize EEG
# ----------------------------------

def normalize_data(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-6)

# ----------------------------------
# Cognitive State Mapping
# ----------------------------------

def map_labels(labels, task='multi_class'):

    valence = labels[:, 0]
    arousal = labels[:, 1]
    threshold = 5.0

    if task == 'binary_arousal':
        return np.where(arousal >= threshold, 1, 0)

    elif task == 'multi_class':

        y = np.zeros(len(labels), dtype=int)

        low_v = valence < threshold
        high_v = valence >= threshold
        low_a = arousal < threshold
        high_a = arousal >= threshold

        y[low_v & low_a] = 0     # Fatigue
        y[high_v & low_a] = 1   # Relaxed
        y[low_v & high_a] = 2   # Stress
        y[high_v & high_a] = 3  # Focused

        return y

# ----------------------------------
# Flatten EEG for ANN
# ----------------------------------

def flatten_eeg(data):
    return data.reshape(data.shape[0], -1)
