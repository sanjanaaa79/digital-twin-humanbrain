import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut=1.0, highcut=45.0, fs=128.0, order=5):
    """
    Apply Bandpass filter to EEG data.
    
    Args:
        data (np.array): EEG data of shape (trials, channels, samples) or (channels, samples).
        
    Returns:
        np.array: Filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply along the time axis (last axis)
    y = lfilter(b, a, data, axis=-1)
    return y

def normalize_data(data):
    """
    Normalize data per trial and per channel to have zero mean and unit variance.
    Alternatively, could use MinMaxScaler. Here we use Standard extraction approach.
    """
    # data: (trials, channels, samples)
    # We want to normalize each channel of each trial independently or globally?
    # Usually per trial/channel to remove baseline drifts.
    
    # Reshape to (trials*channels, samples) for scaler? No, we want to normalize across time.
    # Mean/Std over the time dimension.
    
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    
    return (data - mean) / (std + 1e-6)

def map_labels(labels, task='binary_arousal'):
    """
    Map original DEAP labels (1-9) to specific classes.
    
    labels: (trials, 4) -> [Valence, Arousal, Dominance, Liking]
    """
    valence = labels[:, 0]
    arousal = labels[:, 1]
    
    threshold = 5.0
    
    if task == 'binary_arousal':
        # 0: Low Arousal, 1: High Arousal
        y = np.where(arousal >= threshold, 1, 0)
        return y
        
    elif task == 'multi_class':
        # 0: Fatigue (Low V, Low A)
        # 1: Relaxed (High V, Low A)
        # 2: Stress (Low V, High A)
        # 3: Focused (High V, High A)
        
        y = np.zeros(len(labels))
        
        # Vectorized mapping
        # Low V (<5), Low A (<5) -> Fatigue (0)
        # High V (>=5), Low A (<5) -> Relaxed (1)
        # Low V (<5), High A (>=5) -> Stress (2)
        # High V (>=5), High A (>=5) -> Focused (3)
        
        # We can iterate or use masks. Masks are cleaner.
        low_v = valence < threshold
        high_v = valence >= threshold
        low_a = arousal < threshold
        high_a = arousal >= threshold
        
        y[low_v & low_a] = 0 # Fatigue
        y[high_v & low_a] = 1 # Relaxed
        y[low_v & high_a] = 2 # Stress
        y[high_v & high_a] = 3 # Focused
        
        return y.astype(int)
    else:
        raise ValueError("Unknown task")

def segment_data(data, labels, window_size=128, overlap=0):
    """
    Segment trials into smaller windows using vectorized operations.
    
    Args:
        data: (trials, channels, samples)
        labels: (trials,)
        window_size: samples per window
        overlap: samples overlap
    
    Returns:
        X_segments: (total_segments, channels * window_size)
        y_segments: (total_segments,)
    """
    trials, channels, samples = data.shape
    stride = window_size - overlap
    
    # Calculate number of segments per trial
    n_segments = (samples - window_size) // stride + 1
    
    # Create windows using stride_tricks or simple slicing loop (but faster than per-sample)
    # Since we want to flatten afterwards, let's pre-allocate
    total_segments = trials * n_segments
    
    X_seg = np.zeros((total_segments, channels * window_size), dtype=np.float32)
    y_seg = np.zeros(total_segments, dtype=int)
    
    for i in range(trials):
        # Create a view of the current trial with windows
        # shape: (n_segments, channels, window_size)
        # We can implement a manual loop over segments which is faster than per-sample
        # But we can also use array operations.
        
        # Proper vectorization:
        # Create indices
        starts = np.arange(0, samples - window_size + 1, stride)
        
        # This loop runs 125 times per trial -> 40 trials -> 5000 times. 
        # Total segments for 32 subjects: 32 * 5000 = 160,000.
        # Faster to loop over trials and vector assignment.
        
        # Slice all segments for this trial
        # We can use a sliding window view, or just loop for now but with pre-allocation
        # A simple loop over starts is fine if we pre-allocate.
        
        for j, start in enumerate(starts):
            end = start + window_size
            segment = data[i, :, start:end]
            idx = i * n_segments + j
            X_seg[idx] = segment.flatten()
            y_seg[idx] = labels[i]
            
    return X_seg, y_seg

if __name__ == "__main__":
    # Test
    # Fake data
    data = np.random.randn(2, 32, 128*60) # 2 trials, 32 ch, 60s
    labels = np.array([[3, 3, 3, 3], [7, 7, 7, 7]]) # 1 Low/Low, 1 High/High
    
    # 1. Map Labels
    y_bin = map_labels(labels, 'binary_arousal')
    y_multi = map_labels(labels, 'multi_class')
    print(f"Binary Labels: {y_bin}")
    print(f"Multi Labels: {y_multi}")
    
    # 2. Filter
    data_filt = apply_filter(data)
    
    # 3. Normalize
    data_norm = normalize_data(data_filt)
    
    # 4. Segment
    X_seg, y_seg = segment_data(data_norm, y_multi, window_size=128)
    print(f"Segmented X shape: {X_seg.shape}")
    print(f"Segmented y shape: {y_seg.shape}")
