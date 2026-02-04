import pickle
import numpy as np
import os
import glob

def load_data_file(file_path):
    """
    Load a single DEAP data file.
    
    Args:
        file_path (str): Path to the .dat file.
        
    Returns:
        dict: Dictionary containing 'data' and 'labels'.
              data shape: (40, 40, 8064) - (trials, channels, samples)
              labels shape: (40, 4) - (trials, labels)
    """
    with open(file_path, 'rb') as f:
        content = pickle.load(f, encoding='latin1')
    return content

def load_all_data(data_dir, subjects=None):
    """
    Load data for all or specified subjects.
    
    Args:
        data_dir (str): Directory containing .dat files.
        subjects (list): List of subject IDs to load (e.g., ['01', '02']). If None, load all.
        
    Returns:
        np.array, np.array: Combined data and labels.
    """
    data_list = []
    labels_list = []
    
    # search for s*.dat patterns
    if subjects:
        file_paths = [os.path.join(data_dir, f"s{s}.dat") for s in subjects]
    else:
        file_paths = glob.glob(os.path.join(data_dir, "s*.dat"))
    
    file_paths.sort()
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        print(f"Loading {os.path.basename(file_path)}...")
        content = load_data_file(file_path)
        
        # Extract EEG channels (0-31) and Data
        # shape is (40, 40, 8064)
        # We only want the first 32 channels (EEG)
        eeg_data = content['data'][:, :32, :]
        
        data_list.append(eeg_data)
        labels_list.append(content['labels'])
        
    if not data_list:
        raise ValueError("No data loaded. Check directory path and files.")
        
    # Concatenate all subjects
    # Shape: (N_subjects * 40, 32, 8064)
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    return X, y

if __name__ == "__main__":
    # Test the loader
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root dir
    try:
        # Load just one subject to test
        X, y = load_all_data(data_dir, subjects=['01'])
        print(f"Loaded Data Shape: {X.shape}")
        print(f"Loaded Labels Shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")
