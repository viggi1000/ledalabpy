import numpy as np
from scipy import signal
from typing import Tuple

def butterworth_filter(data: np.ndarray, sampling_rate: float, cutoff: float, order: int = 4, 
                      filter_type: str = 'lowpass') -> np.ndarray:
    """
    Apply Butterworth filter to the data.
    
    Parameters:
    -----------
    data : ndarray
        Data vector
    sampling_rate : float
        Sampling rate in Hz
    cutoff : float or tuple
        Cutoff frequency in Hz (or tuple of frequencies for bandpass/bandstop)
    order : int
        Filter order
    filter_type : str
        Filter type ('lowpass', 'highpass', 'bandpass', or 'bandstop')
        
    Returns:
    --------
    ndarray
        Filtered data
    """
    nyquist = 0.5 * sampling_rate
    
    if filter_type in ['bandpass', 'bandstop']:
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("Cutoff must be a tuple of two frequencies for bandpass/bandstop")
        normal_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
    else:
        normal_cutoff = cutoff / nyquist
        
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def adaptive_smoothing(data: np.ndarray, sampling_rate: float, 
                      sensitivity: float = 0.00001) -> Tuple[np.ndarray, int]:
    """
    Apply adaptive smoothing to the data.
    
    Parameters:
    -----------
    data : ndarray
        Data vector
    sampling_rate : float
        Sampling rate in Hz
    sensitivity : float
        Sensitivity parameter
        
    Returns:
    --------
    tuple
        (smoothed_data, window_size)
    """
    window_min = 0.01  # seconds
    window_max = 1.0   # seconds
    window_step = 0.01 # seconds
    
    best_error = float('inf')
    best_window = window_min
    best_smoothed = data.copy()
    
    for window in np.arange(window_min, window_max + window_step, window_step):
        window_size = int(round(window * sampling_rate))
        if window_size < 2:
            continue
            
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 6.0
        smoothed = gaussian_filter1d(data, sigma)
        
        error = np.sqrt(np.mean(np.diff(smoothed) ** 2)) / np.std(smoothed)
        
        if error < best_error * (1 + sensitivity):
            best_error = error
            best_window = window
            best_smoothed = smoothed
        else:
            break
            
    return best_smoothed, int(round(best_window * sampling_rate))
