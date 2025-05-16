import numpy as np
from typing import Tuple, List

def downsamp(time_data: np.ndarray, data: np.ndarray, factor: int, method: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample time series data.
    
    Parameters:
    -----------
    time_data : ndarray
        Time vector
    data : ndarray
        Data vector
    factor : int
        Downsampling factor
    method : str
        Downsampling method ('mean' or 'decimate')
        
    Returns:
    --------
    tuple
        (downsampled_time, downsampled_data)
    """
    if factor <= 1:
        return time_data, data
        
    n = len(data)
    n_down = n // factor
    
    if method == 'mean':
        remainder = n % factor
        if remainder > 0:
            pad_data = np.concatenate([data, np.ones(factor - remainder) * data[-1]])
            pad_time = np.concatenate([time_data, 
                                      np.linspace(time_data[-1], 
                                                 time_data[-1] + (factor - remainder) * (time_data[1] - time_data[0]), 
                                                 factor - remainder)])
            n = len(pad_data)
        else:
            pad_data = data
            pad_time = time_data
            
        data_reshaped = pad_data[:n_down * factor].reshape(-1, factor)
        down_data = np.mean(data_reshaped, axis=1)
        
        down_time = pad_time[::factor][:n_down]
        
    elif method == 'decimate':
        from scipy import signal
        down_data = signal.decimate(data, factor)
        down_time = time_data[::factor][:len(down_data)]
    else:
        raise ValueError(f"Unknown downsample method: {method}")
        
    return down_time, down_data

def smooth(data: np.ndarray, window_size: int, method: str = 'gauss') -> np.ndarray:
    """
    Smooth data using specified method.
    
    Parameters:
    -----------
    data : ndarray
        Data vector
    window_size : int
        Window size (must be odd for 'gauss' and 'median')
    method : str
        Smoothing method ('gauss', 'mean', or 'median')
        
    Returns:
    --------
    ndarray
        Smoothed data
    """
    if window_size <= 1:
        return data.copy()
        
    if method in ['gauss', 'median'] and window_size % 2 == 0:
        window_size += 1
        
    if method == 'gauss':
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 6.0  # Convert window to sigma
        smoothed = gaussian_filter1d(data, sigma)
    elif method == 'mean':
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(data, window_size)
    elif method == 'median':
        from scipy.signal import medfilt
        smoothed = medfilt(data, kernel_size=window_size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
        
    return smoothed

def within_limits(value: float, min_val: float, max_val: float) -> float:
    """Constrain a value within specified limits."""
    return max(min_val, min(max_val, value))
    
def divisors(n: int) -> List[int]:
    """Get all divisors of a number."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)
    
def get_peaks(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find minima and maxima in data.
    
    Parameters:
    -----------
    data : ndarray
        Data vector
        
    Returns:
    --------
    tuple
        (minima_indices, maxima_indices)
    """
    deriv = np.diff(data)
    sign_deriv = np.sign(deriv)
    sign_change = np.diff(sign_deriv)
    
    minima = np.where(sign_change > 0)[0] + 1
    
    maxima = np.where(sign_change < 0)[0] + 1
    
    return minima, maxima
