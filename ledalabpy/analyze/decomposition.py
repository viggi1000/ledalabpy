import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

def segment_driver(driver: np.ndarray, remainder: np.ndarray, 
                  threshold: float, segment_width: int) -> Tuple[np.ndarray, List[np.ndarray], 
                                                             List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Segment the driver signal into impulses.
    
    Parameters:
    -----------
    driver : ndarray
        Driver signal
    remainder : ndarray
        Remainder signal
    threshold : float
        Threshold for peak detection
    segment_width : int
        Width of segments in samples
        
    Returns:
    --------
    tuple
        (onset_idx, impulse_list, overshoot_list, impulse_min_idx, impulse_max_idx)
    """
    from ..utils.math_utils import get_peaks
    
    min_idx, max_idx = get_peaks(driver)
    
    if len(max_idx) == 0:
        return np.array([]), [], [], np.array([]), np.array([])
    
    sign_peaks = []
    min_before_after = []
    
    for i in range(len(max_idx)):
        before_idx = min_idx[min_idx < max_idx[i]]
        after_idx = min_idx[min_idx > max_idx[i]]
        
        if len(before_idx) > 0 and len(after_idx) > 0:
            before_val = driver[before_idx[-1]]
            after_val = driver[after_idx[0]]
            peak_val = driver[max_idx[i]]
            
            if max(peak_val - before_val, peak_val - after_val) > threshold:
                sign_peaks.append(max_idx[i])
                min_before_after.append([before_idx[-1], after_idx[0]])
                
    if len(sign_peaks) == 0:
        return np.array([]), [], [], np.array([]), np.array([])
        
    max_idx = np.array(sign_peaks)
    min_idx = np.array(min_before_after)
    
    onset_idx = []
    impulse_list = []
    overshoot_list = []
    
    for i in range(len(max_idx)):
        segm_start = max(0, min(min_idx[i, 0], max_idx[i] - segment_width // 2))
        segm_end = min(len(driver), segm_start + segment_width)
        
        impulse = np.zeros_like(driver)
        impulse_data = driver[segm_start:segm_end].copy()
        impulse_data[segm_start + np.arange(segm_end - segm_start) >= min_idx[i, 1]] = 0
        impulse[segm_start:segm_end] = impulse_data
        
        overshoot = np.zeros_like(driver)
        
        if i < len(max_idx) - 1:
            rem_min, rem_max = get_peaks(remainder)
            rem_idx = np.where((rem_max > max_idx[i]) & (rem_max < max_idx[i + 1]))[0]
        else:
            rem_min, rem_max = get_peaks(remainder)
            rem_idx = np.where(rem_max > max_idx[i])[0]
            
        if len(rem_idx) > 0:
            rem_idx = rem_idx[0]  # Take first remainder peak
            ovs_start = max(rem_min[rem_idx], segm_start)
            ovs_end = min(rem_min[rem_idx + 1] if rem_idx + 1 < len(rem_min) else len(driver), segm_end)
            overshoot[ovs_start:ovs_end] = remainder[ovs_start:ovs_end]
        
        onset_idx.append(segm_start)
        impulse_list.append(impulse)
        overshoot_list.append(overshoot)
    
    return np.array(onset_idx), impulse_list, overshoot_list, min_idx[:, 0], max_idx

def deconvolve(data: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deconvolve data with kernel using FFT.
    
    Parameters:
    -----------
    data : ndarray
        Data to deconvolve
    kernel : ndarray
        Kernel to deconvolve with
        
    Returns:
    --------
    tuple
        (deconvolved_data, remainder)
    """
    kernel = kernel / np.sum(kernel)
    
    n = len(data)
    m = len(kernel)
    padded_data = np.concatenate([data, np.zeros(m - 1)])
    padded_kernel = np.concatenate([kernel, np.zeros(n - 1)])
    
    data_fft = np.fft.fft(padded_data)
    kernel_fft = np.fft.fft(padded_kernel)
    
    eps = np.finfo(float).eps
    result_fft = data_fft / (kernel_fft + eps)
    
    result = np.real(np.fft.ifft(result_fft))
    
    reconvolved = np.convolve(result[:n], kernel, mode='full')[:n]
    remainder = data - reconvolved
    
    return result[:n], remainder
