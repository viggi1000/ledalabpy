import numpy as np
from typing import Tuple, List, Optional

def detect_artifacts(data: np.ndarray, threshold: float = 5.0, 
                    window_size: int = 10) -> List[Tuple[int, int]]:
    """
    Detect artifacts in EDA data.
    
    Parameters:
    -----------
    data : ndarray
        EDA data
    threshold : float
        Threshold for artifact detection (standard deviations)
    window_size : int
        Window size for local statistics
        
    Returns:
    --------
    list
        List of (start, end) indices of detected artifacts
    """
    deriv = np.diff(data)
    
    from scipy.ndimage import uniform_filter1d
    local_mean = uniform_filter1d(deriv, window_size)
    local_std = np.sqrt(uniform_filter1d(np.square(deriv - local_mean), window_size))
    
    z_scores = np.abs(deriv - local_mean) / (local_std + 1e-10)
    artifact_points = np.where(z_scores > threshold)[0]
    
    artifacts = []
    if len(artifact_points) > 0:
        start = artifact_points[0]
        for i in range(1, len(artifact_points)):
            if artifact_points[i] > artifact_points[i-1] + 1:
                artifacts.append((start, artifact_points[i-1] + 1))
                start = artifact_points[i]
        artifacts.append((start, artifact_points[-1] + 1))
    
    return artifacts

def interpolate_artifacts(data: np.ndarray, artifacts: List[Tuple[int, int]]) -> np.ndarray:
    """
    Interpolate artifacts in EDA data.
    
    Parameters:
    -----------
    data : ndarray
        EDA data
    artifacts : list
        List of (start, end) indices of artifacts
        
    Returns:
    --------
    ndarray
        EDA data with interpolated artifacts
    """
    corrected_data = data.copy()
    
    for start, end in artifacts:
        if start > 0 and end < len(data):
            x = np.array([start - 1, end])
            y = np.array([data[start - 1], data[end]])
            xnew = np.arange(start, end)
            ynew = np.interp(xnew, x, y)
            corrected_data[start:end] = ynew
    
    return corrected_data
