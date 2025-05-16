import numpy as np
import scipy.io
from typing import Tuple, Dict, Any, Optional
from ..data_models import EDAData

def load_mat_file(file_path: str) -> Dict[str, Any]:
    """
    Load EDA data from a MATLAB .mat file.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file
        
    Returns:
    --------
    dict
        Dictionary containing the loaded data
    """
    return scipy.io.loadmat(file_path)
    
def extract_eda_from_mat(mat_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract EDA data from a loaded .mat file.
    
    Parameters:
    -----------
    mat_data : dict
        Dictionary containing the loaded .mat data
        
    Returns:
    --------
    tuple
        (time_data, conductance_data, sampling_rate)
    """
    if 'data' in mat_data and 'conductance' in mat_data['data'][0, 0]:
        conductance_data = np.array(mat_data['data']['conductance'][0, 0][0], dtype='float64')
        
        if 'time' in mat_data['data'][0, 0]:
            time_data = np.array(mat_data['data']['time'][0, 0][0], dtype='float64')
        else:
            sampling_rate = 100.0
            if 'samplingrate' in mat_data['data'][0, 0]:
                sampling_rate = float(mat_data['data']['samplingrate'][0, 0][0, 0])
            time_data = np.arange(len(conductance_data)) / sampling_rate
            
        return time_data, conductance_data, sampling_rate
    else:
        if 'conductance' in mat_data:
            conductance_data = np.array(mat_data['conductance'], dtype='float64')
            
            if 'time' in mat_data:
                time_data = np.array(mat_data['time'], dtype='float64')
            else:
                sampling_rate = 100.0
                if 'samplingrate' in mat_data:
                    sampling_rate = float(mat_data['samplingrate'])
                time_data = np.arange(len(conductance_data)) / sampling_rate
                
            return time_data, conductance_data, sampling_rate
        else:
            raise ValueError("Unexpected .mat file structure")
