import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, Literal

from .data_models import EDAData, EDAAnalysis, EDASetting
from .analyze.cda import continuous_decomposition_analysis
from .analyze.dda import discrete_decomposition_analysis
from .utils.math_utils import smooth, downsamp

def import_data(raw_data: np.ndarray, sampling_rate: float, 
               downsample: int = 1) -> EDAData:
    """
    Import and prepare EDA data for analysis.
    
    Parameters:
    -----------
    raw_data : ndarray
        Raw EDA data
    sampling_rate : float
        Sampling rate in Hz
    downsample : int
        Downsampling factor
        
    Returns:
    --------
    EDAData
        Prepared EDA data
    """
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data, dtype='float64')
        
    time_data = np.arange(len(raw_data)) / sampling_rate
    
    if downsample > 1:
        time_data, raw_data = downsamp(time_data, raw_data, downsample)
        sampling_rate = sampling_rate / downsample
    
    data = EDAData(
        time_data=time_data,
        conductance_data=raw_data,
        sampling_rate=sampling_rate
    )
    
    data.conductance_error = np.sqrt(np.mean(np.diff(raw_data) ** 2) / 2)
    
    from .preprocessing.filter import adaptive_smoothing
    data.conductance_smooth_data, _ = adaptive_smoothing(raw_data, sampling_rate)
    
    return data

def analyze(data: Union[np.ndarray, EDAData], method: Literal["cda", "dda"] = "cda", 
           sampling_rate: Optional[float] = None,
           optimize: int = 2, settings: Optional[EDASetting] = None) -> EDAAnalysis:
    """
    Analyze EDA data using the specified method.
    
    Parameters:
    -----------
    data : ndarray or EDAData
        EDA data
    method : str
        Analysis method ('cda' for Continuous Decomposition Analysis, 'dda' for Discrete Decomposition Analysis)
    sampling_rate : float
        Sampling rate in Hz (required if data is ndarray)
    optimize : int
        Optimization level (0=none, 1=quick, 2=full)
    settings : EDASetting
        Analysis settings
        
    Returns:
    --------
    EDAAnalysis
        Analysis results
    """
    if isinstance(data, np.ndarray):
        if sampling_rate is None:
            raise ValueError("Sampling rate must be provided when data is a numpy array")
        data = import_data(data, sampling_rate)
    
    if settings is None:
        settings = EDASetting()
    
    if method.lower() == "cda":
        analysis = continuous_decomposition_analysis(data, settings, optimize)
    elif method.lower() == "dda":
        analysis = discrete_decomposition_analysis(data, settings, optimize)
    else:
        raise ValueError(f"Unknown analysis method: {method}. Use 'cda' or 'dda'.")
    
    return analysis

def extract_features(analysis: EDAAnalysis) -> Dict[str, Any]:
    """
    Extract features from analysis results.
    
    Parameters:
    -----------
    analysis : EDAAnalysis
        Analysis results
        
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    if analysis.tonic_data is not None:
        features['tonic_mean'] = np.mean(analysis.tonic_data)
        features['tonic_std'] = np.std(analysis.tonic_data)
    
    if analysis.phasic_data is not None:
        features['phasic_mean'] = np.mean(analysis.phasic_data)
        features['phasic_std'] = np.std(analysis.phasic_data)
    
    if analysis.amp is not None and len(analysis.amp) > 0:
        features['scr_count'] = len(analysis.amp)
        features['scr_amplitude_mean'] = np.mean(analysis.amp)
        features['scr_amplitude_std'] = np.std(analysis.amp)
        
        if analysis.onset is not None and analysis.peak_time is not None:
            rise_times = analysis.peak_time - analysis.onset
            features['scr_rise_time_mean'] = np.mean(rise_times)
            features['scr_rise_time_std'] = np.std(rise_times)
    
    if analysis.method == "dda" and analysis.area is not None and len(analysis.area) > 0:
        features['scr_area_mean'] = np.mean(analysis.area)
        features['scr_area_std'] = np.std(analysis.area)
    
    return features

def get_result(raw_data: np.ndarray, result_type: str, sampling_rate: float, 
              method: Literal["cda", "dda"] = "cda", downsample: int = 1, optimize: int = 2) -> np.ndarray:
    """
    Run analysis and return the specified result.
    
    Parameters:
    -----------
    raw_data : ndarray
        Raw EDA data
    result_type : str
        Type of result to return ('phasicdata', 'tonicdata', 'phasicdriver', etc.)
    method : str
        Analysis method ('cda' or 'dda')
    sampling_rate : float
        Sampling rate in Hz
    downsample : int
        Downsampling factor
    optimize : int
        Optimization level (0=none, 1=quick, 2=full)
        
    Returns:
    --------
    ndarray
        Requested result
    """
    data = import_data(raw_data, sampling_rate, downsample)
    
    analysis = analyze(data, method, optimize=optimize)
    
    result_type = result_type.lower()
    
    if result_type == "phasicdata":
        return analysis.phasic_data
    elif result_type == "tonicdata":
        return analysis.tonic_data
    elif result_type == "phasicdriver":
        return analysis.driver
    elif result_type == "tonicdriver":
        return analysis.tonic_driver
    else:
        raise ValueError(f"Unknown result type: {result_type}")
