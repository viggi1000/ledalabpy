import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union, Literal

@dataclass
class EDASetting:
    """Settings for EDA analysis."""
    tau_min: float = 0.001
    tau_max: float = 100.0
    tau_min_diff: float = 0.01
    dist0_min: float = 0.001
    sig_peak: float = 0.001
    segm_width: Optional[float] = None
    
    peak_detection_threshold: Optional[float] = None  # Default: 0.00000001 (set in implementation)
    
    amplitude_scaling: Optional[float] = None  # General amplitude scaling (overrides method-specific)
    amplitude_scaling_cda: Optional[float] = None  # Default: 400.0 (set in implementation)
    amplitude_scaling_dda: Optional[float] = None  # Default: 10000.0 (set in implementation)
    
    tau0_sdeco: List[float] = field(default_factory=lambda: [1.0, 3.75])
    smoothwin_sdeco: float = 0.2
    tonic_grid_size_sdeco: float = 10.0
    
    tau0_nndeco: List[float] = field(default_factory=lambda: [0.75, 20.0, 0.02])
    d0_autoupdate: bool = True
    tonic_is_const: bool = False
    tonic_slow_increase: bool = False
    smoothwin_nndeco: float = 0.2
    tonic_grid_size_nndeco: float = 10.0

@dataclass
class EDAData:
    """Container for EDA time series data."""
    time_data: np.ndarray
    conductance_data: np.ndarray
    sampling_rate: float
    conductance_smooth_data: Optional[np.ndarray] = None
    conductance_error: Optional[float] = None
    conductance_min: Optional[float] = None
    conductance_max: Optional[float] = None
    
    def __post_init__(self):
        self.N = len(self.conductance_data)
        if self.conductance_min is None:
            self.conductance_min = np.min(self.conductance_data)
        if self.conductance_max is None:
            self.conductance_max = np.max(self.conductance_data)

@dataclass
class EDAAnalysis:
    """Results of EDA analysis."""
    method: Literal["cda", "dda"] = "cda"
    tau: Optional[np.ndarray] = None
    dist0: Optional[float] = None  # For DDA
    
    phasic_data: Optional[np.ndarray] = None
    tonic_data: Optional[np.ndarray] = None
    driver: Optional[np.ndarray] = None
    tonic_driver: Optional[np.ndarray] = None
    kernel: Optional[np.ndarray] = None
    remainder: Optional[np.ndarray] = None
    
    driver_sc: Optional[np.ndarray] = None
    phasic_driver_raw: Optional[np.ndarray] = None
    
    phasic_component: Optional[List[np.ndarray]] = None
    phasic_remainder: Optional[List[np.ndarray]] = None
    driver_raw: Optional[np.ndarray] = None
    
    impulse_onset: Optional[np.ndarray] = None
    impulse_peak_time: Optional[np.ndarray] = None
    onset: Optional[np.ndarray] = None
    peak_time: Optional[np.ndarray] = None
    amp: Optional[np.ndarray] = None
    
    area: Optional[np.ndarray] = None
    overshoot_amp: Optional[np.ndarray] = None
    
    error: Dict[str, float] = field(default_factory=dict)

class EDAProcessor:
    """Main class for EDA processing."""
    def __init__(self, settings: Optional[EDASetting] = None):
        self.settings = settings or EDASetting()
        self.data = None
        self.analysis = None
        
    def import_data(self, data: np.ndarray, sampling_rate: float, downsample: int = 1) -> EDAData:
        """Import and prepare EDA data for analysis."""
        pass
        
    def analyze(self, method: Literal["cda", "dda"] = "cda", optimize: int = 2) -> EDAAnalysis:
        """Analyze EDA data using the specified method."""
        pass
