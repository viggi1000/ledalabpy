import numpy as np
import pytest
from ledalabpy.analyze.dda import discrete_decomposition_analysis
from ledalabpy.data_models import EDAData, EDASetting

def test_dda_synthetic_data():
    """Test DDA with synthetic data."""
    sampling_rate = 100  # Hz
    duration = 10  # seconds
    time_data = np.linspace(0, duration, int(duration * sampling_rate))
    
    tonic = 2 + 0.1 * time_data
    
    from ledalabpy.analyze.template import bateman_gauss
    
    phasic = np.zeros_like(time_data)
    phasic += bateman_gauss(time_data, 2, 0.5, 1, 3, 0.1)
    phasic += bateman_gauss(time_data, 6, 0.7, 1, 3, 0.1)
    
    data = tonic + phasic
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, len(data))
    data += noise
    
    eda_data = EDAData(
        time_data=time_data,
        conductance_data=data,
        sampling_rate=sampling_rate
    )
    
    settings = EDASetting()
    analysis = discrete_decomposition_analysis(eda_data, settings, optimize=0)
    
    assert analysis.phasic_data is not None
    assert analysis.tonic_data is not None
    assert analysis.driver is not None
    
    assert np.corrcoef(analysis.tonic_data, tonic)[0, 1] > 0.5
    assert np.corrcoef(analysis.phasic_data, phasic)[0, 1] > 0.5
    
    assert analysis.amp is not None
    assert len(analysis.amp) >= 1  # Should detect at least one SCR
    assert analysis.area is not None
    assert len(analysis.area) >= 1
