import numpy as np
import pytest
from ledalabpy.preprocessing.filter import butterworth_filter, adaptive_smoothing
from ledalabpy.preprocessing.artifact import detect_artifacts, interpolate_artifacts

def test_butterworth_filter():
    """Test Butterworth filter."""
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz signal
    noise = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz noise
    data = signal + noise
    
    filtered_data = butterworth_filter(data, 100, 1.0, 4, 'lowpass')
    
    assert np.std(filtered_data - signal) < np.std(data - signal)
    
    filtered_data = butterworth_filter(data, 100, 5.0, 4, 'highpass')
    
    assert np.std(filtered_data - noise) < np.std(data - noise)
    
    filtered_data = butterworth_filter(data, 100, (0.1, 1.0), 4, 'bandpass')
    
    assert np.std(filtered_data - signal) < np.std(data - signal)

def test_adaptive_smoothing():
    """Test adaptive smoothing."""
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz signal
    noise = 0.5 * np.random.randn(len(t))  # Random noise
    data = signal + noise
    
    smoothed_data, window_size = adaptive_smoothing(data, 100)
    
    assert np.std(smoothed_data - signal) < np.std(data - signal)
    
    assert window_size > 0
    assert window_size < 100  # Should be less than 1 second at 100 Hz

def test_detect_artifacts():
    """Test artifact detection."""
    data = np.zeros(100)
    data[30:35] = 5.0  # Artifact
    data[70:75] = -5.0  # Artifact
    
    artifacts = detect_artifacts(data, threshold=3.0)
    
    assert len(artifacts) == 2
    assert artifacts[0][0] == 29 or artifacts[0][0] == 30
    assert artifacts[1][0] == 69 or artifacts[1][0] == 70

def test_interpolate_artifacts():
    """Test artifact interpolation."""
    data = np.zeros(100)
    data[30:35] = 5.0  # Artifact
    data[70:75] = -5.0  # Artifact
    
    artifacts = [(30, 35), (70, 75)]
    
    corrected_data = interpolate_artifacts(data, artifacts)
    
    assert np.all(corrected_data[30:35] < 1.0)
    assert np.all(corrected_data[70:75] > -1.0)
    
    assert np.allclose(corrected_data[:30], data[:30])
    assert np.allclose(corrected_data[35:70], data[35:70])
    assert np.allclose(corrected_data[75:], data[75:])
