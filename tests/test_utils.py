import numpy as np
import pytest
from ledalabpy.utils.math_utils import smooth, downsamp, get_peaks

def test_smooth():
    """Test smoothing function."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))
    
    y_smooth = smooth(y, 5, 'gauss')
    assert len(y_smooth) == len(y)
    assert np.std(y_smooth) < np.std(y)  # Smoothing should reduce variance
    
    y_smooth = smooth(y, 5, 'mean')
    assert len(y_smooth) == len(y)
    assert np.std(y_smooth) < np.std(y)
    
    y_smooth = smooth(y, 5, 'median')
    assert len(y_smooth) == len(y)
    assert np.std(y_smooth) < np.std(y)
    
    y_smooth = smooth(y, 1, 'gauss')
    assert np.allclose(y_smooth, y)

def test_downsamp():
    """Test downsampling function."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    x_down, y_down = downsamp(x, y, 2, 'mean')
    assert len(x_down) == 50
    assert len(y_down) == 50
    
    x_down, y_down = downsamp(x, y, 2, 'decimate')
    assert len(x_down) == 50
    assert len(y_down) == 50
    
    x_down, y_down = downsamp(x, y, 1, 'mean')
    assert np.allclose(x_down, x)
    assert np.allclose(y_down, y)
    
    x = np.linspace(0, 10, 101)
    y = np.sin(x)
    x_down, y_down = downsamp(x, y, 2, 'mean')
    assert len(x_down) == 51
    assert len(y_down) == 51

def test_get_peaks():
    """Test peak detection function."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    minima, maxima = get_peaks(y)
    
    assert len(minima) > 0
    assert len(maxima) > 0
    
    for idx in maxima:
        if idx > 0 and idx < len(y) - 1:
            assert y[idx] > y[idx-1]
            assert y[idx] > y[idx+1]
    
    for idx in minima:
        if idx > 0 and idx < len(y) - 1:
            assert y[idx] < y[idx-1]
            assert y[idx] < y[idx+1]
