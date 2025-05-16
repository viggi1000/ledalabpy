import numpy as np
from scipy.stats import norm

def bateman(time, onset, amp, tau1, tau2):
    """
    Bateman function (bi-exponential function).
    
    Parameters:
    -----------
    time : ndarray
        Time vector
    onset : float
        Onset time
    amp : float
        Amplitude
    tau1 : float
        Rise time constant
    tau2 : float
        Decay time constant
        
    Returns:
    --------
    ndarray
        Bateman function values
    """
    component = np.zeros_like(time, dtype=float)
    idx = time > onset
    if np.any(idx):
        if tau1 == tau2:
            tau1 = tau1 * 0.99999
            
        t = time[idx] - onset
        component[idx] = np.exp(-t / tau1) - np.exp(-t / tau2)
        component[idx] = component[idx] / (np.exp(-tau1/tau2 * np.log(tau1/tau2) / (tau1/tau2 - 1)))
        component[idx] = component[idx] * amp
    
    return component

def bateman_gauss(time, onset, amp, tau1, tau2, sigma):
    """
    Bateman function convolved with a Gaussian kernel.
    
    Parameters:
    -----------
    time : ndarray
        Time vector
    onset : float
        Onset time
    amp : float
        Amplitude
    tau1 : float
        Rise time constant
    tau2 : float
        Decay time constant
    sigma : float
        Gaussian kernel standard deviation
        
    Returns:
    --------
    ndarray
        Bateman-Gauss function values
    """
    component = bateman(time, onset, amp, tau1, tau2)
    
    if sigma > 0:
        sr = round(1 / np.mean(np.diff(time)))
        winwidth2 = int(np.ceil(sr * sigma * 4))  # Round half winwidth: 4 SD to each side
        t = np.arange(1, winwidth2 * 2 + 2)  # Odd number (2*winwidth-half+1)
        g = norm.pdf(t, winwidth2 + 1, sigma * sr)
        g = g / np.max(g) * amp
        
        padded = np.concatenate([
            np.ones(winwidth2) * component[0], 
            component, 
            np.ones(winwidth2) * component[-1]
        ])
        
        bg = np.convolve(padded, g, mode='full')
        component = bg[winwidth2 * 2:-(winwidth2 * 2)]
    
    return component
