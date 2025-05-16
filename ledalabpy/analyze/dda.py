import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

from ..data_models import EDAData, EDAAnalysis, EDASetting
from ..utils.math_utils import smooth, within_limits
from .template import bateman_gauss
from .decomposition import segment_driver

def interimpulsefit(driver: np.ndarray, time_data: np.ndarray, 
                  min_idx: np.ndarray, max_idx: np.ndarray) -> float:
    """
    Fit inter-impulse segments to estimate tonic level.
    
    Parameters:
    -----------
    driver : ndarray
        Driver signal
    time_data : ndarray
        Time vector
    min_idx : ndarray
        Indices of impulse minima
    max_idx : ndarray
        Indices of impulse maxima
        
    Returns:
    --------
    float
        Estimated tonic level
    """
    from scipy import interpolate
    
    grid_x = []
    grid_y = []
    
    grid_x.append(0)
    grid_y.append(driver[0])
    
    for i in range(len(min_idx)):
        grid_x.append(min_idx[i])
        grid_y.append(driver[min_idx[i]])
    
    grid_x.append(len(driver) - 1)
    grid_y.append(driver[-1])
    
    f = interpolate.PchipInterpolator(grid_x, grid_y)
    
    return np.min(f(np.arange(len(driver))))

def longdiv(data: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Long division deconvolution.
    
    Parameters:
    -----------
    data : ndarray
        Data vector
    kernel : ndarray
        Kernel vector
        
    Returns:
    --------
    tuple
        (quotient, remainder)
    """
    n = len(data)
    m = len(kernel)
    q = np.zeros(n)
    r = data.copy()
    
    for i in range(n - m + 1):
        if kernel[0] != 0:
            q[i] = r[i] / kernel[0]
            r[i:i+m] -= q[i] * kernel
        else:
            q[i] = 0
    
    return q, r

def deconv_analysis(data: np.ndarray, time_data: np.ndarray, sampling_rate: float, 
                  x: np.ndarray, settings: EDASetting) -> Tuple[float, np.ndarray]:
    """
    Discrete deconvolution analysis.
    
    Parameters:
    -----------
    data : ndarray
        EDA data
    time_data : ndarray
        Time vector
    sampling_rate : float
        Sampling rate in Hz
    x : ndarray
        Parameters [tau1, tau2, dist0]
    settings : EDASetting
        Analysis settings
        
    Returns:
    --------
    tuple
        (error, parameters)
    """
    x[0] = within_limits(x[0], settings.tau_min, 2)
    x[1] = within_limits(x[1], settings.tau_min, settings.tau_max)
    x[2] = within_limits(x[2], settings.dist0_min, 10.0)
    
    if x[1] < x[0]:
        x[0], x[1] = x[1], x[0]
    
    if abs(x[0] - x[1]) < settings.tau_min_diff:
        x[1] = x[1] + settings.tau_min_diff
    
    tau = x[:2]
    dist0 = x[2]
    
    dt = 1 / sampling_rate
    smoothwin = settings.smoothwin_nndeco * 8  # Gauss 8 SD
    winwidth_max = 3  # sec
    swin = round(min(smoothwin, winwidth_max) * sampling_rate)
    
    tb = time_data - time_data[0] + dt
    
    bg = bateman_gauss(tb, 5, 1, tau[0], tau[1], 0.4)
    idx = np.argmax(bg)
    
    eps = np.finfo(float).eps
    fade_in = bg[:idx+10] / (bg[idx+10] + eps) * data[0]
    fade_in = fade_in[fade_in != 0]
    n_fi = len(fade_in)
    d_ext = np.concatenate([fade_in, data])
    
    t_ext = np.arange(time_data[0] - dt, time_data[0] - n_fi * dt, -dt)[::-1]
    t_ext = np.concatenate([t_ext, time_data])
    tb = t_ext - t_ext[0] + dt
    
    kernel = bateman_gauss(tb, 0, 0, tau[0], tau[1], 0)
    kernel = kernel / (np.sum(kernel) + eps)  # Normalize to sum = 1
    kernel[kernel == 0] = 10 * np.finfo(float).eps
    
    eps = np.finfo(float).eps
    sigc = max(0.01, 2 * settings.sig_peak / (np.max(bg) + eps))
    
    from .decomposition import deconvolve
    qt, _ = deconvolve(np.concatenate([d_ext, np.ones(len(kernel) - 1) * d_ext[-1]]), kernel)
    qts = smooth(qt, swin, 'gauss')
    
    segm_width = round(sampling_rate * (settings.segm_width if settings.segm_width is not None else 12))
    onset_idx, impulse, overshoot, imp_min, imp_max = segment_driver(
        qts, np.zeros_like(qts), sigc * 20, segm_width
    )
    
    targetdata_min = interimpulsefit(qts, t_ext, imp_min, imp_max)
    
    if dist0 == 0 or settings.d0_autoupdate:
        dist0 = targetdata_min
        x[2] = targetdata_min
    
    d = data + dist0
    eps = np.finfo(float).eps
    fade_in = bg[:idx+10] / (bg[idx+10] + eps) * d[0]
    fade_in = fade_in[fade_in != 0]
    n_fi = len(fade_in)
    d_ext = np.concatenate([fade_in, d])
    
    q, r = longdiv(d_ext, kernel)
    r = r[:n_fi + len(data)]
    driver = smooth(q, swin, 'gauss')
    remd = smooth(r, swin, 'gauss')
    
    q0, _ = deconvolve(np.concatenate([d_ext, np.ones(len(kernel) - 1) * d_ext[-1]]), kernel)
    q0s = smooth(q0, swin, 'gauss')
    
    segm_width = round(sampling_rate * (settings.segm_width if settings.segm_width is not None else 12))
    onset_idx, impulse, overshoot, imp_min, imp_max = segment_driver(
        driver, remd, sigc, segm_width
    )
    
    n_ext = len(t_ext)
    n_offs = n_ext - len(data)
    phasic_component = []
    phasic_remainder = [np.zeros(len(data))]
    amp = []
    area = []
    overshoot_amp = []
    peaktime_idx = []
    
    for i in range(len(onset_idx)):
        ons = onset_idx[i]
        imp = impulse[i]
        ovs = overshoot[i]
        pco = np.convolve(imp, kernel)
        
        imp_resp = np.zeros(n_ext)
        imp_resp[ons:ons+len(ovs)] = ovs
        imp_resp[ons:] = imp_resp[ons:] + pco[:len(t_ext) - ons]
        imp_resp = imp_resp[n_offs:]
        
        phasic_component.append(imp_resp)
        phasic_remainder.append(phasic_remainder[-1] + imp_resp)
        
        amp.append(np.max(imp_resp))
        peaktime_idx.append(np.argmax(imp_resp))
        area.append((np.sum(imp) + np.sum(ovs)) / sampling_rate)
        overshoot_amp.append(np.max(ovs))
    
    phasic_data = phasic_remainder[-1]
    
    onset_idx = onset_idx.astype(int)
    onset = t_ext[onset_idx]
    tidx = np.where(onset >= 0)[0]
    n_offset = len(t_ext) - len(time_data)
    driver = driver[n_offset:]
    remainder = remd[n_offset:]
    driver_raw = q[n_offset:]
    driver_rawdata = qts[n_offset:]
    driver_sdeconv = q0s[n_offset:]
    
    err_mse = np.mean((data - phasic_data) ** 2)
    err_rmse = np.sqrt(err_mse)
    err_chi2 = err_rmse / data.std() * 10
    
    err1d = np.sqrt(np.mean(np.diff(driver) ** 2))
    err2d = np.sqrt(np.mean(np.diff(remainder) ** 2))
    
    err = (err1d + err2d) * err_chi2 * (10 + len(onset_idx) / 4) / time_data[-1]
    
    return err, x

def discrete_decomposition_analysis(data: EDAData, settings: Optional[EDASetting] = None, 
                                 optimize: int = 2) -> EDAAnalysis:
    """
    Perform Discrete Decomposition Analysis (DDA) on EDA data.
    
    Parameters:
    -----------
    data : EDAData
        EDA data object
    settings : EDASetting
        Analysis settings
    optimize : int
        Optimization level (0=none, 1=quick, 2=full)
        
    Returns:
    --------
    EDAAnalysis
        Analysis results
    """
    if settings is None:
        settings = EDASetting()
        
    target_data = data.conductance_data.copy()
    target_time = data.time_data.copy()
    target_sr = data.sampling_rate
    
    fs_min = 4  # Hz
    n_max = 3000  # samples
    
    if len(data.conductance_data) > n_max:
        from ..utils.math_utils import downsamp, divisors
        
        fs = round(data.sampling_rate)
        n = len(data.conductance_data)
        
        factor_list = divisors(fs)
        fs_list = fs / np.array(factor_list)
        idx = np.flatnonzero(fs_list >= fs_min)
        
        if len(idx) > 0:
            factor_list = np.array(factor_list)[idx]
            fs_list = fs_list[idx]
            
            n_new = n / factor_list
            idx = np.flatnonzero(n_new < n_max)
            
            if len(idx) > 0:
                idx = idx[0]
            else:
                idx = len(factor_list) - 1  # Use largest factor
                
            fac = factor_list[idx]
            target_time, target_data = downsamp(data.time_data, data.conductance_data, int(fac))
            target_sr = fs_list[idx]
    
    x = np.array([settings.tau0_nndeco[0], settings.tau0_nndeco[1], 0.1])  # Initial tau1, tau2, dist0
    err, x = deconv_analysis(target_data, target_time, target_sr, x, settings)
    
    if optimize > 0:
        from .cda import deconv_optimize
        opt_x = deconv_optimize(x, target_data, target_time, target_sr, settings, "dda", optimize)
        if opt_x is not None:
            x = opt_x
            err, x = deconv_analysis(target_data, target_time, target_sr, x, settings)
    
    err, x = deconv_analysis(data.conductance_data, data.time_data, data.sampling_rate, x, settings)
    
    tau = x[:2]
    dist0 = min(x[2], 10.0)  # Cap at 10.0 which is a reasonable maximum
    
    dt = 1 / data.sampling_rate
    tb = data.time_data - data.time_data[0] + dt
    kernel = bateman_gauss(tb, 0, 0, tau[0], tau[1], 0)
    eps = np.finfo(float).eps
    kernel = kernel / (np.sum(kernel) + eps)  # Add epsilon to prevent division by zero
    
    d = data.conductance_data + dist0
    
    smoothwin = settings.smoothwin_nndeco * 8
    swin = round(min(smoothwin, 3) * data.sampling_rate)
    
    bg = bateman_gauss(tb, 5, 1, tau[0], tau[1], 0.4)
    idx = np.argmax(bg)
    eps = np.finfo(float).eps
    fade_in = bg[:idx+10] / (bg[idx+10] + eps) * d[0]
    fade_in = fade_in[fade_in != 0]
    n_fi = len(fade_in)
    d_ext = np.concatenate([fade_in, d])
    
    t_ext = np.arange(data.time_data[0] - dt, data.time_data[0] - n_fi * dt, -dt)[::-1]
    t_ext = np.concatenate([t_ext, data.time_data])
    
    q, r = longdiv(d_ext, kernel)
    driver = smooth(q, swin, 'gauss')
    remainder = smooth(r[:n_fi + len(data.conductance_data)], swin, 'gauss')
    
    eps = np.finfo(float).eps
    sigc = max(0.001, settings.sig_peak / (np.max(bg) + eps))
    segm_width = round(data.sampling_rate * (settings.segm_width if settings.segm_width is not None else 12))
    onset_idx, impulse, overshoot, imp_min, imp_max = segment_driver(
        driver[n_fi:], remainder[n_fi:], sigc, segm_width
    )
    
    phasic_component = []
    phasic_remainder = [np.zeros(len(data.conductance_data))]
    amp = []
    area = []
    overshoot_amp = []
    peaktime_idx = []
    
    for i in range(len(onset_idx)):
        ons = onset_idx[i]
        imp = impulse[i]
        ovs = overshoot[i]
        pco = np.convolve(imp, kernel)
        
        imp_resp = np.zeros(len(data.conductance_data))
        imp_resp[ons:ons+len(ovs)] = ovs[ons:ons+len(ovs)]
        imp_resp[ons:] = imp_resp[ons:] + pco[:len(data.conductance_data) - ons]
        
        phasic_component.append(imp_resp)
        phasic_remainder.append(phasic_remainder[-1] + imp_resp)
        
        amp.append(np.max(imp_resp))
        peaktime_idx.append(np.argmax(imp_resp))
        area.append((np.sum(imp) + np.sum(ovs)) / data.sampling_rate)
        overshoot_amp.append(np.max(ovs))
    
    phasic_data = phasic_remainder[-1]
    tonic_data = data.conductance_data - phasic_data
    
    analysis = EDAAnalysis(
        method="dda",
        tau=tau,
        dist0=dist0,
        driver=driver[n_fi:],
        remainder=remainder[n_fi:],
        kernel=kernel,
        phasic_data=phasic_data,
        tonic_data=tonic_data,
        phasic_component=phasic_component,
        phasic_remainder=phasic_remainder[1:],  # Skip the first zero array
        driver_raw=q[n_fi:],
        error={
            'MSE': np.mean((data.conductance_data - phasic_data - tonic_data) ** 2),
            'RMSE': np.sqrt(np.mean((data.conductance_data - phasic_data - tonic_data) ** 2)),
            'compound': err
        }
    )
    
    if len(onset_idx) > 0:
        analysis.impulse_onset = data.time_data[imp_min]
        analysis.impulse_peak_time = data.time_data[imp_max]
        analysis.onset = data.time_data[onset_idx]
        analysis.peak_time = data.time_data[onset_idx + peaktime_idx]
        analysis.amp = np.array(amp)
        analysis.area = np.array(area)
        analysis.overshoot_amp = np.array(overshoot_amp)
    
    return analysis
