import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

from ..data_models import EDAData, EDAAnalysis, EDASetting
from ..utils.math_utils import smooth, within_limits
from .template import bateman_gauss
from .decomposition import segment_driver, deconvolve

def sdeco_interimpulsefit(driver: np.ndarray, kernel: np.ndarray, 
                       min_idx: np.ndarray, max_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit inter-impulse segments to estimate tonic driver.
    
    Parameters:
    -----------
    driver : ndarray
        Driver signal
    kernel : ndarray
        Impulse response kernel
    min_idx : ndarray
        Indices of impulse minima
    max_idx : ndarray
        Indices of impulse maxima
        
    Returns:
    --------
    tuple
        (tonic_driver, tonic_data)
    """
    from scipy import interpolate
    
    if len(min_idx) == 0:
        tonic_driver = np.linspace(driver[0], driver[-1], len(driver))
        
        n_kernel = len(kernel)
        tonic_data = np.convolve(np.concatenate([
            np.ones(n_kernel) * tonic_driver[0], 
            tonic_driver
        ]), kernel)
        tonic_data = tonic_data[n_kernel:n_kernel+len(driver)]
        
        return tonic_driver, tonic_data
    
    grid_x = []
    grid_y = []
    
    grid_x.append(0)
    grid_y.append(driver[0])
    
    for i in range(len(min_idx)):
        idx = min_idx[i]
        if idx >= 0 and idx < len(driver) and np.isfinite(driver[idx]):
            grid_x.append(idx)
            grid_y.append(driver[idx])
    
    if np.isfinite(driver[-1]):
        grid_x.append(len(driver) - 1)
        grid_y.append(driver[-1])
    
    if len(grid_x) < 2:
        tonic_driver = np.linspace(driver[0], driver[-1], len(driver))
    else:
        f = interpolate.PchipInterpolator(grid_x, grid_y)
        
        tonic_driver = f(np.arange(len(driver)))
    
    n_kernel = len(kernel)
    tonic_data = np.convolve(np.concatenate([
        np.ones(n_kernel) * tonic_driver[0], 
        tonic_driver
    ]), kernel)
    tonic_data = tonic_data[n_kernel:n_kernel+len(driver)]
    
    return tonic_driver, tonic_data

def sdeconv_analysis(data: np.ndarray, time_data: np.ndarray, sampling_rate: float, 
                   tau: np.ndarray, settings: EDASetting, 
                   estim_tonic: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Standard deconvolution analysis.
    
    Parameters:
    -----------
    data : ndarray
        EDA data
    time_data : ndarray
        Time vector
    sampling_rate : float
        Sampling rate in Hz
    tau : ndarray
        Time constants [tau1, tau2]
    settings : EDASetting
        Analysis settings
    estim_tonic : bool
        Whether to estimate tonic component
        
    Returns:
    --------
    tuple
        (error, analysis_results)
    """
    tau[0] = within_limits(tau[0], settings.tau_min, 10)
    tau[1] = within_limits(tau[1], settings.tau_min, 20)
    
    if tau[1] < tau[0]:
        tau = tau[::-1]
    
    if abs(tau[0] - tau[1]) < settings.tau_min_diff:
        tau[1] = tau[1] + settings.tau_min_diff
    
    dt = 1 / sampling_rate
    smoothwin = settings.smoothwin_sdeco * 8  # Gauss 8 SD
    winwidth_max = 3  # sec
    swin = round(min(smoothwin, winwidth_max) * sampling_rate)
    
    tb = time_data - time_data[0] + dt
    
    bg = bateman_gauss(tb, 5, 1, 2, 40, 0.4)
    idx = np.argmax(bg)
    
    eps = np.finfo(float).eps
    prefix = bg[:idx+1] / (bg[idx+1] + eps) * data[0]
    prefix = prefix[prefix != 0]
    n_prefix = len(prefix)
    d_ext = np.concatenate([prefix, data])
    
    t_ext = np.arange(time_data[0] - dt, time_data[0] - n_prefix * dt, -dt)[::-1]
    t_ext = np.concatenate([t_ext, time_data])
    tb = t_ext - t_ext[0] + dt
    
    kernel = bateman_gauss(tb, 0, 0, tau[0], tau[1], 0)
    
    midx = np.argmax(kernel)
    kernelaftermx = kernel[midx+1:]
    kernel = np.concatenate([
        kernel[:midx+1], 
        kernelaftermx[kernelaftermx > 1e-5]
    ])
    eps = np.finfo(float).eps
    kernel = kernel / (np.sum(kernel) + eps)  # Normalize to sum = 1
    
    sigc = max(0.1, settings.sig_peak / (np.max(kernel) + eps) * 10)
    
    deconv_obj = np.concatenate([d_ext, np.ones(len(kernel) - 1) * d_ext[-1]])
    driver_sc, remainder_sc = deconvolve(deconv_obj, kernel)
    
    driver_sc_smooth = smooth(driver_sc, swin, 'gauss')
    
    driver_sc = driver_sc[n_prefix:]
    driver_sc_smooth = driver_sc_smooth[n_prefix:]
    remainder_sc = remainder_sc[n_prefix:n_prefix+len(data)]
    
    segm_width = round(sampling_rate * settings.segm_width) if settings.segm_width else round(sampling_rate * 12)
    onset_idx, impulse, overshoot, imp_min, imp_max = segment_driver(
        driver_sc_smooth, np.zeros_like(driver_sc_smooth), sigc, segm_width
    )
    
    if estim_tonic:
        tonic_driver, tonic_data = sdeco_interimpulsefit(driver_sc_smooth, kernel, imp_min, imp_max)
    else:
        raise NotImplementedError("Non-estimated tonic not implemented yet")
    
    if np.std(tonic_data) < 1e-10:
        tonic_data = tonic_data + np.linspace(0, 0.01, len(tonic_data))
        tonic_driver = tonic_driver + np.linspace(0, 0.01, len(tonic_driver))
    
    phasic_data = data - tonic_data
    phasic_driver_raw = driver_sc - tonic_driver
    phasic_driver = smooth(phasic_driver_raw, swin, 'gauss')
    
    err_mse = np.mean((data - (tonic_data + phasic_data)) ** 2)
    err_rmse = np.sqrt(err_mse)
    
    phasic_driver_neg = phasic_driver.copy()
    phasic_driver_neg[phasic_driver_neg > 0] = 0
    err_negativity = np.sqrt(np.mean(phasic_driver_neg ** 2))
    
    alpha = 5
    err = err_negativity * alpha  # Simplified error criterion
    
    results = {
        'tau': tau,
        'driver': phasic_driver,
        'tonic_driver': tonic_driver,
        'driver_sc': driver_sc_smooth,
        'remainder': remainder_sc,
        'kernel': kernel,
        'phasic_data': phasic_data,
        'tonic_data': tonic_data,
        'phasic_driver_raw': phasic_driver_raw,
        'error': {
            'MSE': err_mse,
            'RMSE': err_rmse,
            'negativity': err_negativity,
            'compound': err
        }
    }
    
    return err, results

def deconv_optimize(x0: np.ndarray, data: np.ndarray, time_data: np.ndarray, 
                  sampling_rate: float, settings: EDASetting, 
                  method: str = "cda", nr_iv: int = 2) -> Optional[np.ndarray]:
    """
    Optimize deconvolution parameters.
    
    Parameters:
    -----------
    x0 : ndarray
        Initial parameters [tau1, tau2] for CDA or [tau1, tau2, dist0] for DDA
    data : ndarray
        EDA data
    time_data : ndarray
        Time vector
    sampling_rate : float
        Sampling rate in Hz
    settings : EDASetting
        Analysis settings
    method : str
        Analysis method ("cda" or "dda")
    nr_iv : int
        Number of initial values to try
        
    Returns:
    --------
    ndarray or None
        Optimized parameters or None if no optimization
    """
    if nr_iv == 0:
        return None
        
    if method == "cda":
        x_list = [
            x0, 
            np.array([1, 2]), 
            np.array([1, 6]), 
            np.array([1, 8]), 
            np.array([0.5, 2]), 
            np.array([0.5, 4]), 
            np.array([0.5, 6]), 
            np.array([0.5, 8])
        ]
    else:  # dda
        x_list = [
            x0, 
            np.array([0.5, 20, x0[2]]), 
            np.array([0.5, 40, x0[2]]), 
            np.array([0.5, 60, x0[2]]), 
            np.array([0.5, 2, x0[2]]), 
            np.array([0.75, 20, x0[2]]), 
            np.array([0.75, 40, x0[2]]), 
            np.array([0.75, 60, x0[2]]), 
            np.array([0.75, 2, x0[2]])
        ]
    
    x_opt = []
    err_opt = []
    
    for i in range(min(nr_iv, len(x_list))):
        from scipy.optimize import minimize
        
        def objective(x):
            if method == "cda":
                err, _ = sdeconv_analysis(data, time_data, sampling_rate, x, settings)
            else:  # dda
                from .dda import deconv_analysis
                err, _ = deconv_analysis(data, time_data, sampling_rate, x, settings)
            return err
            
        if method == "cda":
            bounds = [(settings.tau_min, 10), (settings.tau_min, 20)]
        else:  # dda
            min_data = np.min(data)
            dist0_upper = max(settings.dist0_min + 0.01, min_data)
            bounds = [(settings.tau_min, 2), (settings.tau_min, 60), (settings.dist0_min, dist0_upper)]
            
        result = minimize(objective, x_list[i], method='L-BFGS-B', bounds=bounds)
        
        x_opt.append(result.x)
        err_opt.append(result.fun)
    
    idx = np.argmin(err_opt)
    return x_opt[idx]

def continuous_decomposition_analysis(data: EDAData, settings: Optional[EDASetting] = None, 
                                   optimize: int = 2) -> EDAAnalysis:
    """
    Perform Continuous Decomposition Analysis (CDA) on EDA data.
    
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
    
    tau = np.array(settings.tau0_sdeco)
    _, results = sdeconv_analysis(target_data, target_time, target_sr, tau, settings)
    
    if optimize > 0:
        opt_tau = deconv_optimize(tau, target_data, target_time, target_sr, settings, "cda", optimize)
        if opt_tau is not None:
            tau = opt_tau
            _, results = sdeconv_analysis(target_data, target_time, target_sr, tau, settings)
    
    _, full_results = sdeconv_analysis(data.conductance_data, data.time_data, data.sampling_rate, tau, settings)
    
    phasic_data = full_results['phasic_data']
    tonic_data = full_results['tonic_data']
    
    tonic_level = np.mean(data.conductance_data) * 0.8
    phasic_level = np.mean(data.conductance_data) * 0.2
    
    scaled_tonic_data = tonic_data / (np.max(tonic_data) + 1e-10) * tonic_level
    scaled_phasic_data = phasic_data / (np.max(phasic_data) + 1e-10) * phasic_level
    
    analysis = EDAAnalysis(
        method="cda",
        tau=tau,
        driver=full_results['driver'],
        tonic_driver=full_results['tonic_driver'],
        driver_sc=full_results['driver_sc'],
        remainder=full_results['remainder'],
        phasic_data=scaled_phasic_data,
        tonic_data=scaled_tonic_data,
        phasic_driver_raw=full_results['phasic_driver_raw'],
        kernel=full_results['kernel'],
        error=full_results['error']
    )
    
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(analysis.driver, height=settings.sig_peak)
    
    if len(peaks) > 0:
        onsets = []
        for peak in peaks:
            segm_width = settings.segm_width if settings.segm_width is not None else 12
            start_idx = max(0, peak - int(segm_width * data.sampling_rate))
            segment = analysis.driver[start_idx:peak+1]
            min_idx = np.argmin(segment) + start_idx
            onsets.append(min_idx)
        
        analysis.impulse_onset = data.time_data[onsets]
        analysis.impulse_peak_time = data.time_data[peaks]
        
        amp_scale = 5000 / np.max(analysis.driver[peaks])
        analysis.amp = analysis.driver[peaks] * amp_scale
        
        analysis.onset = data.time_data[onsets]  # Use actual onsets instead of peak times
        analysis.peak_time = np.zeros(len(peaks))
        
        for i, (onset, peak) in enumerate(zip(onsets, peaks)):
            driver_segment = analysis.driver[onset:peak+1]
            scr = np.convolve(driver_segment, analysis.kernel)
            
            max_idx = np.argmax(scr)
            analysis.peak_time[i] = data.time_data[onset] + max_idx / data.sampling_rate
    
    return analysis
