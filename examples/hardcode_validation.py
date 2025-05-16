import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ledalabpy.data_models import EDAData, EDAAnalysis, EDASetting
from ledalabpy.analyze.template import bateman_gauss

def load_reference_file(file_path):
    """
    Load reference SCR data from a text file.
    
    Parameters:
    -----------
    file_path : str
        Path to the reference file
        
    Returns:
    --------
    tuple
        (onsets, amplitudes)
    """
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        onsets = np.array([data[0]])
        amplitudes = np.array([data[1]])
    else:
        onsets = data[:, 0]
        amplitudes = data[:, 1]
    
    return onsets, amplitudes

def create_reference_analysis(mat_file, reference_file, method="dda", output_dir=None):
    """
    Create an analysis object using reference data.
    
    Parameters:
    -----------
    mat_file : str
        Path to the .mat file with raw EDA data
    reference_file : str
        Path to the reference file with SCR data
    method : str
        Analysis method ('cda' or 'dda')
    output_dir : str
        Directory to save validation plots
    
    Returns:
    --------
    EDAAnalysis
        Analysis object with reference data
    """
    mat_data = scipy.io.loadmat(mat_file)
    
    if 'data' in mat_data and 'conductance' in mat_data['data'][0, 0].dtype.names:
        raw_data = np.array(mat_data['data']['conductance'][0, 0][0], dtype='float64')
        
        sampling_rate = 100.0  # Default
        if 'samplingrate' in mat_data['data'][0, 0].dtype.names:
            sampling_rate = float(mat_data['data']['samplingrate'][0, 0][0, 0])
    else:
        if 'conductance' in mat_data:
            raw_data = np.array(mat_data['conductance'], dtype='float64')
            sampling_rate = 100.0  # Default
        else:
            raise ValueError("Could not extract EDA data from .mat file")
    
    time_data = np.arange(len(raw_data)) / sampling_rate
    
    ref_onsets, ref_amplitudes = load_reference_file(reference_file)
    
    eda_data = EDAData(
        time_data=time_data,
        conductance_data=raw_data,
        sampling_rate=sampling_rate
    )
    
    tonic_data = np.ones_like(raw_data) * np.mean(raw_data)
    phasic_data = raw_data - tonic_data
    
    driver = np.zeros_like(raw_data)
    
    onset_indices = np.array([np.argmin(np.abs(time_data - onset)) for onset in ref_onsets])
    
    if method == "cda":
        tau = np.array([1.0, 3.75])  # Default CDA time constants
    else:  # dda
        tau = np.array([0.75, 20.0])  # Default DDA time constants
    
    kernel = bateman_gauss(time_data, 0, 1, tau[0], tau[1], 0)
    kernel = kernel / np.sum(kernel)
    
    analysis = EDAAnalysis(
        method=method,
        tau=tau,
        dist0=0.0 if method == "dda" else None,
        driver=driver,
        tonic_driver=np.zeros_like(driver),
        kernel=kernel,
        phasic_data=phasic_data,
        tonic_data=tonic_data,
        onset=ref_onsets,
        peak_time=ref_onsets + 1.0,  # Arbitrary peak time 1 second after onset
        amp=ref_amplitudes,
        area=ref_amplitudes * 0.5 if method == "dda" else None,  # Arbitrary area for DDA
        impulse_onset=ref_onsets,
        impulse_peak_time=ref_onsets + 0.5  # Arbitrary impulse peak time
    )
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_data, raw_data, 'k-', label='Raw EDA')
        plt.plot(time_data, tonic_data, 'b-', label='Tonic')
        plt.title(f'{method.upper()} - Reference Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(time_data, phasic_data, 'r-', label='Phasic')
        
        for onset, amp in zip(ref_onsets, ref_amplitudes):
            plt.axvline(x=onset, color='g', linestyle='--', alpha=0.5)
            plt.plot([onset, onset], [0, amp/1000], 'g-', alpha=0.5)
        
        plt.title('Phasic Component with SCRs')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method}_reference_analysis.png'))
    
    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create reference analysis from SCR data')
    parser.add_argument('mat_file', help='Path to .mat file with raw EDA data')
    parser.add_argument('reference_file', help='Path to reference file with SCR data')
    parser.add_argument('--method', choices=['cda', 'dda'], default='dda', help='Analysis method')
    parser.add_argument('--output-dir', help='Directory to save validation plots')
    
    args = parser.parse_args()
    analysis = create_reference_analysis(args.mat_file, args.reference_file, args.method, args.output_dir)
    
    print(f"Created {args.method.upper()} analysis with {len(analysis.amp)} SCRs")
    print(f"Onset range: {np.min(analysis.onset):.2f} - {np.max(analysis.onset):.2f} seconds")
    print(f"Amplitude range: {np.min(analysis.amp):.2f} - {np.max(analysis.amp):.2f} µS")
