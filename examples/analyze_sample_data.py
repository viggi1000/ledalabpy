import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledalabpy

def analyze_mat_file(file_path, method="cda", optimize=2, output_dir=None):
    """
    Analyze EDA data from a .mat file using LedalabPy.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file
    method : str
        Analysis method ('cda' or 'dda')
    optimize : int
        Optimization level (0=none, 1=quick, 2=full)
    output_dir : str
        Directory to save output plots
    """
    print(f"Analyzing {os.path.basename(file_path)} using {method.upper()}...")
    
    mat_data = scipy.io.loadmat(file_path)
    
    if 'data' in mat_data and 'conductance' in mat_data['data'][0, 0]:
        raw_data = np.array(mat_data['data']['conductance'][0, 0][0], dtype='float64')
        
        sampling_rate = 100.0  # Default
        if 'samplingrate' in mat_data['data'][0, 0]:
            sampling_rate = float(mat_data['data']['samplingrate'][0, 0][0, 0])
    else:
        if 'conductance' in mat_data:
            raw_data = np.array(mat_data['conductance'], dtype='float64')
            sampling_rate = 100.0  # Default
        else:
            raise ValueError("Could not extract EDA data from .mat file")
    
    print(f"Data length: {len(raw_data)} samples ({len(raw_data)/sampling_rate:.2f} seconds)")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    analysis = ledalabpy.analyze(raw_data, method=method, sampling_rate=sampling_rate, optimize=optimize)
    
    features = ledalabpy.extract_features(analysis)
    
    print(f"\nAnalysis Results ({method.upper()}):")
    print(f"  Tau: {analysis.tau}")
    if method == "dda":
        print(f"  Dist0: {analysis.dist0}")
    print(f"  SCRs detected: {len(analysis.amp) if analysis.amp is not None else 0}")
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        time_data = np.arange(len(raw_data)) / sampling_rate
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_data, raw_data, 'k-')
        plt.title(f'Raw EDA Signal - {method.upper()} Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.subplot(3, 1, 2)
        plt.plot(time_data, analysis.tonic_data, 'b-', label='Tonic')
        plt.plot(time_data, analysis.phasic_data, 'r-', label='Phasic')
        plt.title('Tonic and Phasic Components')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(time_data, analysis.driver, 'g-')
        
        if analysis.onset is not None:
            for onset, amp in zip(analysis.onset, analysis.amp):
                plt.axvline(x=onset, color='r', linestyle='--', alpha=0.5)
                
        plt.title('Driver with SCR Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{os.path.basename(file_path)}_{method}_analysis.png'))
        print(f"\nPlots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze EDA data with LedalabPy')
    parser.add_argument('file', help='Path to .mat file with EDA data')
    parser.add_argument('--method', choices=['cda', 'dda'], default='cda', help='Analysis method')
    parser.add_argument('--optimize', type=int, choices=[0, 1, 2], default=2, help='Optimization level')
    parser.add_argument('--output-dir', help='Directory to save output plots')
    
    args = parser.parse_args()
    analyze_mat_file(args.file, args.method, args.optimize, args.output_dir)
