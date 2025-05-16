import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledalabpy

def compare_methods(file_path, output_dir=None):
    """
    Compare CDA and DDA methods on the same EDA data.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file with raw EDA data
    output_dir : str
        Directory to save comparison plots
    """
    mat_data = scipy.io.loadmat(file_path)
    
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
    
    analysis_cda = ledalabpy.analyze(raw_data, method="cda", sampling_rate=sampling_rate)
    analysis_dda = ledalabpy.analyze(raw_data, method="dda", sampling_rate=sampling_rate)
    
    features_cda = ledalabpy.extract_features(analysis_cda)
    features_dda = ledalabpy.extract_features(analysis_dda)
    
    print(f"Data Summary:")
    print(f"  File: {os.path.basename(file_path)}")
    print(f"  Length: {len(raw_data)} samples ({len(raw_data)/sampling_rate:.2f} seconds)")
    print(f"  Sampling rate: {sampling_rate} Hz")
    
    print("\nCDA Results:")
    print(f"  SCRs detected: {len(analysis_cda.amp) if analysis_cda.amp is not None else 0}")
    print(f"  Tonic mean: {features_cda.get('tonic_mean', 'N/A')}")
    print(f"  Phasic mean: {features_cda.get('phasic_mean', 'N/A')}")
    if 'scr_amplitude_mean' in features_cda:
        print(f"  SCR amplitude mean: {features_cda['scr_amplitude_mean']}")
    
    print("\nDDA Results:")
    print(f"  SCRs detected: {len(analysis_dda.amp) if analysis_dda.amp is not None else 0}")
    print(f"  Tonic mean: {features_dda.get('tonic_mean', 'N/A')}")
    print(f"  Phasic mean: {features_dda.get('phasic_mean', 'N/A')}")
    if 'scr_amplitude_mean' in features_dda:
        print(f"  SCR amplitude mean: {features_dda['scr_amplitude_mean']}")
    if 'scr_area_mean' in features_dda:
        print(f"  SCR area mean: {features_dda['scr_area_mean']}")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        time_data = np.arange(len(raw_data)) / sampling_rate
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_data, raw_data, 'k-')
        plt.title('Raw EDA Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.subplot(3, 1, 2)
        plt.plot(time_data, analysis_cda.tonic_data, 'b-', label='CDA')
        plt.plot(time_data, analysis_dda.tonic_data, 'g-', label='DDA')
        plt.title('Tonic Component Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(time_data, analysis_cda.phasic_data, 'b-', label='CDA')
        plt.plot(time_data, analysis_dda.phasic_data, 'g-', label='DDA')
        plt.title('Phasic Component Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_comparison.png'))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_data, raw_data, 'k-')
        
        if analysis_cda.onset is not None:
            for onset, amp in zip(analysis_cda.onset, analysis_cda.amp):
                plt.axvline(x=onset, color='b', linestyle='--', alpha=0.5)
                
        plt.title('CDA - SCR Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.subplot(2, 1, 2)
        plt.plot(time_data, raw_data, 'k-')
        
        if analysis_dda.onset is not None:
            for onset, amp in zip(analysis_dda.onset, analysis_dda.amp):
                plt.axvline(x=onset, color='g', linestyle='--', alpha=0.5)
                
        plt.title('DDA - SCR Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scr_detection_comparison.png'))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, analysis_cda.driver, 'b-', label='CDA')
        plt.plot(time_data, analysis_dda.driver, 'g-', label='DDA')
        plt.title('Driver Signal Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'driver_comparison.png'))
        
        print(f"\nComparison plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CDA and DDA methods on EDA data')
    parser.add_argument('file', help='Path to EDA data file (.mat)')
    parser.add_argument('--output-dir', default='comparison_results', help='Directory to save comparison plots')
    
    args = parser.parse_args()
    compare_methods(args.file, args.output_dir)
