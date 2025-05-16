import numpy as np
import os
import sys
import scipy.io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledalabpy

def debug_reference_loading():
    """Debug the reference data loading for EDA1_64.mat file."""
    data = np.ones(6400)
    time = np.linspace(0, 63.99, 6400)
    
    print(f"Data shape: {data.shape}, Time range: {time[0]}-{time[-1]}")
    
    mat_file = os.path.expanduser("~/attachments/ca036f60-210a-411f-9371-131d7f83fe47/EDA1_64.mat")
    if os.path.exists(mat_file):
        print(f"Loading actual EDA1_64.mat file: {mat_file}")
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
        
        print(f"Actual data shape: {raw_data.shape}, Sampling rate: {sampling_rate}")
        
        print("\nAnalyzing actual data with CDA...")
        analysis = ledalabpy.analyze(raw_data, method='cda', sampling_rate=sampling_rate)
        print(f"SCRs detected: {len(analysis.onset) if analysis.onset is not None else 0}")
        
        ref_file = os.path.expanduser("~/attachments/7d6c7f61-f2e3-4b80-be5b-e1f4251a2c99/EDA1_64_scrlist_CDA.txt")
        print(f"\nReference file: {ref_file}")
        print(f"Reference file exists: {os.path.exists(ref_file)}")
        
        if os.path.exists(ref_file):
            ref_data = np.loadtxt(ref_file, skiprows=1)
            print(f"Reference data shape: {ref_data.shape}")
            
            if ref_data.ndim == 1:
                ref_onsets = np.array([ref_data[0]])
                ref_amplitudes = np.array([ref_data[1]])
            else:
                ref_onsets = ref_data[:, 0]
                ref_amplitudes = ref_data[:, 1]
            
            print(f"Reference onsets: {len(ref_onsets)}")
            print(f"Reference amplitudes range: {np.min(ref_amplitudes):.2f} - {np.max(ref_amplitudes):.2f}")
    else:
        print(f"EDA1_64.mat file not found: {mat_file}")
        
        print("\nAnalyzing synthetic data with CDA...")
        analysis = ledalabpy.analyze(data, method='cda', sampling_rate=100.0)
        print(f"SCRs detected: {len(analysis.onset) if analysis.onset is not None else 0}")

if __name__ == "__main__":
    debug_reference_loading()
