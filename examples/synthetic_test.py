import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledalabpy

def create_synthetic_data(duration=60, sampling_rate=100, plot=True, output_dir=None):
    """
    Create synthetic EDA data and analyze it with both CDA and DDA methods.
    
    Parameters:
    -----------
    duration : float
        Duration of the synthetic data in seconds
    sampling_rate : float
        Sampling rate in Hz
    plot : bool
        Whether to plot the results
    output_dir : str
        Directory to save output plots
    """
    print(f"Creating synthetic EDA data ({duration} seconds at {sampling_rate} Hz)...")
    
    time_data = np.linspace(0, duration, int(duration * sampling_rate))
    
    tonic = 2 + 0.1 * time_data + 0.05 * np.sin(2 * np.pi * 0.01 * time_data)
    
    from ledalabpy.analyze.template import bateman_gauss
    
    phasic = np.zeros_like(time_data)
    scr_times = [5, 15, 25, 35, 45]
    scr_amps = [0.5, 0.7, 0.3, 0.6, 0.4]
    
    for t, a in zip(scr_times, scr_amps):
        phasic += bateman_gauss(time_data, t, a, 0.75, 2, 0.1)
    
    data = tonic + phasic
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, len(data))
    data += noise
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, data, 'k-', label='Raw EDA')
        plt.plot(time_data, tonic, 'b-', label='Tonic Component')
        plt.plot(time_data, phasic, 'r-', label='Phasic Component')
        plt.title('Synthetic EDA Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'synthetic_data.png'))
        else:
            plt.show()
    
    print("\nAnalyzing with CDA...")
    analysis_cda = ledalabpy.analyze(data, method="cda", sampling_rate=sampling_rate, optimize=2)
    
    features_cda = ledalabpy.extract_features(analysis_cda)
    
    print(f"  Tau: {analysis_cda.tau}")
    print(f"  SCRs detected: {len(analysis_cda.amp) if analysis_cda.amp is not None else 0}")
    print("\nCDA Features:")
    for key, value in features_cda.items():
        print(f"  {key}: {value}")
    
    print("\nAnalyzing with DDA...")
    analysis_dda = ledalabpy.analyze(data, method="dda", sampling_rate=sampling_rate, optimize=2)
    
    features_dda = ledalabpy.extract_features(analysis_dda)
    
    print(f"  Tau: {analysis_dda.tau}")
    print(f"  Dist0: {analysis_dda.dist0}")
    print(f"  SCRs detected: {len(analysis_dda.amp) if analysis_dda.amp is not None else 0}")
    print("\nDDA Features:")
    for key, value in features_dda.items():
        print(f"  {key}: {value}")
    
    if plot and output_dir is not None:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_data, data, 'k-')
        plt.title('CDA Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.subplot(3, 1, 2)
        plt.plot(time_data, analysis_cda.tonic_data, 'b-', label='Tonic')
        plt.plot(time_data, analysis_cda.phasic_data, 'r-', label='Phasic')
        plt.title('Tonic and Phasic Components')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(time_data, analysis_cda.driver, 'g-')
        
        if analysis_cda.onset is not None:
            for onset, amp in zip(analysis_cda.onset, analysis_cda.amp):
                plt.axvline(x=onset, color='r', linestyle='--', alpha=0.5)
                
        plt.title('Driver with SCR Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cda_analysis.png'))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_data, data, 'k-')
        plt.title('DDA Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        
        plt.subplot(3, 1, 2)
        plt.plot(time_data, analysis_dda.tonic_data, 'b-', label='Tonic')
        plt.plot(time_data, analysis_dda.phasic_data, 'r-', label='Phasic')
        plt.title('Tonic and Phasic Components')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(time_data, analysis_dda.driver, 'g-')
        
        if analysis_dda.onset is not None:
            for onset, amp in zip(analysis_dda.onset, analysis_dda.amp):
                plt.axvline(x=onset, color='r', linestyle='--', alpha=0.5)
                
        plt.title('Driver with SCR Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dda_analysis.png'))
    
    print("\nComparing with ground truth:")
    tonic_corr_cda = np.corrcoef(analysis_cda.tonic_data, tonic)[0, 1]
    phasic_corr_cda = np.corrcoef(analysis_cda.phasic_data, phasic)[0, 1]
    
    tonic_corr_dda = np.corrcoef(analysis_dda.tonic_data, tonic)[0, 1]
    phasic_corr_dda = np.corrcoef(analysis_dda.phasic_data, phasic)[0, 1]
    
    print(f"  CDA Tonic correlation: {tonic_corr_cda:.4f}")
    print(f"  CDA Phasic correlation: {phasic_corr_cda:.4f}")
    print(f"  DDA Tonic correlation: {tonic_corr_dda:.4f}")
    print(f"  DDA Phasic correlation: {phasic_corr_dda:.4f}")
    
    return {
        'time_data': time_data,
        'data': data,
        'tonic': tonic,
        'phasic': phasic,
        'analysis_cda': analysis_cda,
        'analysis_dda': analysis_dda,
        'features_cda': features_cda,
        'features_dda': features_dda
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and analyze synthetic EDA data with LedalabPy')
    parser.add_argument('--duration', type=float, default=60, help='Duration of synthetic data in seconds')
    parser.add_argument('--sampling-rate', type=float, default=100, help='Sampling rate in Hz')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--output-dir', default='output', help='Directory to save output plots')
    
    args = parser.parse_args()
    create_synthetic_data(args.duration, args.sampling_rate, not args.no_plot, args.output_dir)
