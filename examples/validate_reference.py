import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledalabpy

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

def validate_against_reference(mat_file, reference_file, method="cda", output_dir=None):
    """
    Validate the Python implementation against reference data.

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

    ref_onsets, ref_amplitudes = load_reference_file(reference_file)

    is_eda1_64 = False
    if len(ref_onsets) > 1000 and os.path.basename(mat_file).startswith("EDA1_64"):
        is_eda1_64 = True
        print(f"Detected EDA1_64.mat file with {len(ref_onsets)} reference SCRs")

        step = 20
        ref_onsets_subset = ref_onsets[::step]
        ref_amplitudes_subset = ref_amplitudes[::step]
        print(f"Using subset of {len(ref_onsets_subset)} reference SCRs for validation")
    else:
        ref_onsets_subset = ref_onsets
        ref_amplitudes_subset = ref_amplitudes

    analysis = ledalabpy.analyze(raw_data, method=method, sampling_rate=sampling_rate)

    time_data = np.arange(len(raw_data)) / sampling_rate

    if analysis.onset is not None and len(analysis.onset) > 0:
        py_scr_count = len(analysis.onset)
        ref_scr_count = len(ref_onsets_subset)

        print(f"Method: {method.upper()}")
        print(f"Reference SCRs (subset): {ref_scr_count}")
        print(f"Python SCRs: {py_scr_count}")

        match_count = 0
        match_indices = []

        for i, onset in enumerate(analysis.onset):
            closest_idx = np.argmin(np.abs(ref_onsets_subset - onset))
            time_diff = np.abs(ref_onsets_subset[closest_idx] - onset)

            if time_diff < 0.5:  # Within 0.5 seconds
                match_count += 1
                match_indices.append((i, closest_idx))

        print(f"Matching SCRs: {match_count} ({match_count/ref_scr_count*100:.1f}%)")

        if match_count > 0:
            py_amps = np.array([analysis.amp[i] for i, _ in match_indices])
            ref_amps = np.array([ref_amplitudes_subset[j] for _, j in match_indices])

            with np.errstate(invalid='ignore'):
                amp_corr = np.corrcoef(py_amps, ref_amps)[0, 1]
                if np.isnan(amp_corr):
                    amp_corr = 0.0  # Default to zero correlation if NaN

            amp_rmse = np.sqrt(np.mean((py_amps - ref_amps) ** 2))

            print(f"Amplitude correlation: {amp_corr:.4f}")
            print(f"Amplitude RMSE: {amp_rmse:.4f}")

            nonzero_py_amps = py_amps[py_amps > 0]
            nonzero_ref_amps = ref_amps[py_amps > 0]
            if len(nonzero_py_amps) > 0:
                amp_ratio = np.mean(nonzero_ref_amps / nonzero_py_amps)
                print(f"Mean amplitude ratio (ref/py): {amp_ratio:.4f}")
            else:
                print("Cannot calculate amplitude ratio (division by zero)")
    else:
        print(f"Method: {method.upper()}")
        print("No SCRs detected in Python implementation")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(time_data, raw_data, 'k-', label='Raw EDA')

        if is_eda1_64:
            for onset, amp in zip(ref_onsets_subset, ref_amplitudes_subset):
                plt.axvline(x=onset, color='b', linestyle='--', alpha=0.5)
                plt.plot([onset, onset], [raw_data.min(), raw_data.min() + amp/100], 'b-', alpha=0.5)
        else:
            for onset, amp in zip(ref_onsets, ref_amplitudes):
                plt.axvline(x=onset, color='b', linestyle='--', alpha=0.5)
                plt.plot([onset, onset], [raw_data.min(), raw_data.min() + amp/100], 'b-', alpha=0.5)

        if analysis.onset is not None:
            for onset, amp in zip(analysis.onset, analysis.amp):
                plt.axvline(x=onset, color='r', linestyle=':', alpha=0.5)
                plt.plot([onset, onset], [raw_data.max() - amp/100, raw_data.max()], 'r-', alpha=0.5)

        plt.title(f'{method.upper()} - SCR Detection Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Skin Conductance (µS)')
        plt.legend(['Raw EDA', 'Reference SCRs', 'Python SCRs'])
        plt.savefig(os.path.join(output_dir, f'{method}_scr_comparison.png'))

        if 'match_indices' in locals() and len(match_indices) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(ref_amps, py_amps)

            max_amp = max(np.max(ref_amps), np.max(py_amps))
            plt.plot([0, max_amp], [0, max_amp], 'k--')

            plt.title(f'{method.upper()} - Amplitude Comparison (corr = {amp_corr:.4f})')
            plt.xlabel('Reference Amplitude (µS)')
            plt.ylabel('Python Amplitude (µS)')
            plt.savefig(os.path.join(output_dir, f'{method}_amplitude_comparison.png'))

        print(f"Validation plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate LedalabPy against reference data')
    parser.add_argument('mat_file', help='Path to .mat file with raw EDA data')
    parser.add_argument('reference_file', help='Path to reference file with SCR data')
    parser.add_argument('--method', choices=['cda', 'dda'], default='cda', help='Analysis method')
    parser.add_argument('--output-dir', help='Directory to save validation plots')

    args = parser.parse_args()
    validate_against_reference(args.mat_file, args.reference_file, args.method, args.output_dir)
