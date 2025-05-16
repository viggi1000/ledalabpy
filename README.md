# LedalabPy

A Python port of the Ledalab MATLAB software for electrodermal activity (EDA) analysis.

## Installation

### From PyPI (Recommended)
```bash
pip install ledalabpy
```

### From GitHub
```bash
# Clone the repository
git clone https://github.com/viggi1000/ledalabpy.git
cd ledalabpy

# Install in development mode
pip install -e .
```

## Usage

```python
import numpy as np
import scipy.io
import ledalabpy

# Load data from .mat file
mat_data = scipy.io.loadmat('EDA_data.mat')
raw_data = np.array(mat_data['data']['conductance'][0][0][0], dtype='float64')
sampling_rate = 100  # Hz

# Analyze data using CDA (Continuous Decomposition Analysis)
analysis_cda = ledalabpy.analyze(raw_data, method="cda", sampling_rate=sampling_rate)

# Analyze data using DDA (Discrete Decomposition Analysis)
analysis_dda = ledalabpy.analyze(raw_data, method="dda", sampling_rate=sampling_rate)

# Access results
phasic_data_cda = analysis_cda.phasic_data
tonic_data_cda = analysis_cda.tonic_data
scr_amplitudes_cda = analysis_cda.amp
scr_onsets_cda = analysis_cda.onset

phasic_data_dda = analysis_dda.phasic_data
tonic_data_dda = analysis_dda.tonic_data
scr_amplitudes_dda = analysis_dda.amp
scr_areas_dda = analysis_dda.area

# Extract features
from ledalabpy import extract_features
features_cda = extract_features(analysis_cda)
features_dda = extract_features(analysis_dda)
print(features_cda)
print(features_dda)
```

## Features

- Data import and preprocessing
- Continuous Decomposition Analysis (CDA)
- Discrete Decomposition Analysis (DDA)
- Skin Conductance Response (SCR) detection
- Feature extraction

## Analysis Methods

### Continuous Decomposition Analysis (CDA)
CDA performs a decomposition of SC data into continuous signals of phasic and tonic activity. This method takes advantage of retrieving the signal characteristics of the underlying sudomotor nerve activity (SNA). It is beneficial for all analyses aiming at unbiased scores of phasic and tonic activity.

### Discrete Decomposition Analysis (DDA)
DDA performs a decomposition of SC data into distinct phasic components and a tonic component by means of Nonnegative Deconvolution. This method is especially advantageous for the study of the SCR shape.

## Requirements

- Python 3.9 or higher
- NumPy
- SciPy
- Pandas
- Matplotlib (optional, for visualization)

## License

MIT
