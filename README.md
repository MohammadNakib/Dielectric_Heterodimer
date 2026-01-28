# Mie Coefficient and Polarizability Calculations for Spheres

## Overview

This project calculates Mie coefficients, polarizabilities, and extinction proxies for two spheres with different refractive indices. The code is designed to handle calculations for both idealized spheres and spheres with crystalline Silicon dispersion properties. It also performs a Fano fit analysis to extract resonance, width, and asymmetry from extinction spectra. The code visualizes and saves the results in both plots and CSV data files.

## Key Features

- **Mie Coefficients Calculation**: Computes Mie scattering efficiencies for two spheres using different refractive indices and radii. 
- **Electric Dipole Polarizability**: Calculates the Mie-derived electric dipole polarizability based on Mie coefficients.
- **Crystalline Silicon Dispersion**: Includes an option to perform the same calculations using real and imaginary parts of the refractive index for crystalline Silicon.
- **Extinction Proxy Spectra**: Solves a coupled dipole system for different gaps and polarizations, computing extinction proxy spectra for each configuration.
- **Fano Fit Analysis**: Performs a Fano fit on the extinction spectra to extract resonance wavelength, width, and asymmetry.
- **Data Export**: Saves figures and data as PDF and CSV files for easy reference.

## File Structure

- `figs/`: Directory where plots (PDF) are saved.
- `data/`: Directory where the CSV data for Mie coefficients, polarizabilities, and extinction spectra are saved.
- `Wang-25C.csv`: CSV file containing the refractive index data for crystalline Silicon (for Track 2 calculations).

## Requirements

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `miepython`
  - `scipy`

You can install the required libraries by running:
```bash
pip install numpy matplotlib pandas miepython scipy
How to Run
Mie Coefficient Calculation (Track 1):

Calculates the Mie coefficients for two spheres with given refractive indices (n_m1 = 3.47, n_m2 = 2.41).

The radii of the spheres are r1 = 90e-9 meters and r2 = 65e-9 meters.

The wavelengths range from 500 nm to 1000 nm.

Crystalline Silicon Dispersion Calculation (Track 2):

Loads crystalline Silicon refractive index data (Wang-25C.csv) and computes Mie coefficients and polarizabilities for both spheres using this data.

Extinction Proxy Calculation:

Computes extinction proxy spectra for different gaps between the spheres (e.g., 10 nm, 20 nm, etc.) and for two different polarization states (longitudinal and transverse).

Fano Fit Analysis:

Fits a Fano resonance model to the extinction proxy spectra for a specific gap size and polarization. The resonance wavelength, width, and asymmetry are extracted.

Example Output
The code generates the following outputs:

Plots of Mie coefficients and polarizabilities for both spheres.

Extinction proxy spectra for different gap sizes and polarizations.

Fano fit analysis results including the resonance wavelength, width, and asymmetry.

CSV files with Mie coefficients, polarizabilities, and extinction proxy spectra data.

Saving Data
The code automatically saves the figures in the figs/ directory and the corresponding data in the data/ directory as CSV files. The filenames include timestamps to ensure uniqueness. You can use the data for further analysis or visualization.

Notes
The code assumes that the refractive index data for crystalline Silicon is stored in a CSV file named Wang-25C.csv. If this file is not available, the code will not perform the Track 2 calculations.

The Fano fit analysis is performed only for a specific polarization and gap size. You can modify the polarization and gap size for other analyses.
