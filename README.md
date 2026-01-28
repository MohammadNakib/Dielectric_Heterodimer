# Dielectric Heterodimer Simulation: Mie Coefficients, Polarizability, and Fano Fit Analysis

## Overview

This Python project simulates the optical response of a two-sphere dielectric heterodimer using an electric-dipole coupled-dipole model. The project calculates Mie coefficients, derives electric dipole polarizabilities, generates extinction spectra, and performs Fano fit analysis. It also incorporates dispersion effects (e.g., crystalline silicon refractive index) for accurate material modeling.

## Requirements

Ensure that you have **Python 3.10+** installed, and then install the required libraries:

```bash
pip install -r requirements.txt
```
Project Structure
simulate_dimer.py: Main script to run the full simulation.

requirements.txt: List of required Python libraries.

figs/: Directory to save output figures (PDFs).

data/: Directory to save data (CSV files).

Running the Simulation
Set Up Environment:

Install the required dependencies using:

pip install -r requirements.txt
Run the Simulation:
To execute the simulation, run:

python simulate_dimer.py
This will:

Calculate Mie coefficients for two spheres.

Compute electric dipole polarizabilities.

Generate extinction spectra for different gap sizes and polarization configurations.

Perform Fano fit analysis on selected extinction spectra.

View the Results:

Figures: The figs/ directory will contain PDF files of Mie coefficients, polarizabilities, extinction spectra, and Fano fit results.

Data: The data/ directory will contain CSV files with Mie coefficients, polarizabilities, and extinction spectra data.

Key Functions in simulate_dimer.py
1. save_figure(fig, name_prefix, figs_dir)
Saves a figure as a PDF with a timestamped file name.

2. save_data(data, file_name, data_dir)
Saves a DataFrame as a CSV file in the specified directory.

3. mie_coefficients(n_particle_real, n_particle_imag, radius, wavelength)
Calculates Mie coefficients (qext) for a given particle using its refractive index, radius, and wavelength.

4. mie_derived_polarizability(a1, wavelength, n_m)
Converts Mie coefficients into electric dipole polarizabilities.

5. dyadic_green_function(R, k)
Computes the Green's function for particle-particle interactions in the coupled-dipole model.

6. solve_coupled_dipoles(alpha1, alpha2, E_inc1, E_inc2, R, k_values)
Solves the coupled-dipole equations for two spheres and computes the dipole moments for each wavelength.

7. solve_dimer_system(alpha1_arr, alpha2_arr, r1_pos, r2_pos, wl_arr, pol_vec)
Solves the system of coupled dipoles and calculates the extinction proxy spectrum for different gap sizes and polarizations.

8. fano_func(lam, y0, A, lam0, Gamma, q)
Defines the Fano lineshape function to fit asymmetric spectral features.

9. compute_condition_number(A)
Computes the condition number of matrix A to assess the stability of the system.

Example Output
Figures:

Plots of Mie coefficients, polarizabilities, extinction spectra, and Fano fits are saved as PDF files in the figs/ directory.

Data Files:

CSV files for Mie coefficients, polarizabilities, and extinction spectra are saved in the data/ directory.
