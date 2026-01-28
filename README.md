---
# Dielectric Heterodimer Simulation: Mie Coefficients, Polarizability, and Fano Fit Analysis
---
## Overview

This Python project simulates the optical response of a two-sphere dielectric heterodimer using an electric-dipole coupled-dipole model. The project calculates Mie coefficients, derives electric dipole polarizabilities, generates extinction spectra, and performs Fano fit analysis. It also incorporates dispersion effects (e.g., crystalline silicon refractive index) for accurate material modeling.
---
## Requirements

Ensure that you have **Python 3.10+** installed, and then install the required libraries:

```bash
pip install -r requirements.txt
```
---
## Project Structure

- **`simulate_dimer.py`**: Main script to run the full simulation.
- **`requirements.txt`**: List of required Python libraries.
- **`figs/`**: Directory to save output figures (PDFs).
- **`data/`**: Directory to save data (CSV files).
---
## Running the Simulation

1. **Set Up Environment**:
   - Install the required dependencies using:

     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Simulation**:
   To execute the simulation, run:

   ```bash
   python simulate_dimer.py
    ```
---
## Key Functions in `simulate_dimer.py`

### 1. `save_figure(fig, name_prefix, figs_dir)`
   - **Purpose**: Saves a figure as a PDF with a timestamped file name.
   - **Parameters**:
     - `fig`: The figure object to be saved.
     - `name_prefix`: Prefix for the saved figure file name.
     - `figs_dir`: Directory where the figure will be saved.

### 2. `save_data(data, file_name, data_dir)`
   - **Purpose**: Saves a DataFrame as a CSV file in the specified directory.
   - **Parameters**:
     - `data`: The DataFrame containing the data to save.
     - `file_name`: The name of the CSV file.
     - `data_dir`: Directory where the data will be saved.

### 3. `mie_coefficients(n_particle_real, n_particle_imag, radius, wavelength)`
   - **Purpose**: Calculates Mie coefficients (`qext`) for a given particle using its refractive index, radius, and wavelength.
   - **Parameters**:
     - `n_particle_real`: Real part of the refractive index.
     - `n_particle_imag`: Imaginary part of the refractive index.
     - `radius`: Radius of the sphere.
     - `wavelength`: Array of wavelengths for the simulation.
   - **Returns**: The Mie coefficient `a1(λ)` for the particle.

### 4. `mie_derived_polarizability(a1, wavelength, n_m)`
   - **Purpose**: Converts Mie coefficients into electric dipole polarizabilities.
   - **Parameters**:
     - `a1`: Mie coefficient `a1(λ)` for the sphere.
     - `wavelength`: Array of wavelengths.
     - `n_m`: Refractive index of the background medium.
   - **Returns**: The electric dipole polarizability `α(λ)` for the sphere.

### 5. `dyadic_green_function(R, k)`
   - **Purpose**: Computes the Green’s function for particle-particle interactions in the coupled-dipole model.
   - **Parameters**:
     - `R`: Vector representing the distance between two particles.
     - `k`: Wave number for the given wavelength.
   - **Returns**: The Green’s function matrix for the system.

### 6. `solve_coupled_dipoles(alpha1, alpha2, E_inc1, E_inc2, R, k_values)`
   - **Purpose**: Solves the coupled-dipole equations for two spheres to compute the dipole moments for each wavelength.
   - **Parameters**:
     - `alpha1`, `alpha2`: Polarizabilities for each sphere.
     - `E_inc1`, `E_inc2`: Incident electric fields for spheres 1 and 2.
     - `R`: Distance between the two spheres.
     - `k_values`: Array of wave numbers.
   - **Returns**: The dipole moments for both spheres for each wavelength.

### 7. `solve_dimer_system(alpha1_arr, alpha2_arr, r1_pos, r2_pos, wl_arr, pol_vec)`
   - **Purpose**: Solves the system of coupled dipoles and calculates the extinction proxy spectrum for different gap sizes and polarizations.
   - **Parameters**:
     - `alpha1_arr`, `alpha2_arr`: Arrays of polarizabilities for each sphere.
     - `r1_pos`, `r2_pos`: Positions of spheres 1 and 2.
     - `wl_arr`: Array of wavelengths.
     - `pol_vec`: Polarization direction.
   - **Returns**: The extinction proxy spectrum \( S(\lambda) \) for each wavelength.

### 8. `fano_func(lam, y0, A, lam0, Gamma, q)`
   - **Purpose**: Defines the Fano lineshape function to fit asymmetric spectral features.
   - **Parameters**:
     - `lam`: Array of wavelengths.
     - `y0`, `A`, `lam0`, `Gamma`, `q`: Fit parameters for the Fano model.
   - **Returns**: The fitted Fano lineshape at each wavelength.

### 9. `compute_condition_number(A)`
   - **Purpose**: Computes the condition number of matrix `A` to assess the stability of the coupled-dipole system.
   - **Parameters**:
     - `A`: The matrix to compute the condition number for.
   - **Returns**: The condition number of the matrix `A`.
---
## Example Output

1. **Figures**:
   - The simulation generates and saves the following figures as PDF files in the `figs/` directory:
     - **Mie Coefficients**: A plot showing the Mie coefficients for both spheres.
     - **Polarizabilities**: A plot depicting the Mie-derived electric dipole polarizability for each sphere.
     - **Extinction Spectra**: Graphs displaying the extinction spectra for various gap sizes and polarization directions.
     - **Fano Fit Results**: A figure showing the Fano fit overlaid on the extinction spectra, with residuals.

2. **Data Files**:
   - CSV files containing the following data will be saved in the `data/` directory:
     - **Mie Coefficients**: Mie coefficients for both spheres at each wavelength.
     - **Polarizabilities**: Electric dipole polarizabilities for both spheres at each wavelength.
     - **Extinction Spectra**: Extinction spectra for different gap sizes and polarization configurations.

