import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.interpolate import interp1d

# Set the font to Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'

# Create output folders for figures and data
figs_dir = "figs"
data_dir = "data"
os.makedirs(figs_dir, exist_ok=True)  # Create the figures folder
os.makedirs(data_dir, exist_ok=True)  # Create the data folder

# Function to save figures with a unique name based on timestamp
def save_figure(fig, name_prefix, figs_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Create a timestamp for uniqueness
    file_name = f"{name_prefix}_{timestamp}.pdf"  # Name the file with the timestamp
    file_path = os.path.join(figs_dir, file_name)  # Save in the 'figs' folder
    fig.savefig(file_path, format='pdf')  # Save the figure in PDF format
    print(f"Figure saved as: {file_path}")  # Print the saved file's path

# Part I: Track 1 (constant refractive index)
# Constants for Track 1 (constant refractive index)
n_m1 = 3.47  # Refractive index of Sphere 1
n_m2 = 2.41  # Refractive index of Sphere 2
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
wavelength = np.linspace(500, 1000, 400) * 1e-9  # Wavelength range (500 nm to 1000 nm)

# Mie Coefficient Calculation (Track 1: constant refractive index)
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Calculate Mie coefficients for both spheres
a1_sphere1 = mie_coefficient(r1, wavelength, n_m1)
a1_sphere2 = mie_coefficient(r2, wavelength, n_m2)

# Plot the absolute value of the polarizability for both spheres
fig, ax = plt.subplots(figsize=(10, 10))  # Square-shaped figure
ax.plot(wavelength * 1e9, np.abs(a1_sphere1), label=f'Sphere 1 (n={n_m1})', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength * 1e9, np.abs(a1_sphere2), label=f'Sphere 2 (n={n_m2})', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)| (C·m²/V)', fontsize=14, fontweight='bold')  # Added units to the y-axis
ax.set_title('Track 1: Mie Coefficients for Dielectric Spheres (Constant Refractive Index)', fontsize=16, fontweight='bold')

# Bold the tick labels and values
ax.tick_params(axis='both', which='major', labelsize=12, width=2)  # Bold the ticks and make them thicker
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')  # Bold the tick numbers (labels)

# Remove grid lines as per your request
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
from matplotlib import font_manager
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "mie_coefficients_track1", figs_dir)

# Show the figure
plt.show()

# Track 2: Dispersion for Crystalline Si (c-Si) for Sphere 1 only
# Load the refractive index and extinction coefficient data for crystalline Si from CSV
data = pd.read_csv('data/Wang-25C.csv')

# Extract the wavelength, n (refractive index), and k (extinction coefficient)
wavelength_data = data['Wavelength'].values * 1e-9  # Convert to meters
n_data = data['n'].values
k_data = data['k'].values

# Interpolate the n(λ) and k(λ) values onto the simulation wavelength grid
wavelength_simulation = np.linspace(500, 1000, 400) * 1e-9  # Wavelength grid (500 nm to 1000 nm)

# Interpolate refractive index and extinction coefficient
n_interp = interp1d(wavelength_data, n_data, kind='cubic', fill_value="extrapolate")
k_interp = interp1d(wavelength_data, k_data, kind='cubic', fill_value="extrapolate")

# Get the interpolated values for n(λ) and k(λ)
n_interp_values = n_interp(wavelength_simulation)
k_interp_values = k_interp(wavelength_simulation)

# Calculate the complex refractive index ˜n(λ) = n(λ) + i k(λ) for Sphere 1 (only)
n_complex = n_interp_values + 1j * k_interp_values

# Mie Coefficient Calculation with Dispersion for Sphere 1
def mie_coefficient_dispersion(radius, wavelength, n_complex):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1 * n_complex

# Constants for spheres
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters

# Calculate Mie coefficients for Sphere 1 with the complex refractive index (dispersion) and Sphere 2 with constant refractive index
a1_sphere1_disp = mie_coefficient_dispersion(r1, wavelength_simulation, n_complex)
a1_sphere2_const = mie_coefficient(r2, wavelength_simulation, 2.41)  # Use constant refractive index for Sphere 2

# Plot the absolute value of the polarizability for both spheres (Track 2)
fig, ax = plt.subplots(figsize=(10, 10))  # Square-shaped figure
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1_disp), label=f'Sphere 1 (c-Si)', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere2_const), label=f'Sphere 2 (n=2.41)', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)| (C·m²/V)', fontsize=14, fontweight='bold')  # Added units to the y-axis
ax.set_title('Track 2: Mie Coefficients for Dielectric Spheres (Dispersion)', fontsize=16, fontweight='bold')

# Bold the tick labels and values
ax.tick_params(axis='both', which='major', labelsize=12, width=2)  # Bold the ticks and make them thicker
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')  # Bold the tick numbers (labels)

# Remove grid lines as per your request
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
from matplotlib import font_manager
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "mie_coefficients_track2", figs_dir)

# Show the figure
plt.show()




#Part II





from scipy.linalg import solve
import numpy as np

# Define the Dyadic Green function (describes the dipole-dipole interaction)
def dyadic_green_function(R, wavelength):
    """
    Calculate the Green function for the dipole-dipole interaction between two spheres.
    
    R : ndarray
        The vector from sphere 1 to sphere 2 (3D distance vector).
    wavelength : ndarray
        The array of wavelengths for which the Green function is calculated.
    
    Returns:
    G : ndarray
        A 3x3xN array of Green function matrices for each wavelength.
    """
    R_mag = np.linalg.norm(R)  # Magnitude of the distance vector between the spheres
    R_hat = R / R_mag  # Unit vector along the line connecting the two spheres
    
    # Initialize the Green function array
    G = np.zeros((3, 3, len(wavelength)), dtype=complex)
    
    # Loop through each wavelength and calculate the Green function for that wavelength
    for i in range(len(wavelength)):
        k = 2 * np.pi / wavelength[i]  # Wave number (k = 2π/λ)
        
        # Green function for dipole-dipole interaction
        G_i = np.exp(1j * k * R_mag) / (4 * np.pi * R_mag**3) * (
            (k**2 / R_mag**2) * (np.eye(3) - np.outer(R_hat, R_hat)) +
            (1 - 1j * k * R_mag) * (3 * np.outer(R_hat, R_hat) - np.eye(3))
        )
        G[:, :, i] = G_i  # Store the Green function for this wavelength

    return G

# Mie Coefficient Calculation (using the complex refractive index for c-Si, and constant for Sphere 2)
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Constants for spheres and simulation setup
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
wavelength_simulation = np.linspace(500, 1000, 400) * 1e-9  # Wavelength range (500 nm to 1000 nm)

# Distance vector between the spheres (assumed along the x-axis)
R = np.array([1e-6, 0, 0])  # Distance in meters

# Get the Green function for the interaction between spheres
G = dyadic_green_function(R, wavelength_simulation)

# For Track 1 (constant refractive index)
a1_sphere1 = mie_coefficient(r1, wavelength_simulation, 3.47)  # For Sphere 1 (n = 3.47)
a1_sphere2 = mie_coefficient(r2, wavelength_simulation, 2.41)  # For Sphere 2 (n = 2.41)

# For Track 2 (with dispersion for Sphere 1 and constant for Sphere 2)
def mie_coefficient_dispersion(radius, wavelength, n_complex):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1 * n_complex

# For Track 2: Dispersion (Sphere 1) and Constant Index (Sphere 2)
# Calculate Mie coefficients for Sphere 1 (using complex refractive index for c-Si) and Sphere 2 (constant refractive index)
a1_sphere1_disp = mie_coefficient_dispersion(r1, wavelength_simulation, 3.47)  # For Sphere 1 (c-Si)
a1_sphere2_const = mie_coefficient(r2, wavelength_simulation, 2.41)  # For Sphere 2 (constant refractive index)

# Constructing the 6x6 matrix system at each wavelength
A = np.zeros((6, 6, len(wavelength_simulation)), dtype=complex)

# Fill in the diagonal elements (self-interactions) for each wavelength
for i in range(len(wavelength_simulation)):
    A[:3, :3, i] = np.diag([a1_sphere1_disp[i], a1_sphere1_disp[i], a1_sphere1_disp[i]])  # Self-interaction of Sphere 1 (with dispersion)
    A[3:6, 3:6, i] = np.diag([a1_sphere2_const[i], a1_sphere2_const[i], a1_sphere2_const[i]])  # Self-interaction of Sphere 2 (constant refractive index)

# Fill in the off-diagonal elements (interaction between spheres) for each wavelength
for i in range(len(wavelength_simulation)):
    A[:3, 3:6, i] = G[:, :, i]  # Interaction between Sphere 1 and Sphere 2
    A[3:6, :3, i] = G[:, :, i].conj()  # Interaction between Sphere 2 and Sphere 1 (complex conjugate)

# External electric fields (unit fields for simplicity)
E1 = np.ones(3, dtype=complex)  # External field on Sphere 1
E2 = np.ones(3, dtype=complex)  # External field on Sphere 2

# Set up the right-hand side of the system (external fields acting on the spheres)
b = np.hstack([E1, E2])

# Solve the system of equations to get the dipole moments of each sphere at each wavelength
dipole_moments = np.zeros((6, len(wavelength_simulation)), dtype=complex)

for i in range(len(wavelength_simulation)):
    dipole_moments[:, i] = solve(A[:, :, i], b)  # Solve the system for each wavelength

# Output the dipole moments for each wavelength
print("Dipole Moments at each wavelength:", dipole_moments)




#Part III