# Dielectric Heterodimer Simulation: Coupled-Dipole Model, Fano Fit, and Full-Wave Verification

## Overview

This project simulates the optical response of a two-sphere dielectric heterodimer using an electric-dipole coupled-dipole model. The goal is to compute and analyze asymmetric spectral features using Mie scattering, fit them with a Fano lineshape, and study how the asymmetry varies with gap size and polarization direction. Additionally, the project verifies the results against full-wave simulations.

### Key Features:
- **Mie Coefficients Calculation** for two spheres using their refractive indices and radii.
- **Polarizability Calculation** derived from the Mie coefficients.
- **Coupled-Dipole Model** to simulate the interaction between two particles.
- **Fano Fit** to identify asymmetric spectral features.
- **Full-Wave Verification** comparing the coupled-dipole model results with a full-wave solver (e.g., Meep, COMSOL).

## Files Included

- `simulate_dimer.py`: Python script for running the simulation.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: This file containing instructions.
- `report.tex`: LaTeX file to compile a report summarizing the results.
- `figs/`: Folder containing generated figures (in PDF format).
- `data/`: Folder containing material data (e.g., refractive index data for crystalline Si).

## Installation

### Step 1: Clone the repository
Clone the project to your local machine:
```bash
git clone https://your-repository-url.git
cd your-repository
Step 2: Set up a virtual environment
It is recommended to use a virtual environment to manage dependencies:

python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Step 3: Install dependencies
Install the required dependencies using the requirements.txt file:

pip install -r requirements.txt
Step 4: Install additional dependencies
Make sure the following dependencies are installed:

miepython for Mie scattering coefficient calculations.

numpy, scipy, and matplotlib for numerical computations and plotting.

pandas for data handling.

scipy.optimize.curve_fit for Fano fit.

Running the Simulation
Once the dependencies are installed, you can run the simulation by executing:

python simulate_dimer.py
What the script does:
Mie Coefficients Calculation: The script computes Mie scattering coefficients for the two spheres and calculates their polarizabilities.

Polarizability Calculation: Using the Mie coefficients, it computes the Mie-derived electric dipole polarizability for each sphere.

Coupled-Dipole Model: Solves for dipole moments of the spheres under incident fields using the coupled-dipole model.

Fano Fit: Fits a Fano lineshape to the extinction spectra and reports the fit parameters.

Full-Wave Verification: Compares the coupled-dipole model’s extinction spectrum with a full-wave simulation for a specific gap size and polarization.

Expected Output:
Figures: The script generates various plots, including Mie coefficients, electric dipole polarizability, normalized extinction spectra, and Fano fit results.

Data: Mie coefficients and polarizabilities for each sphere are saved in CSV format.

Report: The results are compiled into a LaTeX report (report.pdf), which includes figures, equations, and analysis.

Simulation Parameters
Spheres:

Sphere 1: Radius = 90 nm, Refractive Index = 3.47

Sphere 2: Radius = 65 nm, Refractive Index = 2.41

Gap Sweep: 10, 20, 40, 60, 80 nm

Wavelength Range: 500 nm to 1000 nm (with at least 400 data points)

Polarization Directions:

Polarization A: 
𝐸
0
∥
𝑥
^
E 
0
​
 ∥ 
x
^
  (Longitudinal)

Polarization B: 
𝐸
0
∥
𝑦
^
E 
0
​
 ∥ 
y
^
​
  (Transverse)

Track 2 (with Crystalline Si Dispersion):
Use refractive index data for crystalline Si (Si) from refractiveindex.info (or another trusted source).

Interpolate the refractive index data for both spheres based on the simulation wavelength grid.

Model dispersion in the material by using the complex refractive index:

𝑛
~
(
𝜆
)
=
𝑛
(
𝜆
)
+
𝑖
𝑘
(
𝜆
)
n
~
 (λ)=n(λ)+ik(λ)
Key Equations
1. Mie Coefficient Calculation (a1)
The Mie coefficient 
𝑎
1
(
𝜆
)
a 
1
​
 (λ) is computed from the extinction efficiency 
𝑄
ext
Q 
ext
​
  using:

𝑎
1
(
𝜆
)
=
𝑄
ext
⋅
2
𝜋
𝑘
a 
1
​
 (λ)=Q 
ext
​
 ⋅ 
k
2π
​
 
Where 
𝑘
=
2
𝜋
𝜆
k= 
λ
2π
​
  is the wave number.

2. Mie-Derived Electric Dipole Polarizability
The electric dipole polarizability 
𝛼
(
𝜆
)
α(λ) is calculated from the Mie coefficient 
𝑎
1
(
𝜆
)
a 
1
​
 (λ) using:

𝛼
(
𝜆
)
=
6
𝜋
𝑖
𝑘
3
⋅
𝑎
1
(
𝜆
)
α(λ)= 
k 
3
 
6πi
​
 ⋅a 
1
​
 (λ)
Where 
𝑘
=
2
𝜋
𝜆
k= 
λ
2π
​
  is the wave number.

3. Coupled-Dipole Equations
The dipole moment 
𝑑
𝑖
d 
i
​
  of each sphere is determined by solving the coupled-dipole equations:

𝑑
𝑖
=
𝛼
𝑖
(
𝐸
inc
(
𝑟
𝑖
)
+
∑
𝑗
≠
𝑖
𝐺
(
𝑟
𝑖
−
𝑟
𝑗
)
𝑑
𝑗
)
d 
i
​
 =α 
i
​
  
​
 E 
inc
​
 (r 
i
​
 )+ 
j

=i
∑
​
 G(r 
i
​
 −r 
j
​
 )d 
j
​
  
​
 
Where 
𝐺
(
𝑟
𝑖
−
𝑟
𝑗
)
G(r 
i
​
 −r 
j
​
 ) is the Green function describing the interaction between the two particles, and 
𝛼
𝑖
α 
i
​
  is the polarizability of the sphere.

4. Dyadic Green Function
The dyadic Green function 
𝐺
(
𝑅
)
G(R) for the interaction between two dipoles is computed using:

𝐺
(
𝑅
)
=
𝑒
𝑖
𝑘
𝑅
4
𝜋
𝑅
3
[
(
𝑘
2
𝑅
2
)
(
𝐼
−
𝑅
^
𝑅
^
)
+
(
1
−
𝑖
𝑘
𝑅
)
(
3
𝑅
^
𝑅
^
−
𝐼
)
]
G(R)= 
4πR 
3
 
e 
ikR
 
​
 [(k 
2
 R 
2
 )(I− 
R
^
  
R
^
 )+(1−ikR)(3 
R
^
  
R
^
 −I)]
Where 
𝑅
^
=
𝑅
𝑅
R
^
 = 
R
R
​
  is the unit vector along the direction of separation between the two spheres, and 
𝑘
=
2
𝜋
𝜆
k= 
λ
2π
​
  is the wave number.

5. Fano Fit
The Fano lineshape is fitted to the observed extinction spectrum:

𝑦
(
𝜆
)
=
𝑦
0
+
𝐴
(
𝑞
+
𝜖
1
+
𝜖
2
)
y(λ)=y 
0
​
 +A( 
1+ϵ 
2
 
q+ϵ
​
 )
Where 
𝜖
=
𝜆
−
𝜆
0
Γ
ϵ= 
Γ
λ−λ 
0
​
 
​
 , and:

𝑦
0
y 
0
​
  is the baseline offset,

𝐴
A is the amplitude of the peak,

𝜆
0
λ 
0
​
  is the resonance wavelength,

Γ
Γ is the width of the resonance,

𝑞
q is the asymmetry factor.

Full-Wave Verification
The full-wave verification compares the coupled-dipole model’s extinction spectrum with that of a full-wave solver (e.g., Meep, COMSOL, or CST Studio Suite). The comparison will include:

Peak positions.

Qualitative asymmetry trends (Fano fit results).

References
miepython documentation. https://miepython.readthedocs.io/.

Pymiescatt documentation. https://pymiescatt.readthedocs.io/.

A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and S. G. Johnson. "Meep: A flexible free-software package for electromagnetic simulations by the FDTD method." Computer Physics Communications, 181(3):687–702, 2010.

COMSOL. Wave Optics Module User’s Guide, 2025.

Refractiveindex.info. "Refractive index of crystalline silicon." Accessed January 2026.


---

This is the entire `README.md` content in one markdown code block for you to copy and paste directly into your project folder.
