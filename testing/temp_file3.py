import sympy as sp

# Define symbols
G = sp.symbols('G')
M, R, n, k, T, mu, m_p, m_e, h, c, k_B, m_H, m_D, L, L_acc = sp.symbols('M R n k T mu m_p m_e h c k_B m_H m_D L L_acc')

# Define equations for thermal, gravitational, and chemical energies
thermal_energy = (3/2) * (k * T) * M
gravitational_energy = -(3/5) * (G * M**2) / R
chemical_energy = (m_D - m_H) * c**2

total_energy = thermal_energy + gravitational_energy + chemical_energy

total_energy
import numpy as np
from scipy.integrate import odeint

# Constants
G = 6.67430e-11  # Gravitational constant
M_sun = 1.989e30  # Solar mass in kg
R_sun = 6.9634e8  # Solar radius in meters
T_c = 1e6  # Effective temperature in Kelvin
mu = 0.6  # Mean molecular weight
m_p = 1.6726219e-27  # Proton mass in kg

# Initial conditions
M_initial = 0.1 * M_sun  # Initial mass
R_initial = 0.1 * R_sun  # Initial radius

# Function for the rate of change of radius with respect to mass
# This function represents the derived evolution equation

def radius_evolution(R, M):
    return -G * M / (4 * np.pi * R**4) * T_c / (mu * m_p)

# Numerical integration
mass_values = np.linspace(M_initial, 1.0 * M_sun, 100)  # Mass values for integration
radius_values = odeint(radius_evolution, R_initial, mass_values)  # Numerical integration of radius evolution

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(mass_values / M_sun, radius_values / R_sun, label='Radius Evolution')
plt.xlabel('Mass (Solar Masses)')
plt.ylabel('Radius (Solar Radii)')
plt.title('Protostellar Evolution: Radius vs. Mass')
plt.legend()
plt.grid(True)
plt.show()

