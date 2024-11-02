import sympy as sp

# Define symbols
G = sp.symbols('G')
M, R, n, k, T, mu, m_p, m_e, h, c, k_B, sigma, L, M_dot = sp.symbols('M R n k T mu m_p m_e h c k_B sigma L M_dot')

# Define constants
G_val = 6.67430e-11  # Gravitational constant
k_B_val = 1.380649e-23  # Boltzmann constant
m_p_val = 1.6726219e-27  # Proton mass
m_e_val = 9.10938356e-31  # Electron mass
h_val = 6.62607015e-34  # Planck constant
c_val = 299792458  # Speed of light
sigma_val = 5.670374419e-8  # Stefan-Boltzmann constant

# Define the total energy expression
U = (3/2) * (G * M**2) / (R * (n - 1)) + (3/2) * k * T / (mu * m_p) * M + (3/2) * k_B * T / (m_e) * M + (3/2) * h * c / (sigma * R) * M + L * M_dot

U
import numpy as np
import matplotlib.pyplot as plt

# Define constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of the planet Earth in kg

# Create a function to calculate gravitational potential

def gravitational_potential(r):
    return -G * M / r

# Generate an array of distances from the planet's center
# Here, let's generate distances from 1000 km to 50000 km
r_values = np.linspace(1000, 50000, 100)

# Calculate the gravitational potential at each distance
potential_values = [gravitational_potential(r) for r in r_values]

# Plot the distance vs. potential
plt.figure(figsize=(10, 6))
plt.plot(r_values, potential_values, color='b', linestyle='-', marker='o')
plt.title('Orbital Potential of a Planet')
plt.xlabel('Distance from Planet Center (m)')
plt.ylabel('Gravitational Potential (Joules)')
plt.grid(True)
plt.show()