import numpy as np

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.98847e30  # Mass of the sun (kg)
R_sun = 6.95700e8  # Radius of the sun (m)

# Star properties (using solar units for simplicity)
M = 0.5 * M_sun  # Mass of the protostar, 0.5 solar masses
R = 2 * R_sun    # Radius of the protostar, 2 solar radii

# Gravitational Energy (E_grav)
n = 3/2  # Polytropic index for low-mass stars
E_grav = -3 / (5 - n) * (G * M**2 / R)
print(f"Gravitational Energy: {E_grav:.2e} J")

# Thermal Energy (E_th) from the virial theorem
E_th = -0.5 * E_grav
print(f"Thermal Energy: {E_th:.2e} J")

# Chemical Energy (E_chem)
# Assuming 0.0156% of the hydrogen mass is deuterium
m_H = 1.6735575e-27  # Mass of hydrogen atom (kg)
E_per_deuterium = 1.44e-13  # Energy released per deuterium fusion (J)
deuterium_fraction = 0.000156
n_deuterium = deuterium_fraction * M / m_H  # Total number of deuterium nuclei
E_chem = n_deuterium * E_per_deuterium
print(f"Chemical Energy: {E_chem:.2e} J")

# Total Energy
E_total = E_th + E_grav + E_chem
print(f"Total Energy: {E_total:.2e} J")

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2 K^4)
M_sun = 1.98847e30  # kg
R_sun = 6.95700e8  # m
T_eff = 3000  # K (Constant effective temperature on Hayashi track)

# Differential equation for radius evolution
def dR_dM(M, R):
    return (G * M) / (8 * np.pi * R**2 * sigma * T_eff**4)

# Initial conditions
M0 = 0.1 * M_sun  # Start at 0.1 solar mass
R0 = 1 * R_sun    # Start at 1 solar radius

# Integration limits
M_final = 1.0 * M_sun  # Final mass: 1 solar mass

# Solve differential equation
result = solve_ivp(dR_dM, [M0, M_final], [R0], method='RK45', dense_output=True)

# Plot the results
masses = np.linspace(M0, M_final, 300)
radii = result.sol(masses)[0]

plt.figure(figsize=(10, 5))
plt.plot(masses / M_sun, radii / R_sun)
plt.xlabel('Mass (Solar Masses)')
plt.ylabel('Radius (Solar Radii)')
plt.title('Radius Evolution of a Protostar on the Hayashi Track')
plt.grid(True)
plt.show()

# Constants for massive stars
T_eff_massive = 10000  # Higher effective temperature for massive stars

# Modified differential equation for radius evolution of massive protostars
def dR_dM_massive(M, R):
    # Adjust the formula to reflect higher luminosity sensitivity
    return (G * M**3) / (8 * np.pi * R**2 * sigma * T_eff_massive**4)

# Initial conditions for a massive star
M0_massive = 10 * M_sun  # Start at 10 solar masses
R0_massive = 3 * R_sun   # Start at 3 solar radii

# Integration limits for a massive star
M_final_massive = 50 * M_sun  # Final mass: 50 solar masses

# Solve differential equation for massive stars
result_massive = solve_ivp(dR_dM_massive, [M0_massive, M_final_massive], [R0_massive], method='RK45', dense_output=True)

# Plot the results for massive stars
masses_massive = np.linspace(M0_massive, M_final_massive, 300)
radii_massive = result_massive.sol(masses_massive)[0]

plt.figure(figsize=(10, 5))
plt.plot(masses_massive / M_sun, radii_massive / R_sun)
plt.xlabel('Mass (Solar Masses)')
plt.ylabel('Radius (Solar Radii)')
plt.title('Radius Evolution of a Massive Protostar')
plt.grid(True)
plt.show()

