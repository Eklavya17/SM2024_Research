import sympy as sp

# Define symbols
G = sp.symbols('G')
M, R, n, k, T, mu, m_p, m_e, h, c, k_B, m_H, m_D, L, L_acc = sp.symbols('M R n k T mu m_p m_e h c k_B m_H m_D L L_acc')

# Define constants
G_val = 6.67430e-11  # Gravitational constant
k_B_val = 1.380649e-23  # Boltzmann constant
m_p_val = 1.6726219e-27  # Proton mass
m_e_val = 9.10938356e-31  # Electron mass
h_val = 6.62607015e-34  # Planck constant
m_H_val = 1.6735575e-27  # Hydrogen atom mass
m_D_val = 2.014102  # Deuterium atom mass
mu_val = 2.3  # Mean molecular weight
T_val = 2000  # Temperature in Kelvin
n_val = 3/2  # Polytropic index
L_acc_val = 1e-3  # Accretion luminosity

# Define expressions for thermal, gravitational, and chemical energies
thermal_energy = 3/2 * k * T * M
gravitational_energy = -3/5 * (G * M**2) / R
chemical_energy = 3/2 * k * T * M / (mu * m_p) * (1 - m_H / m_D)

total_energy = thermal_energy + gravitational_energy + chemical_energy

total_energy