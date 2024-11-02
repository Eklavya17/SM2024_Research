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