import sympy as sp

# Define symbols
G = sp.symbols('G')
M, R, n, k, T, mu, m_p, m_e, h, c, k_B, m_H, m_D, L, E_th, E_grav, E_chem = sp.symbols('M R n k T mu m_p m_e h c k_B m_H m_D L E_th E_grav E_chem')

# Define the total energy expression
E_total = E_th + E_grav + E_chem

# Define the expressions for thermal, gravitational, and chemical energies
E_th = 3/2 * k * T * M
E_grav = - (3/5) * (G * M**2) / R
E_chem = (m_D - m_H) * c**2

E_total
import numpy as np
from scipy.integrate import odeint

# Define the function for numerical integration

def evolve_star(initial_conditions, parameters):
    def radius_evolution(e_total, R, T):
        # Define the differential equation for radius evolution
        dR_dt = 8 * np.pi * R * np.sqrt(e_total) / (4 * np.pi * R**2 * T**4)
        return dR_dt
    
    # Extract initial conditions
    M_initial, R_initial, T_initial = initial_conditions
    
    # Extract parameters
    G, sigma, n, k_B, m_p, m_e, h, c, u, m_H, m_D = parameters
    
    # Perform numerical integration using odeint
    M_values = np.linspace(M_initial, 1.0, 100)  # Integration up to M=1.0 Msun
    results = odeint(radius_evolution, R_initial, M_values, args=(T_initial,))
    
    return M_values, results

# Initial conditions
initial_conditions = [M_initial, R_initial, T_initial]

# Parameters
parameters = [G_value, sigma_value, n_value, k_B_value, m_p_value, m_e_value, h_value, c_value, u_value, m_H_value, m_D_value]

# Perform numerical integration
M_values, radius_evolution_results = evolve_star(initial_conditions, parameters)

M_values, radius_evolution_results
import numpy as np
from scipy.integrate import odeint

# Define the function for numerical integration

def evolve_star(initial_conditions, parameters):
    def radius_evolution(e_total, R, T):
        # Define the differential equation for radius evolution
        dR_dt = 8 * np.pi * R * np.sqrt(e_total) / (4 * np.pi * R**2 * T**4)
        return dR_dt
    
    # Extract initial conditions
    M_initial = initial_conditions[0]
    R_initial = initial_conditions[1]
    T_initial = initial_conditions[2]
    
    # Extract parameters
    G, sigma, n, k_B, m_p, m_e, h, c, u, m_H, m_D = parameters
    
    # Perform numerical integration using odeint
    M_values = np.linspace(M_initial, 1.0, 100)  # Integration up to M=1.0 Msun
    results = odeint(radius_evolution, R_initial, M_values, args=(T_initial,))
    
    return M_values, results

# Initial conditions
initial_conditions = [0.1, 1.0, 10000]  # Example initial conditions

# Parameters
parameters = [6.6743e-11, 5.670374419e-08, 3.0, 1.380649e-23, 1.6726219e-27, 9.10938356e-31, 6.62607015e-34, 299792458, 1.6605390666e-27, 1.6735575e-27, 3.34358348e-27]  # Example parameters

# Perform numerical integration
M_values, radius_evolution_results = evolve_star(initial_conditions, parameters)

M_values, radius_evolution_results
import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(M_values, radius_evolution_results, label='Radius Evolution')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius')
plt.title('Protostar Evolution: Radius vs. Mass')
plt.legend()
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

# Generate a sample plot for luminosity (example data)
M_values = np.linspace(0, 1, 100)
L_values = np.linspace(0, 10, 100)

# Plot the luminosity against mass
plt.figure(figsize=(12, 6))
plt.plot(M_values, L_values, label='Luminosity')
plt.xlabel('Mass (Msun)')
plt.ylabel('Luminosity')
plt.title('Protostar Evolution: Luminosity vs. Mass')
plt.legend()
plt.grid(True)
plt.show()



