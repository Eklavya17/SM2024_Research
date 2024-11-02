import sympy as sp

# Define symbols
G = sp.symbols('G')
M = sp.symbols('M')
R = sp.symbols('R')
n = sp.Rational(3, 2)  # Polytopic index

# Define expressions for thermal, gravitational, and chemical energies
thermal_energy = sp.Rational(3, 2) * G * M**2 / R
gravitational_energy = -sp.Rational(3, 5) * G * M**2 / R
chemical_energy = -sp.Rational(3, 2) * G * M**2 / R

total_energy = thermal_energy + gravitational_energy + chemical_energy
total_energy
import sympy as sp

# Define symbols
R = sp.symbols('R')
L = sp.symbols('L')
M = sp.symbols('M')
T_eff = sp.symbols('T_eff')

# Derive the evolution equation for the star's radius assuming Hayashi track
# Equate luminosity to energy generated by gravitational contraction
radius_evolution_eq = sp.Eq(L, 4 * sp.pi * R**2 * sp.symbols('sigma') * T_eff**4)

radius_evolution_eq
import sympy as sp

# Define symbols and parameters
G = sp.symbols('G')
M = sp.symbols('M')
R = sp.symbols('R')
n = sp.Rational(3, 2)  # Polytropic index
T_eff = sp.symbols('T_eff')
sigma = sp.symbols('sigma')

# Define initial conditions and parameters for numerical integration
initial_mass = 0.1  # Initial mass in solar masses
final_mass = 1.0  # Final mass in solar masses

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation for radius evolution
# Assuming a simple linear relation for radius evolution
# dR/dM = k * R

def radius_evolution(R, M, k):
    return k * R

# Initial conditions
R_initial = 1.0  # Initial radius
k = 0.1  # Constant for radius evolution

# Mass values for integration
mass_values = np.linspace(0.1, 1.0, 100)

# Numerical integration of the radius evolution equation
radius_values = odeint(radius_evolution, R_initial, mass_values, args=(k,))

# Plotting the radius evolution
plt.figure(figsize=(12, 6))
plt.plot(mass_values, radius_values, label='Radius Evolution')
plt.xlabel('Mass ($M_{igodot}$)')
plt.ylabel('Radius')
plt.title('Protostellar Evolution: Radius vs. Mass')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation for radius evolution
# Assuming a simple linear relation for radius evolution
# dR/dM = k * R

def radius_evolution(R, M, k):
    return k * R

# Initial conditions
R_initial = 1.0  # Initial radius
k = 0.1  # Constant for radius evolution

# Mass values for integration
mass_values = np.linspace(0.1, 1.0, 100)

# Numerical integration of the radius evolution equation
radius_values = odeint(radius_evolution, R_initial, mass_values, args=(k,))

# Plotting the radius evolution
plt.figure(figsize=(12, 6))
plt.plot(mass_values, radius_values, label='Radius Evolution')
plt.xlabel('Mass ($M_{igodot}$)')
plt.ylabel('Radius')
plt.title('Protostellar Evolution: Radius vs. Mass')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation for radius evolution
# Assuming a simple linear relation for radius evolution
# dR/dM = k * R

def radius_evolution(R, M, k):
    return k * R

# Initial conditions
R_initial = 1.0  # Initial radius
k = 0.1  # Constant for radius evolution

# Mass values for integration
mass_values = np.linspace(0.1, 1.0, 100)

# Numerical integration of the radius evolution equation
radius_values = odeint(radius_evolution, R_initial, mass_values, args=(k,))

# Plotting the radius evolution
plt.figure(figsize=(12, 6))
plt.plot(mass_values, radius_values, label='Radius Evolution')
plt.xlabel('Mass ($M_{\odot}$)')
plt.ylabel('Radius')
plt.title('Protostellar Evolution: Radius vs. Mass')
plt.legend()
plt.grid(True)
plt.show()