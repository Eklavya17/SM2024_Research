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