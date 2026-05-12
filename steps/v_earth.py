import numpy as np

AU      = 1.496e11    # 1 astronomical unit in meters
M_SUN   = 1.989e30    # solar mass in kg
V_EARTH = 29_780      # Earth orbital velocity in m/s

pos_earth = np.array([AU, 0.0])
vel_earth = np.array([0.0, V_EARTH])

print("Earth position:", pos_earth)
print("Distance to Sun:", np.linalg.norm(pos_earth), "m")
print("Orbital velocity:", np.linalg.norm(vel_earth), "m/s")

# Expected output:
# Earth position: [1.496e+11 0.000e+00]
# Distance to Sun: 149600000000.0 m
# Orbital velocity: 29780.0 m/s
