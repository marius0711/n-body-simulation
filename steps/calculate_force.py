import numpy as np

G     = 6.674e-11
M_SUN = 1.989e30
AU    = 1.496e11

pos_sun   = np.array([0.0, 0.0])
pos_earth = np.array([AU,  0.0])

r_vec = pos_sun - pos_earth          # vector from Earth toward Sun
r_mag = np.linalg.norm(r_vec)        # distance (magnitude)

a = G * M_SUN / r_mag**3 * r_vec     # gravitational acceleration vector

print("r_vec:", r_vec)
print("Distance:", r_mag, "m")
print("Acceleration:", a, "m/s²")
print("Magnitude:", np.linalg.norm(a), "m/s²")
