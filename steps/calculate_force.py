import numpy as np

G       = 6.674e-11
M_sun   = 1.989e30
AU      = 1.496e11

pos_sonne = np.array([0.0, 0.0])
pos_erde  = np.array([AU,  0.0])

r_vec = pos_sonne - pos_erde          # Vektor von Erde zur Sonne
r_mag = np.linalg.norm(r_vec)         # Abstand (Betrag)

a = G * M_sun / r_mag**3 * r_vec      # Beschleunigungsvektor

print("r_vec:", r_vec)
print("Abstand:", r_mag, "m")
print("Beschleunigung:", a, "m/s²")
print("Betrag:", np.linalg.norm(a), "m/s²")