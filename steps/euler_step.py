import numpy as np

G       = 6.674e-11
M_sun   = 1.989e30
AU      = 1.496e11

pos = np.array([AU,     0.0])
vel = np.array([0.0, 29_780])

dt = 86400   # 1 Tag (war: 3600)

# Beschleunigung berechnen (von oben)
r_vec = np.array([0.0, 0.0]) - pos
r_mag = np.linalg.norm(r_vec)
a = G * M_sun / r_mag**3 * r_vec

# Euler-Schritt
pos_neu = pos + vel * dt
vel_neu = vel + a   * dt

print("Position vorher:", pos)
print("Position nachher:", pos_neu)
print("Verschoben um:", pos_neu - pos, "m")
print("Neue Geschwindigkeit:", vel_neu, "m/s")