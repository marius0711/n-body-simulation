import numpy as np

G     = 6.674e-11
M_SUN = 1.989e30
AU    = 1.496e11

pos = np.array([AU,     0.0])
vel = np.array([0.0, 29_780])

dt = 86400   # 1 day

r_vec = np.array([0.0, 0.0]) - pos
r_mag = np.linalg.norm(r_vec)
a = G * M_SUN / r_mag**3 * r_vec

pos_new = pos + vel * dt
vel_new = vel + a   * dt

print("Position before:", pos)
print("Position after:", pos_new)
print("Displacement:", pos_new - pos, "m")
print("New velocity:", vel_new, "m/s")
