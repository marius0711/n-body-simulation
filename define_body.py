import numpy as np
AU = 1.496e11
pos_erde = np.array([AU, 0.0])
vel_erde = np.array([0.0, 29_780])

print("Position Erde:", pos_erde)
print("Abstand zur Sonne:", np.linalg.norm(pos_erde), "m")
print("Geschwindigkeit:", np.linalg.norm(vel_erde), "m/s")