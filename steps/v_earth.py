import numpy as np

AU      = 1.496e11    # 1 Astronomische Einheit in Metern
M_sun   = 1.989e30    # Sonnenmasse in kg
v_earth = 29_780      # Erdbahngeschwindigkeit in m/s

pos_erde = np.array([AU, 0.0])
vel_erde = np.array([0.0, v_earth])

print("Position Erde:", pos_erde)
print("Abstand zur Sonne:", np.linalg.norm(pos_erde), "m")
print("Geschwindigkeit:", np.linalg.norm(vel_erde), "m/s")

# Erwartete Ausgabe:
# Position Erde: [1.496e+11 0.000e+00]
# Abstand zur Sonne: 149600000000.0 m
# Geschwindigkeit: 29780.0 m/s