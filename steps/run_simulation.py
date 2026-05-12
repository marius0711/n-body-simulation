import numpy as np

G     = 6.674e-11
M_sun = 1.989e30
AU    = 1.496e11

bodies = [
    {"name": "Sonne", "mass": M_sun,    "pos": np.array([0.0, 0.0]), "vel": np.array([0.0, 0.0])},
    {"name": "Erde",  "mass": 5.972e24, "pos": np.array([AU,  0.0]), "vel": np.array([0.0, 29_780.0])},
]

dt = 86400   # 1 Tag (war: 3600)
steps = 365   # 1 Jahr = 365 Tage

trajectories = [[] for _ in bodies]

for step in range(steps):
    # Positionen speichern
    for i, body in enumerate(bodies):
        trajectories[i].append(body["pos"].copy())

    # Beschleunigungen berechnen
    accels = [np.zeros(2) for _ in bodies]
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i == j:
                continue
            r_vec  = bodies[j]["pos"] - bodies[i]["pos"]
            r_mag  = np.linalg.norm(r_vec)
            r_safe = np.sqrt(r_mag**2 + 1e8**2)   # Softening
            accels[i] += G * bodies[j]["mass"] / r_safe**3 * r_vec

    # Euler-Schritt
    for i, body in enumerate(bodies):
        body["pos"] = body["pos"] + body["vel"] * dt
        body["vel"] = body["vel"] + accels[i]   * dt

# Trajektorien in Arrays umwandeln
trajectories = [np.array(t) for t in trajectories]

print(f"Simulation fertig: {steps} Schritte")
print(f"Startposition Erde: {trajectories[1][0]}")
print(f"Endposition Erde:   {trajectories[1][-1]}")
print(f"Abstand Start→Ende: {np.linalg.norm(trajectories[1][-1] - trajectories[1][0]):.2e} m")