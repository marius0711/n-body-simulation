import numpy as np
import matplotlib.pyplot as plt
import datetime

G     = 6.674e-11
M_sun = 1.989e30
AU    = 1.496e11

def make_bodies():
    return [
        {"name": "Sonne", "mass": M_sun,    "pos": np.array([0.0, 0.0]), "vel": np.array([0.0, 0.0]),    "color": "yellow"},
        {"name": "Erde",  "mass": 5.972e24, "pos": np.array([AU,  0.0]), "vel": np.array([0.0, 29_780.0]), "color": "deepskyblue"},
    ]

def compute_accels(bodies):
    accels = [np.zeros(2) for _ in bodies]
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i == j:
                continue
            r_vec  = bodies[j]["pos"] - bodies[i]["pos"]
            r_mag  = np.linalg.norm(r_vec)
            r_safe = np.sqrt(r_mag**2 + 1e8**2)
            accels[i] += G * bodies[j]["mass"] / r_safe**3 * r_vec
    return accels

def compute_energy(bodies):
    # Kinetische Energie
    E_kin = sum(0.5 * b["mass"] * np.dot(b["vel"], b["vel"]) for b in bodies)
    # Potenzielle Energie (alle Paare)
    E_pot = 0.0
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            r = np.linalg.norm(bodies[j]["pos"] - bodies[i]["pos"])
            E_pot -= G * bodies[i]["mass"] * bodies[j]["mass"] / r
    return E_kin + E_pot

def euler_step(bodies, dt):
    accels = compute_accels(bodies)
    for i, b in enumerate(bodies):
        b["pos"] = b["pos"] + b["vel"] * dt
        b["vel"] = b["vel"] + accels[i] * dt

def rk4_step(bodies, dt):
    # Zustand sichern
    pos0 = [b["pos"].copy() for b in bodies]
    vel0 = [b["vel"].copy() for b in bodies]

    # k1 — Steigung am Anfang
    a1 = compute_accels(bodies)
    k1_pos = [b["vel"].copy() for b in bodies]
    k1_vel = [a.copy() for a in a1]

    # k2 — Mitte, mit k1 verschoben
    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k1_pos[i] * dt/2
        b["vel"] = vel0[i] + k1_vel[i] * dt/2
    a2 = compute_accels(bodies)
    k2_pos = [b["vel"].copy() for b in bodies]
    k2_vel = [a.copy() for a in a2]

    # k3 — Mitte, mit k2 verschoben
    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k2_pos[i] * dt/2
        b["vel"] = vel0[i] + k2_vel[i] * dt/2
    a3 = compute_accels(bodies)
    k3_pos = [b["vel"].copy() for b in bodies]
    k3_vel = [a.copy() for a in a3]

    # k4 — Ende, mit k3 verschoben
    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k3_pos[i] * dt
        b["vel"] = vel0[i] + k3_vel[i] * dt
    a4 = compute_accels(bodies)
    k4_pos = [b["vel"].copy() for b in bodies]
    k4_vel = [a.copy() for a in a4]

    # Gewichtetes Mittel — RK4-Formel
    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + (k1_pos[i] + 2*k2_pos[i] + 2*k3_pos[i] + k4_pos[i]) * dt/6
        b["vel"] = vel0[i] + (k1_vel[i] + 2*k2_vel[i] + 2*k3_vel[i] + k4_vel[i]) * dt/6

def run(integrator, dt, steps):
    bodies = make_bodies()
    traj   = []
    energy = []
    for _ in range(steps):
        traj.append(bodies[1]["pos"].copy())
        energy.append(compute_energy(bodies))
        integrator(bodies, dt)
    return np.array(traj), np.array(energy)

# ── Simulation ────────────────────────────────
dt    = 86400       # 1 Tag
steps = 365 * 3     # 3 Jahre

print("Simuliere Euler...")
traj_e, energy_e = run(euler_step, dt, steps)

print("Simuliere RK4...")
traj_r, energy_r = run(rk4_step,  dt, steps)

# Energiefehler relativ zum Startwert
err_e = np.abs((energy_e - energy_e[0]) / energy_e[0])
err_r = np.abs((energy_r - energy_r[0]) / energy_r[0])

drift_e = np.linalg.norm(traj_e[-1] - traj_e[0])
drift_r = np.linalg.norm(traj_r[-1] - traj_r[0])

print(f"\nEuler  — Drift: {drift_e:.2e} m  |  Max. Energiefehler: {err_e.max():.2e}")
print(f"RK4    — Drift: {drift_r:.2e} m  |  Max. Energiefehler: {err_r.max():.2e}")

# ── Plot ──────────────────────────────────────
start_date = datetime.date(2024, 1, 1)
end_date   = start_date + datetime.timedelta(seconds=steps * dt)

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="black")
fig.suptitle(f"Euler vs. RK4  ·  dt={dt}s  ·  {start_date} → {end_date}",
             color="white", fontsize=13, fontweight="bold", y=0.98)

# Links: Euler-Bahn
ax = axes[0]
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_title("Euler — Erdbahn", color="#aaaaaa", fontsize=9)
ax.plot(traj_e[:, 0], traj_e[:, 1], color="tomato", lw=0.8, label="Euler")
ax.scatter(0, 0, color="yellow", s=80, zorder=5)
ax.scatter(traj_e[0,0],  traj_e[0,1],  color="tomato", s=60, zorder=5)
ax.scatter(traj_e[-1,0], traj_e[-1,1], color="tomato", s=40, zorder=5, marker="x")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

# Mitte: RK4-Bahn
ax = axes[1]
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_title("RK4 — Erdbahn", color="#aaaaaa", fontsize=9)
ax.plot(traj_r[:, 0], traj_r[:, 1], color="limegreen", lw=0.8, label="RK4")
ax.scatter(0, 0, color="yellow", s=80, zorder=5)
ax.scatter(traj_r[0,0],  traj_r[0,1],  color="limegreen", s=60, zorder=5)
ax.scatter(traj_r[-1,0], traj_r[-1,1], color="limegreen", s=40, zorder=5, marker="x")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

# Rechts: Energiefehler
ax = axes[2]
ax.set_facecolor("black")
ax.set_title("Energiefehler über Zeit", color="#aaaaaa", fontsize=9)
time_years = np.arange(steps) * dt / (365.25 * 86400)
ax.plot(time_years, err_e, color="tomato",    lw=1.2, label="Euler")
ax.plot(time_years, err_r, color="limegreen", lw=1.2, label="RK4")
ax.set_xlabel("Zeit [Jahre]", color="white")
ax.set_ylabel("Relativer Energiefehler", color="white")
ax.set_yscale("log")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
ax.grid(True, color="gray", alpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("orbit_rk4_compare.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gespeichert: orbit_rk4_compare.png")