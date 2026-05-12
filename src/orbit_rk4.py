import datetime
import numpy as np
import matplotlib.pyplot as plt

G     = 6.674e-11
M_SUN = 1.989e30
AU    = 1.496e11


def make_bodies():
    return [
        {"name": "Sun",   "mass": M_SUN,    "pos": np.array([0.0, 0.0]), "vel": np.array([0.0, 0.0]),      "color": "yellow"},
        {"name": "Earth", "mass": 5.972e24, "pos": np.array([AU,  0.0]), "vel": np.array([0.0, 29_780.0]), "color": "deepskyblue"},
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
    E_kin = sum(0.5 * b["mass"] * np.dot(b["vel"], b["vel"]) for b in bodies)
    E_pot = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            r = np.linalg.norm(bodies[j]["pos"] - bodies[i]["pos"])
            E_pot -= G * bodies[i]["mass"] * bodies[j]["mass"] / r
    return E_kin + E_pot


def euler_step(bodies, dt):
    accels = compute_accels(bodies)
    for i, b in enumerate(bodies):
        b["pos"] = b["pos"] + b["vel"] * dt
        b["vel"] = b["vel"] + accels[i] * dt


def rk4_step(bodies, dt):
    pos0 = [b["pos"].copy() for b in bodies]
    vel0 = [b["vel"].copy() for b in bodies]

    a1  = compute_accels(bodies)
    k1p = [b["vel"].copy() for b in bodies]
    k1v = [a.copy() for a in a1]

    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k1p[i] * dt / 2
        b["vel"] = vel0[i] + k1v[i] * dt / 2
    a2  = compute_accels(bodies)
    k2p = [b["vel"].copy() for b in bodies]
    k2v = [a.copy() for a in a2]

    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k2p[i] * dt / 2
        b["vel"] = vel0[i] + k2v[i] * dt / 2
    a3  = compute_accels(bodies)
    k3p = [b["vel"].copy() for b in bodies]
    k3v = [a.copy() for a in a3]

    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + k3p[i] * dt
        b["vel"] = vel0[i] + k3v[i] * dt
    a4  = compute_accels(bodies)
    k4p = [b["vel"].copy() for b in bodies]
    k4v = [a.copy() for a in a4]

    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + (k1p[i] + 2*k2p[i] + 2*k3p[i] + k4p[i]) * dt / 6
        b["vel"] = vel0[i] + (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) * dt / 6


def run(integrator, dt, steps):
    bodies = make_bodies()
    traj   = []
    energy = []
    for _ in range(steps):
        traj.append(bodies[1]["pos"].copy())
        energy.append(compute_energy(bodies))
        integrator(bodies, dt)
    return np.array(traj), np.array(energy)


# ── Simulation ────────────────────────────────────────────────────────────────
dt    = 86400       # 1 day
steps = 365 * 3     # 3 years

print("Running Euler...")
traj_e, energy_e = run(euler_step, dt, steps)

print("Running RK4...")
traj_r, energy_r = run(rk4_step,  dt, steps)

err_e = np.abs((energy_e - energy_e[0]) / energy_e[0])
err_r = np.abs((energy_r - energy_r[0]) / energy_r[0])

drift_e = np.linalg.norm(traj_e[-1] - traj_e[0])
drift_r = np.linalg.norm(traj_r[-1] - traj_r[0])

print(f"\nEuler   — drift: {drift_e:.2e} m  |  max energy error: {err_e.max():.2e}")
print(f"RK4     — drift: {drift_r:.2e} m  |  max energy error: {err_r.max():.2e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
start_date = datetime.date(2024, 1, 1)
end_date   = start_date + datetime.timedelta(seconds=steps * dt)

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="black")
fig.suptitle(f"Euler vs. RK4  ·  dt={dt}s  ·  {start_date} → {end_date}",
             color="white", fontsize=13, fontweight="bold", y=0.98)

ax = axes[0]
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_title("Euler — Earth orbit", color="#aaaaaa", fontsize=9)
ax.plot(traj_e[:, 0], traj_e[:, 1], color="tomato", lw=0.8, label="Euler")
ax.scatter(0, 0, color="yellow", s=80, zorder=5)
ax.scatter(traj_e[0,  0], traj_e[0,  1], color="tomato", s=60, zorder=5)
ax.scatter(traj_e[-1, 0], traj_e[-1, 1], color="tomato", s=40, zorder=5, marker="x")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

ax = axes[1]
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_title("RK4 — Earth orbit", color="#aaaaaa", fontsize=9)
ax.plot(traj_r[:, 0], traj_r[:, 1], color="limegreen", lw=0.8, label="RK4")
ax.scatter(0, 0, color="yellow", s=80, zorder=5)
ax.scatter(traj_r[0,  0], traj_r[0,  1], color="limegreen", s=60, zorder=5)
ax.scatter(traj_r[-1, 0], traj_r[-1, 1], color="limegreen", s=40, zorder=5, marker="x")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

ax = axes[2]
ax.set_facecolor("black")
ax.set_title("Energy error over time", color="#aaaaaa", fontsize=9)
time_years = np.arange(steps) * dt / (365.25 * 86400)
ax.plot(time_years, err_e, color="tomato",    lw=1.2, label="Euler")
ax.plot(time_years, err_r, color="limegreen", lw=1.2, label="RK4")
ax.set_xlabel("Time [years]", color="white")
ax.set_ylabel("Relative energy error", color="white")
ax.set_yscale("log")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
ax.grid(True, color="gray", alpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("plots/orbit_rk4_compare.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/orbit_rk4_compare.png")
