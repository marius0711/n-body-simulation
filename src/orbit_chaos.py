import numpy as np
import matplotlib.pyplot as plt
import datetime

G     = 6.674e-11
M_sun = 1.989e30
AU    = 1.496e11

# Kleine Störung — 1000 km auf einer Distanz von 150 Mio km
delta = 1e6   # 1000 km in Metern

def make_bodies(offset):
    return [
        {
            "name":  "Sonne",
            "mass":  M_sun,
            "pos":   np.array([0.0, 0.0]),
            "vel":   np.array([0.0, 0.0]),
        },
        {
            "name":  "Erde",
            "mass":  5.972e24,
            "pos":   np.array([AU + offset, 0.0]),  # ← minimale Abweichung
            "vel":   np.array([0.0, 29_780.0]),
        },
        {
            "name":  "Jupiter",
            "mass":  1.898e27,
            "pos":   np.array([5.2 * AU, 0.0]),
            "vel":   np.array([0.0, 13_070.0]),
        },
    ]

def run(bodies, dt, steps):
    traj = []
    for _ in range(steps):
        traj.append(bodies[1]["pos"].copy())   # nur Erdposition speichern
        accels = [np.zeros(2) for _ in bodies]
        for i in range(len(bodies)):
            for j in range(len(bodies)):
                if i == j:
                    continue
                r_vec  = bodies[j]["pos"] - bodies[i]["pos"]
                r_mag  = np.linalg.norm(r_vec)
                r_safe = np.sqrt(r_mag**2 + 1e8**2)
                accels[i] += G * bodies[j]["mass"] / r_safe**3 * r_vec
        for i, body in enumerate(bodies):
            body["pos"] = body["pos"] + body["vel"] * dt
            body["vel"] = body["vel"] + accels[i]   * dt
    return np.array(traj)

dt    = 86400
steps = 365 * 10   # 10 Jahre

# Drei Simulationen: original, +1000 km, +2000 km
traj_0 = run(make_bodies(0),       dt, steps)
traj_1 = run(make_bodies(delta),   dt, steps)
traj_2 = run(make_bodies(delta*2), dt, steps)

# Divergenz berechnen
div_1 = np.array([np.linalg.norm(traj_0[i] - traj_1[i]) for i in range(steps)])
div_2 = np.array([np.linalg.norm(traj_0[i] - traj_2[i]) for i in range(steps)])

start_date = datetime.date(2024, 1, 1)
end_date   = start_date + datetime.timedelta(seconds=steps * dt)

# ── Plot: 2 Subplots ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
fig.suptitle("Chaos-Experiment · 3-Körper-Problem",
             color="white", fontsize=14, fontweight="bold", y=0.97)

# Links: Bahnen
ax1.set_facecolor("black")
ax1.set_aspect("equal")
ax1.set_title(f"Erdbahnen · Δpos = 0 / 1000 / 2000 km  |  {start_date} → {end_date}",
              color="#aaaaaa", fontsize=8, pad=8)
ax1.plot(traj_0[:, 0], traj_0[:, 1], color="deepskyblue", lw=0.6, alpha=0.9, label="Erde original")
ax1.plot(traj_1[:, 0], traj_1[:, 1], color="lime",        lw=0.6, alpha=0.7, label="Erde +1000 km")
ax1.plot(traj_2[:, 0], traj_2[:, 1], color="red",         lw=0.6, alpha=0.7, label="Erde +2000 km")
ax1.scatter(0, 0, color="yellow", s=100, zorder=5, label="Sonne")
ax1.legend(facecolor="black", labelcolor="white", fontsize=8)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("gray")

# Rechts: Divergenz über Zeit
ax2.set_facecolor("black")
ax2.set_title("Divergenz der Bahnen über Zeit",
              color="#aaaaaa", fontsize=8, pad=8)
time_years = np.arange(steps) * dt / (365.25 * 86400)
ax2.plot(time_years, div_1 / 1e9, color="lime", lw=1.2, label="Δ original vs +1000 km")
ax2.plot(time_years, div_2 / 1e9, color="red",  lw=1.2, label="Δ original vs +2000 km")
ax2.set_xlabel("Zeit [Jahre]", color="white")
ax2.set_ylabel("Abstand [10⁶ km]", color="white")
ax2.legend(facecolor="black", labelcolor="white", fontsize=8)
ax2.tick_params(colors="white")
ax2.grid(True, color="gray", alpha=0.2)
for spine in ax2.spines.values():
    spine.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("orbit_chaos.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gespeichert: orbit_chaos.png")