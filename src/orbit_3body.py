import datetime
import numpy as np
import matplotlib.pyplot as plt

G     = 6.674e-11
M_SUN = 1.989e30
AU    = 1.496e11

bodies = [
    {"name": "Sun",     "mass": M_SUN,    "pos": np.array([0.0, 0.0]),      "vel": np.array([0.0, 0.0]),      "color": "yellow"},
    {"name": "Earth",   "mass": 5.972e24, "pos": np.array([AU,  0.0]),      "vel": np.array([0.0, 29_780.0]), "color": "deepskyblue"},
    {"name": "Jupiter", "mass": 1.898e27, "pos": np.array([5.2*AU, 0.0]),   "vel": np.array([0.0, 13_070.0]), "color": "orange"},
]

dt    = 86400       # 1 day
steps = 365 * 5     # 5 years

trajectories = [[] for _ in bodies]

for step in range(steps):
    for i, body in enumerate(bodies):
        trajectories[i].append(body["pos"].copy())

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

trajectories = [np.array(t) for t in trajectories]

# ── Plot ──────────────────────────────────────────────────────────────────────
start_date = datetime.date(2024, 1, 1)
end_date   = start_date + datetime.timedelta(seconds=steps * dt)

fig, ax = plt.subplots(figsize=(9, 9), facecolor="black")
ax.set_facecolor("black")
ax.set_aspect("equal")
fig.suptitle("3-Body · Sun + Earth + Jupiter",
             color="white", fontsize=14, fontweight="bold", y=0.97)
ax.set_title(f"dt = {dt}s  |  {steps} steps  |  {start_date} → {end_date}",
             color="#aaaaaa", fontsize=9, pad=10)

for body, traj in zip(bodies, trajectories):
    ax.plot(traj[:, 0], traj[:, 1], color=body["color"], lw=0.8, alpha=0.8, label=body["name"])
    ax.scatter(traj[0,  0], traj[0,  1], color=body["color"], s=80, zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], color=body["color"], s=40, zorder=5, marker="x")

ax.legend(facecolor="black", labelcolor="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("plots/orbit_3body.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/orbit_3body.png")
