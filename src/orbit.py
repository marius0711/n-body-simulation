import datetime
import numpy as np
import matplotlib.pyplot as plt

G     = 6.674e-11
M_SUN = 1.989e30
AU    = 1.496e11

bodies = [
    {"name": "Sun",   "mass": M_SUN,    "pos": np.array([0.0, 0.0]), "vel": np.array([0.0, 0.0])},
    {"name": "Earth", "mass": 5.972e24, "pos": np.array([AU,  0.0]), "vel": np.array([0.0, 29_780.0])},
]

dt    = 86400   # 1 day
steps = 365     # 1 year

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

print(f"Drift start→end: {np.linalg.norm(trajectories[1][-1] - trajectories[1][0]):.2e} m")

# ── Plot ──────────────────────────────────────────────────────────────────────
start_date = datetime.date(2024, 1, 1)
end_date   = start_date + datetime.timedelta(seconds=steps * dt)

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")
ax.set_aspect("equal")
fig.suptitle("Earth orbit · Euler integration",
             color="white", fontsize=14, fontweight="bold", y=0.97)
ax.set_title(f"dt = {dt}s  |  {steps} steps  |  {start_date} → {end_date}",
             color="#aaaaaa", fontsize=9, pad=10)

colors = ["yellow", "deepskyblue"]
names  = ["Sun", "Earth"]

for traj, color, name in zip(trajectories, colors, names):
    ax.plot(traj[:, 0], traj[:, 1], color=color, lw=0.8, alpha=0.7, label=name)
    ax.scatter(traj[0, 0],  traj[0, 1],  color=color, s=80, zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=40, zorder=5, marker="x")

ax.annotate("Start", xy=(trajectories[1][0, 0], trajectories[1][0, 1]),
            color="white", fontsize=9, xytext=(10, 10), textcoords="offset points")
ax.annotate("End (drift!)", xy=(trajectories[1][-1, 0], trajectories[1][-1, 1]),
            color="orange", fontsize=9, xytext=(10, -18), textcoords="offset points")

ax.legend(facecolor="black", labelcolor="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("plots/orbit_euler.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/orbit_euler.png")
