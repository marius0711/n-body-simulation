import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G  = 6.674e-11
AU = 1.496e11

np.random.seed(42)
N = 8


def make_random_bodies():
    bodies = []
    colors = ["yellow", "deepskyblue", "tomato", "limegreen",
              "orange", "violet", "cyan", "hotpink"]

    for i in range(N):
        mass  = np.random.uniform(1e24, 2e30)
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(0.5*AU, 5*AU)
        pos   = np.array([np.cos(angle)*dist, np.sin(angle)*dist])

        # approximate circular orbit velocity around total system mass ~2e30
        v_circ = np.sqrt(G * 2e30 / dist) * np.random.uniform(0.7, 1.1)
        tang   = np.array([-np.sin(angle), np.cos(angle)])
        vel    = tang * v_circ

        bodies.append({
            "name":  f"B{i+1}",
            "mass":  mass,
            "pos":   pos,
            "vel":   vel,
            "color": colors[i],
            "size":  max(4, int(np.log10(mass) - 20)),
        })
    return bodies


def compute_accels(bodies):
    accels = [np.zeros(2) for _ in bodies]
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i == j:
                continue
            r_vec  = bodies[j]["pos"] - bodies[i]["pos"]
            r_mag  = np.linalg.norm(r_vec)
            r_safe = np.sqrt(r_mag**2 + 1e9**2)
            accels[i] += G * bodies[j]["mass"] / r_safe**3 * r_vec
    return accels


def rk4_step(bodies, dt):
    pos0 = [b["pos"].copy() for b in bodies]
    vel0 = [b["vel"].copy() for b in bodies]
    a1 = compute_accels(bodies)
    k1p = [b["vel"].copy() for b in bodies]; k1v = [a.copy() for a in a1]
    for i, b in enumerate(bodies): b["pos"] = pos0[i]+k1p[i]*dt/2; b["vel"] = vel0[i]+k1v[i]*dt/2
    a2 = compute_accels(bodies)
    k2p = [b["vel"].copy() for b in bodies]; k2v = [a.copy() for a in a2]
    for i, b in enumerate(bodies): b["pos"] = pos0[i]+k2p[i]*dt/2; b["vel"] = vel0[i]+k2v[i]*dt/2
    a3 = compute_accels(bodies)
    k3p = [b["vel"].copy() for b in bodies]; k3v = [a.copy() for a in a3]
    for i, b in enumerate(bodies): b["pos"] = pos0[i]+k3p[i]*dt; b["vel"] = vel0[i]+k3v[i]*dt
    a4 = compute_accels(bodies)
    k4p = [b["vel"].copy() for b in bodies]; k4v = [a.copy() for a in a4]
    for i, b in enumerate(bodies):
        b["pos"] = pos0[i] + (k1p[i]+2*k2p[i]+2*k3p[i]+k4p[i])*dt/6
        b["vel"] = vel0[i] + (k1v[i]+2*k2v[i]+2*k3v[i]+k4v[i])*dt/6


# ── Setup ─────────────────────────────────────────────────────────────────────
bodies  = make_random_bodies()
dt      = 86400 * 2    # 2 days per frame
TRAIL   = 150
trail_x = [[] for _ in bodies]
trail_y = [[] for _ in bodies]
day     = [0]

fig, ax = plt.subplots(figsize=(9, 9), facecolor="black")
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_xlim(-6*AU, 6*AU)
ax.set_ylim(-6*AU, 6*AU)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

title = fig.suptitle("", color="white", fontsize=12, fontweight="bold")

dots   = [ax.plot([], [], 'o', color=b["color"], ms=b["size"],  zorder=5)[0] for b in bodies]
trails = [ax.plot([], [], '-', color=b["color"], lw=0.7, alpha=0.4)[0] for b in bodies]
labels = [ax.text(0, 0, b["name"], color=b["color"], fontsize=7, zorder=6) for b in bodies]


def update(frame):
    rk4_step(bodies, dt)
    day[0] += 1
    for i, b in enumerate(bodies):
        x, y = b["pos"]
        trail_x[i].append(x)
        trail_y[i].append(y)
        if len(trail_x[i]) > TRAIL:
            trail_x[i].pop(0)
            trail_y[i].pop(0)
        dots[i].set_data([x], [y])
        trails[i].set_data(trail_x[i], trail_y[i])
        labels[i].set_position((x + 0.2*AU, y + 0.2*AU))
    years = day[0] * dt / (365.25 * 86400)
    title.set_text(f"N-body · RK4 · {N} bodies  —  Day {day[0]}  ({years:.2f} years)")
    return dots + trails + labels


ani = animation.FuncAnimation(fig, update, frames=None, interval=20, blit=True)
plt.tight_layout()
plt.show()
