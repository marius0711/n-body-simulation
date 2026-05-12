"""
Integrator comparison: Euler vs. Leapfrog vs. RK4
===================================================
Key question: which integrator conserves energy best over long timescales?

- Euler       O(dt)   — energy drifts monotonically (orbit spirals)
- Leapfrog    O(dt^2) — symplectic: energy error bounded, never drifts
- RK4         O(dt^4) — most accurate per step, but not symplectic

Over short timescales RK4 wins on precision. Over centuries, Leapfrog wins
on stability — it conserves a shadow Hamiltonian exactly. This is why
symplectic integrators dominate long-duration astrophysics simulations.
"""

import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from nbody import (
    sun_earth, compute_energy,
    euler_step, leapfrog_step, rk4_step,
    AU,
)

# ── Simulation settings ───────────────────────────────────────────────────────
DT    = 86400       # 1 day
YEARS = 10
STEPS = int(YEARS * 365.25)


def run(integrator, dt, steps):
    bodies = sun_earth()
    traj   = np.empty((steps, 2))
    energy = np.empty(steps)
    for k in range(steps):
        traj[k]   = bodies[1]["pos"]
        energy[k] = compute_energy(bodies)
        integrator(bodies, dt)
    return traj, energy


print(f"Running {YEARS}-year comparison (dt = {DT}s = 1 day)...")

traj_e, energy_e = run(euler_step,    DT, STEPS)
traj_l, energy_l = run(leapfrog_step, DT, STEPS)
traj_r, energy_r = run(rk4_step,      DT, STEPS)

# Relative energy error — how much does total energy deviate from initial?
rel_e = np.abs((energy_e - energy_e[0]) / energy_e[0])
rel_l = np.abs((energy_l - energy_l[0]) / energy_l[0])
rel_r = np.abs((energy_r - energy_r[0]) / energy_r[0])

drift_e = np.linalg.norm(traj_e[-1] - traj_e[0])
drift_l = np.linalg.norm(traj_l[-1] - traj_l[0])
drift_r = np.linalg.norm(traj_r[-1] - traj_r[0])

print(f"\n{'Integrator':<12} {'Drift [m]':>14} {'Max ΔE/E':>12}")
print("-" * 42)
print(f"{'Euler':<12} {drift_e:>14.2e} {rel_e.max():>12.2e}")
print(f"{'Leapfrog':<12} {drift_l:>14.2e} {rel_l.max():>12.2e}")
print(f"{'RK4':<12} {drift_r:>14.2e} {rel_r.max():>12.2e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
start = datetime.date(2024, 1, 1)
end   = start + datetime.timedelta(days=STEPS)
time  = np.arange(STEPS) * DT / (365.25 * 86400)

fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="black")
fig.suptitle(f"Euler vs. Leapfrog vs. RK4  ·  {YEARS}-year Earth orbit  ·  dt = 1 day",
             color="white", fontsize=13, fontweight="bold", y=0.98)

PALETTE = {"Euler": "tomato", "Leapfrog": "gold", "RK4": "limegreen"}

# Left: orbit trajectories
ax = axes[0]
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_title("Earth orbit", color="#aaaaaa", fontsize=9)
ax.scatter(0, 0, color="yellow", s=100, zorder=6, label="Sun")
for traj, name in [(traj_e, "Euler"), (traj_l, "Leapfrog"), (traj_r, "RK4")]:
    ax.plot(traj[:, 0], traj[:, 1], color=PALETTE[name], lw=0.7, alpha=0.8, label=name)
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

# Middle: energy error over time (log scale)
ax = axes[1]
ax.set_facecolor("black")
ax.set_title("Relative energy error  |ΔE/E₀|", color="#aaaaaa", fontsize=9)
ax.plot(time, rel_e, color=PALETTE["Euler"],    lw=1.2, label="Euler")
ax.plot(time, rel_l, color=PALETTE["Leapfrog"], lw=1.2, label="Leapfrog")
ax.plot(time, rel_r, color=PALETTE["RK4"],      lw=1.2, label="RK4")
ax.set_xlabel("Time [years]", color="white")
ax.set_ylabel("|ΔE / E₀|", color="white")
ax.set_yscale("log")
ax.legend(facecolor="black", labelcolor="white", fontsize=8)
ax.tick_params(colors="white")
ax.grid(True, color="gray", alpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor("gray")

# Right: final orbit drift bar chart
ax = axes[2]
ax.set_facecolor("black")
ax.set_title(f"Orbital drift after {YEARS} years", color="#aaaaaa", fontsize=9)
names  = ["Euler", "Leapfrog", "RK4"]
drifts = [drift_e, drift_l, drift_r]
bars   = ax.bar(names, [d / AU for d in drifts],
                color=[PALETTE[n] for n in names], alpha=0.85, width=0.5)
ax.set_ylabel("Drift [AU]", color="white")
for bar, d in zip(bars, drifts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{d/AU:.3f} AU", ha="center", color="white", fontsize=8)
ax.tick_params(colors="white")
ax.set_facecolor("black")
for sp in ax.spines.values(): sp.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("plots/orbit_compare.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/orbit_compare.png")
