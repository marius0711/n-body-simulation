import numpy as np

G     = 6.674e-11   # gravitational constant [m^3 kg^-1 s^-2]
AU    = 1.496e11    # astronomical unit [m]
M_SUN = 1.989e30    # solar mass [kg]

SOFTENING = 1e8     # 100,000 km — prevents singularity at close approach


def compute_accels(bodies):
    """O(N^2) pairwise gravitational acceleration for each body."""
    accels = [np.zeros(2) for _ in bodies]
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i == j:
                continue
            r_vec  = bodies[j]["pos"] - bodies[i]["pos"]
            r_safe = np.sqrt(np.dot(r_vec, r_vec) + SOFTENING**2)
            accels[i] += G * bodies[j]["mass"] / r_safe**3 * r_vec
    return accels


def compute_energy(bodies):
    """Total mechanical energy: kinetic + gravitational potential."""
    E_kin = sum(0.5 * b["mass"] * np.dot(b["vel"], b["vel"]) for b in bodies)
    E_pot = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            r = np.linalg.norm(bodies[j]["pos"] - bodies[i]["pos"])
            E_pot -= G * bodies[i]["mass"] * bodies[j]["mass"] / r
    return E_kin + E_pot


# ── Integrators ───────────────────────────────────────────────────────────────

def euler_step(bodies, dt):
    """Forward Euler — O(dt) accuracy. Simple but energy drifts over time."""
    accels = compute_accels(bodies)
    for i, b in enumerate(bodies):
        b["pos"] = b["pos"] + b["vel"] * dt
        b["vel"] = b["vel"] + accels[i] * dt


def leapfrog_step(bodies, dt):
    """Velocity Verlet / KDK Leapfrog — O(dt^2), symplectic.

    Symplectic means it conserves a shadow Hamiltonian exactly, so energy
    error stays bounded (oscillates) rather than drifting — unlike Euler or RK4.
    Standard integrator in astrophysics and molecular dynamics.
    """
    accels = compute_accels(bodies)
    for i, b in enumerate(bodies):               # half-kick
        b["vel"] = b["vel"] + 0.5 * accels[i] * dt
    for b in bodies:                             # full drift
        b["pos"] = b["pos"] + b["vel"] * dt
    accels = compute_accels(bodies)
    for i, b in enumerate(bodies):               # half-kick
        b["vel"] = b["vel"] + 0.5 * accels[i] * dt


def rk4_step(bodies, dt):
    """Runge-Kutta 4 — O(dt^4) accuracy. High precision per step, not symplectic."""
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


# ── Body factories ────────────────────────────────────────────────────────────

def sun_earth():
    return [
        {"name": "Sun",   "mass": M_SUN,    "pos": np.array([0.0, 0.0]), "vel": np.array([0.0, 0.0]),      "color": "yellow",      "size": 20},
        {"name": "Earth", "mass": 5.972e24, "pos": np.array([AU,  0.0]), "vel": np.array([0.0, 29_780.0]), "color": "deepskyblue", "size": 8},
    ]


def solar_system():
    return [
        {"name": "Sun",     "mass": M_SUN,    "pos": np.array([0.0, 0.0]),      "vel": np.array([0.0, 0.0]),      "color": "yellow",      "size": 20},
        {"name": "Earth",   "mass": 5.972e24, "pos": np.array([AU,  0.0]),      "vel": np.array([0.0, 29_780.0]), "color": "deepskyblue", "size": 8},
        {"name": "Mars",    "mass": 6.390e23, "pos": np.array([1.52*AU, 0.0]),  "vel": np.array([0.0, 24_077.0]), "color": "tomato",      "size": 6},
        {"name": "Jupiter", "mass": 1.898e27, "pos": np.array([5.2*AU,  0.0]),  "vel": np.array([0.0, 13_070.0]), "color": "orange",      "size": 14},
    ]
