import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nbody import (
    compute_accels, compute_energy,
    euler_step, leapfrog_step, rk4_step,
    sun_earth, G, AU, M_SUN,
)


def fresh():
    return sun_earth()


# ── Force laws ────────────────────────────────────────────────────────────────

def test_earth_pulled_toward_sun():
    bodies = fresh()
    accels = compute_accels(bodies)
    # Earth (index 1) must accelerate in the –x direction toward Sun at origin
    assert accels[1][0] < 0
    assert abs(accels[1][1]) < 1e-6   # no y-component for aligned bodies


def test_newtons_third_law():
    bodies = fresh()
    accels = compute_accels(bodies)
    F_sun   = accels[0] * bodies[0]["mass"]
    F_earth = accels[1] * bodies[1]["mass"]
    np.testing.assert_allclose(F_sun, -F_earth, rtol=1e-10)


def test_force_magnitude():
    bodies = fresh()
    accels = compute_accels(bodies)
    # Expected: a = G*M_sun / AU^2
    a_expected = G * M_SUN / AU**2
    assert abs(np.linalg.norm(accels[1]) - a_expected) / a_expected < 0.01


# ── Energy conservation ───────────────────────────────────────────────────────

def test_euler_energy_drifts():
    bodies = fresh()
    E0 = compute_energy(bodies)
    for _ in range(365):
        euler_step(bodies, 86400)
    E1 = compute_energy(bodies)
    # Euler is not symplectic — energy must show measurable drift after one year
    assert abs((E1 - E0) / E0) > 1e-5


def test_leapfrog_energy_bounded():
    bodies = fresh()
    E0 = compute_energy(bodies)
    for _ in range(365):
        leapfrog_step(bodies, 86400)
    E1 = compute_energy(bodies)
    assert abs((E1 - E0) / E0) < 1e-4, "Leapfrog energy error exceeds 0.01 %"


def test_rk4_energy_accurate():
    bodies = fresh()
    E0 = compute_energy(bodies)
    for _ in range(365):
        rk4_step(bodies, 86400)
    E1 = compute_energy(bodies)
    assert abs((E1 - E0) / E0) < 1e-7, "RK4 energy error exceeds 1e-7"


# ── Accuracy comparison ───────────────────────────────────────────────────────

def test_rk4_more_accurate_than_euler():
    """Over one year, RK4 orbit should drift less than Euler."""
    bodies_e = fresh()
    bodies_r = fresh()
    for _ in range(365):
        euler_step(bodies_e, 86400)
        rk4_step(bodies_r,   86400)
    # Both start at AU — compare final Earth position to start
    start = fresh()[1]["pos"]
    drift_euler = np.linalg.norm(bodies_e[1]["pos"] - start)
    drift_rk4   = np.linalg.norm(bodies_r[1]["pos"] - start)
    assert drift_rk4 < drift_euler


def test_leapfrog_more_stable_than_euler():
    """Over one year, Leapfrog orbit should drift less than Euler."""
    bodies_e = fresh()
    bodies_l = fresh()
    for _ in range(365):
        euler_step(bodies_e,    86400)
        leapfrog_step(bodies_l, 86400)
    start = fresh()[1]["pos"]
    drift_euler    = np.linalg.norm(bodies_e[1]["pos"] - start)
    drift_leapfrog = np.linalg.norm(bodies_l[1]["pos"] - start)
    assert drift_leapfrog < drift_euler
