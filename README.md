# N-Körper Simulation — Gravitationsdynamik & Numerische Integration

> Lernprojekt: Numerische Physik in Python — von der Newtonschen Gravitation bis zur animierten N-Körper-Simulation mit RK4.

Dieses Projekt simuliert die Gravitationswechselwirkung zwischen N Massen im zweidimensionalen Raum. Ausgangspunkt ist das klassische 2-Körper-Problem (Erde + Sonne), das schrittweise zu einem chaotischen N-Körper-System ausgebaut wird. Der Fokus liegt auf dem Verständnis numerischer Integrationsmethoden — insbesondere dem Sprung von Euler zu Runge-Kutta 4 — und der Frage, wie physikalische Erhaltungsgrößen als Qualitätsmaßstab für Algorithmen dienen können.

**Stack:** Python 3 · NumPy · Matplotlib — 100 % Open Source, keine weiteren Dependencies.

---

## Ergebnisse

| Integrator | Drift (3 Jahre, dt=1 Tag) | Relativer Energiefehler |
|---|---|---|
| Euler | `2.18e+11 m` | `30 %` |
| RK4 | `1.91e+09 m` | `0.0000002 %` |

RK4 ist **114× genauer in der Bahn** und **140 Millionen× genauer in der Energie** — bei gleichem `dt`.

---

## Projektstruktur

```
n-body-simulation/
├── README.md
├── .gitignore
├── src/
│   ├── orbit.py              → Euler-Integration, 2-Körper, dt-Experiment
│   ├── orbit_3body.py        → Sonne + Erde + Jupiter, 5 Jahre
│   ├── orbit_chaos.py        → Chaos-Experiment: Sensitivität auf Anfangsbedingungen
│   ├── orbit_rk4.py          → RK4 vs Euler: Bahn + Energiefehler-Vergleich
│   ├── orbit_animate.py      → Sonnensystem-Animation (FuncAnimation)
│   └── orbit_animate_n.py    → N zufällige Körper animiert mit Schweif & RK4
├── steps/                    → Lernschritte (Woche 1, einzelne Konzepte)
│   ├── v_earth.py
│   ├── calculate_force.py
│   ├── euler_step.py
│   └── run_simulation.py
└── plots/                    → Gespeicherte Plot-Outputs
    ├── orbit_euler.png
    ├── orbit_3body.png
    ├── orbit_chaos.png
    └── orbit_rk4_compare.png
```

---

## Physikalische Grundlagen

### Newtonsche Gravitation

```
F_vec = G * m1 * m2 / |r|³  *  r_vec
```

`G = 6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻²` — universelle Gravitationskonstante

### Bewegungsgleichungen (Zustandsraum)

```
dx/dt = v
dv/dt = a = Σ_{j≠i}  G * m_j / |r_ij|³  *  r_ij
```

### Energieerhaltung (Qualitätsmetrik)

```
E_kin = 0.5 * m * |v|²
E_pot = -G * m_i * m_j / |r_ij|     (pro Paar)
E_ges = E_kin + E_pot = const.
```

---

## Numerische Methoden

### Euler (Fehlerordnung O(dt))

```
x(t+dt) = x(t) + v(t) * dt
v(t+dt) = v(t) + a(t) * dt
```

Einfach, aber akkumuliert Energie — Bahnen spiralisieren bei großem `dt`.

### Runge-Kutta 4 (Fehlerordnung O(dt⁴))

```
k1 = f(t,       y)
k2 = f(t+dt/2,  y + k1*dt/2)
k3 = f(t+dt/2,  y + k2*dt/2)
k4 = f(t+dt,    y + k3*dt)

y(t+dt) = y(t) + (k1 + 2*k2 + 2*k3 + k4) * dt/6
```

4× teurer pro Schritt — aber dramatisch stabiler. Ermöglicht `dt = 1 Tag` mit Präzision wie Euler mit `dt = 1 Stunde`.

---

## Lernphasen

| Woche | Thema | Kernerkenntnis |
|---|---|---|
| 1 | Euler + 2-Körper | Erste Ellipse, Euler-Drift durch dt-Experiment sichtbar |
| 2 | 3-Körper + Chaos | 1.000 km Startunterschied → ~1 AU Abweichung nach 10 Jahren |
| 3 | RK4 + Energiemessung | 140 Mio.× bessere Energieerhaltung bei gleichem dt |
| 4 | Animation + N-Körper | Echtzeit-Simulation mit FuncAnimation, Schweife, Masse-Größe |

---

## Setup

```bash
pip install numpy matplotlib
python src/orbit_animate_n.py    # N-Körper-Animation starten
python src/orbit_rk4.py          # Euler vs. RK4 Vergleich
```

---

## Weiterführende Themen

- **Leapfrog / Störmer-Verlet** — symplektischer Integrator, Standard in Astrophysik
- **Barnes-Hut** — O(N log N) statt O(N²), skaliert auf 1000+ Körper
- **3D** — `pos/vel` als `np.array([x,y,z])`, `mpl_toolkits.mplot3d`
- **Reales Sonnensystem** — NASA Horizons API für exakte Startdaten

---

## Verbindungen

| Feld | Verbindung |
|---|---|
| Mechatronik / Regelung | Zustandsraum `[x, v]` identisch mit Systemzustand in der Regelungstechnik |
| Signalverarbeitung | FFT auf Trajektoriendaten → Orbitalperioden extrahieren |
| Machine Learning | Kalman-Filter für verrauschte Positionsdaten |
| Philosophie | Determinismus vs. Chaos: das 3-Körper-Problem ist deterministisch aber unvorhersehbar |
