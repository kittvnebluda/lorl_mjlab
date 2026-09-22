"""
Interactive visualisation of the foot trajectory generator of Eq. 11.

    F(phi_i) = ( h * poly(k) - 0.5 ) * z_Hi ,   k = 2 (phi_i - pi) / pi

    poly(k) = -2k^3 + 3k^2           k in [0, 1]   (rise)
              2k^3 - 9k^2 + 12k - 4  k in [1, 2]   (fall)
              0                      otherwise     (stance)

"""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from numpy import pi
from numpy.typing import NDArray

TWO_PI = 2.0 * pi

# ----------------------------------------------------------------------------
# Eq. 11
# ----------------------------------------------------------------------------


def ftg_height(phi: Iterable[float] | float, h: float, nominal: float) -> NDArray[np.float64]:
    """Scalar coefficient of z_Hi, in metres (negative = below the hip)."""
    phi = np.asarray(phi, dtype=np.float64) % TWO_PI
    k = 2.0 * (phi - pi) / pi

    rise = -2.0 * k**3 + 3.0 * k**2
    fall = 2.0 * k**3 - 9.0 * k**2 + 12.0 * k - 4.0

    poly = np.where(
        (k >= 0.0) & (k < 1.0),
        rise,
        np.where((k >= 1.0) & (k < 2.0), fall, 0.0),
    )
    return h * poly - nominal


def ftg_vector(phi: Iterable[float] | float, h: float, nominal: float) -> NDArray[np.float64]:
    """F(phi_i) in H_i coordinates: [0, 0, s(phi_i)]."""
    s = ftg_height(phi, h, nominal)
    zero = np.zeros_like(s)
    return np.stack([zero, zero, s], axis=-1)


# ----------------------------------------------------------------------------
# Defaults (paper values)
# ----------------------------------------------------------------------------

H_DEFAULT = 0.20  # maximum foot height h  [m]
NOMINAL_DEFAULT = 0.50  # the 0.5 offset in Eq. 11  [m]
F0_DEFAULT = 1.25  # base frequency f0  [Hz]

LEG_NAMES = ["leg 1", "leg 2", "leg 3", "leg 4"]
LEG_COLORS = ["#d1495b", "#2e86ab", "#f4a259", "#5b8c5a"]

DT = 1.0 / 60.0
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 7))
assert fig.canvas.manager is not None
fig.canvas.manager.set_window_title("Foot trajectory generator — Eq. 11")
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[2.4, 1.0],
    height_ratios=[1, 1],
    left=0.07,
    right=0.97,
    top=0.95,
    bottom=0.31,
    hspace=0.55,
    wspace=0.28,
)

ax_phase = fig.add_subplot(gs[0, 0])  # s vs phase
ax_time = fig.add_subplot(gs[1, 0])  # s vs time, four legs
ax_z = fig.add_subplot(gs[:, 1])  # the foot on the z_Hi axis

phi_grid = np.linspace(0.0, TWO_PI, 721)

# --- panel 1: the shape -----------------------------------------------------
(line_phase,) = ax_phase.plot([], [], color="#2e86ab", lw=2.0)
(marker_phase,) = ax_phase.plot([], [], "o", color="#d1495b", ms=8, zorder=5)
ax_phase.axvspan(pi, TWO_PI, color="#2e86ab", alpha=0.07)
ax_phase.text(1.5 * pi, 0, " swing  k ∈ [0, 2)", ha="center", va="top", fontsize=9, color="#2e86ab")
ax_phase.text(0.5 * pi, 0, " stance", ha="center", va="top", fontsize=9, color="#888888")
ax_phase.axvline(1.5 * pi, color="#2e86ab", ls=":", lw=1, alpha=0.7)
ax_phase.set_xlim(0, TWO_PI)
ax_phase.set_xticks([0, pi / 2, pi, 1.5 * pi, TWO_PI])
ax_phase.set_xticklabels(["0", "π/2", "π\nk=0", "3π/2\nk=1", "2π\nk=2"])
ax_phase.set_xlabel("phase  φ")
ax_phase.set_ylabel("coefficient of $z_{H_i}$  [m]")
ax_phase.set_title("Eq. 11, single leg", fontsize=11)
ax_phase.grid(alpha=0.25)

# --- panel 2: four legs over time -------------------------------------------
T_WINDOW = 3.0
t_hist = []
s_hist = [[] for _ in LEG_NAMES]
time_lines = []
for name, color in zip(LEG_NAMES, LEG_COLORS):
    (ln,) = ax_time.plot([], [], color=color, lw=1.6, label=name)
    time_lines.append(ln)
ax_time.set_xlabel("time  [s]")
ax_time.set_ylabel("coefficient of $z_{H_i}$  [m]")
ax_time.set_title("Four legs, $\\phi_{i,0} \\sim U(0, 2\\pi)$", fontsize=11)
ax_time.grid(alpha=0.25)
ax_time.legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.9)

# --- panel 3: the foot on the z axis ----------------------------------------
ax_z.set_title("$F(\\phi_i)$ along $z_{H_i}$", fontsize=11)
ax_z.set_xlim(-0.2, 0.2)
ax_z.set_xticks([])
ax_z.set_ylabel("z  [m]   (relative to hip)")
ax_z.grid(alpha=0.25, axis="y")
ax_z.plot(0, 0, "s", color="#333333", ms=10, zorder=6)
ax_z.annotate("hip  $H_i$", (0, 0), xytext=(8, 5), textcoords="offset points", fontsize=9)
(segment,) = ax_z.plot([], [], color="#d1495b", lw=2.5, alpha=0.35, zorder=2)
(leg_link,) = ax_z.plot([], [], color="#333333", lw=3, alpha=0.6, zorder=4)
(foot_dot,) = ax_z.plot([], [], "o", color="#d1495b", ms=13, zorder=7)
ground_line = ax_z.axhline(0.0, color="#8a6f48", lw=2, alpha=0.8, zorder=0)
apex_line = ax_z.axhline(0.0, color="#2e86ab", ls=":", lw=1, alpha=0.8, zorder=0)
readout = ax_z.text(0.5, 0.02, "", transform=ax_z.transAxes, ha="center", va="bottom", fontsize=9, family="monospace")

# ----------------------------------------------------------------------------
# Sliders
# ----------------------------------------------------------------------------

slider_axes = {
    "h": fig.add_axes((0.10, 0.20, 0.34, 0.028)),
    "nominal": fig.add_axes((0.10, 0.145, 0.34, 0.028)),
    "f0": fig.add_axes((0.62, 0.20, 0.30, 0.028)),
    "speed": fig.add_axes((0.62, 0.145, 0.30, 0.028)),
    "phi0": fig.add_axes((0.10, 0.09, 0.34, 0.028)),
}

s_h = Slider(slider_axes["h"], "h  [m]", 0.0, 0.40, valinit=H_DEFAULT, valfmt="%.3f", color="#2e86ab")
s_nominal = Slider(
    slider_axes["nominal"], "offset  [m]", 0.10, 0.80, valinit=NOMINAL_DEFAULT, valfmt="%.3f", color="#2e86ab"
)
s_f0 = Slider(slider_axes["f0"], "$f_0$  [Hz]", 0.0, 4.0, valinit=F0_DEFAULT, valfmt="%.2f", color="#f4a259")
s_speed = Slider(slider_axes["speed"], "playback  ×", 0.0, 2.0, valinit=1.0, valfmt="%.2f", color="#999999")
s_phi0 = Slider(slider_axes["phi0"], "$\\phi_{1,0}$  [rad]", 0.0, TWO_PI, valinit=0.0, valfmt="%.2f", color="#5b8c5a")

ax_reset = fig.add_axes((0.10, 0.025, 0.10, 0.042))
ax_pause = fig.add_axes((0.225, 0.025, 0.10, 0.042))
ax_sample = fig.add_axes((0.35, 0.025, 0.17, 0.042))
b_reset = Button(ax_reset, "Reset")
b_pause = Button(ax_pause, "Pause")
b_sample = Button(ax_sample, "Resample $\\phi_{i,0}$")

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------

Z_MIN, Z_MAX = -0.85, 0.05


class State:
    phi = 0.0  # phase of leg 1
    offsets = rng.uniform(0.0, TWO_PI, size=4)  # phi_i,0 ~ U(0, 2pi)
    t = 0.0
    running = True


state = State()


def redraw_static(_=None):
    """Recompute everything that depends only on the slider values."""
    h, nominal = s_h.val, s_nominal.val

    line_phase.set_data(phi_grid, ftg_height(phi_grid, h, nominal))

    lo, hi = -nominal, -nominal + h
    ax_phase.set_ylim(Z_MIN, Z_MAX)
    ax_time.set_ylim(Z_MIN, Z_MAX)
    ax_z.set_ylim(Z_MIN, Z_MAX)

    ground_line.set_ydata([lo, lo])
    apex_line.set_ydata([hi, hi])
    segment.set_data([0, 0], [lo, hi])
    fig.canvas.draw_idle()


for sl in (s_h, s_nominal):
    sl.on_changed(redraw_static)


def on_phi0(val):
    state.phi = val % TWO_PI


s_phi0.on_changed(on_phi0)


def clear_history():
    t_hist.clear()
    for hist in s_hist:
        hist.clear()


def on_reset(_event):
    for sl in (s_h, s_nominal, s_f0, s_speed, s_phi0):
        sl.reset()
    state.phi, state.t = 0.0, 0.0
    clear_history()


b_reset.on_clicked(on_reset)


def on_pause(_event):
    state.running = not state.running
    b_pause.label.set_text("Play" if not state.running else "Pause")


b_pause.on_clicked(on_pause)


def on_sample(_event):
    state.offsets = rng.uniform(0.0, TWO_PI, size=4)
    state.offsets[0] = 0.0
    clear_history()


b_sample.on_clicked(on_sample)

# ----------------------------------------------------------------------------
# Animation loop
# ----------------------------------------------------------------------------


def update(_frame):
    h, nominal = s_h.val, s_nominal.val
    freq, speed = s_f0.val, s_speed.val

    if state.running:
        state.phi = (state.phi + TWO_PI * freq * DT * speed) % TWO_PI
        state.t += DT * speed

    s_now = float(ftg_height(state.phi, h, nominal))
    k_now = 2.0 * (state.phi - pi) / pi
    marker_phase.set_data([state.phi], [s_now])

    t_hist.append(state.t)
    for i, off in enumerate(state.offsets):
        s_hist[i].append(float(ftg_height(state.phi + off, h, nominal)))
    while t_hist and t_hist[-1] - t_hist[0] > T_WINDOW:
        t_hist.pop(0)
        for hist in s_hist:
            hist.pop(0)
    for ln, hist in zip(time_lines, s_hist):
        ln.set_data(t_hist, hist)
    if t_hist:
        ax_time.set_xlim(t_hist[0], max(t_hist[0] + T_WINDOW, t_hist[-1]))

    leg_link.set_data([0, 0], [0, s_now])
    foot_dot.set_data([0], [s_now])

    in_swing = state.phi >= pi
    foot_dot.set_color("#d1495b" if in_swing else "#5b8c5a")
    readout.set_text(
        "φ = %5.2f   k = %5.2f\nz = %+.3f m   %s" % (state.phi, k_now, s_now, "swing" if in_swing else "stance")
    )

    return (marker_phase, leg_link, foot_dot, readout, *time_lines)


redraw_static()

timer = fig.canvas.new_timer(interval=int(DT * 1000))
timer.add_callback(lambda: (update(None), fig.canvas.draw_idle()))
timer.start()

plt.show()
