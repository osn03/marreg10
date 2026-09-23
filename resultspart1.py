"""
resultspart1.py -- results for the "Simulation results" section of the
TMR4240 Project Part 1 report (mandatory Simulations 1-4).

Run from the repository root:

    python resultspart1.py              # run all four simulations
    python resultspart1.py --only 3     # run a single simulation (1, 2, 3 or 4)
    python resultspart1.py --show       # also open the figures on screen

Everything is written to ``results_part1/``:

    figures/<name>.pdf     vector figures sized for the IEEE two-column format
    figures/<name>.png     quick-look previews of the same figures
    tables/<name>.tex      booktabs tables with the performance metrics
    results_part1.tex      ready-made figure/table environments to \\input
    metrics.json           all computed numbers (for writing the discussion)

LaTeX preamble needed:

    \\usepackage{graphicx,booktabs}
    \\graphicspath{{results_part1/figures/}}

and in the Simulation results section:

    \\input{results_part1/results_part1.tex}     % everything at once, or
    \\input{results_part1/tables/sim1.tex}       % single tables

The controller, reference model and allocator are built exactly as the
automated checks build them (constructor defaults), so the numbers in the
report match ``python check.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from part_1.config import SimConfig, default_thrusters_gunnerus3
from part_1.controller import DPController
from part_1.reference import ReferenceModel
from part_1.current import Current
from part_1.wind import Wind
from simulation.simulation_part_1 import DPSimulator3DOF

# =============================================================================
# Settings
# =============================================================================
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_part1"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"

DT = 0.05                 # [s]  time step (forward Euler, as required by the vessel)
SS_WINDOW = 100.0         # [s]  steady-state window at the end of a run / leg
POS_BAND = 0.5            # [m]  settling band for position error
PSI_BAND_DEG = 1.0        # [deg] settling band for heading error
SAT_LEVEL = 0.99          # a thruster counts as saturated at >= 99 % of u_max
WIND_SEED = 0             # seed for the slowly varying wind (reproducible)

T_SIM1 = 800.0            # [s]  Simulation 1a/1b duration
T_SIM2 = 900.0            # [s]  Simulation 2 duration
T_SIM3 = 600.0            # [s]  Simulation 3 duration
HOLD_SIM4 = 300.0         # [s]  Simulation 4 hold time per setpoint (prescribed)
ROTATION_TIME = 300.0     # [s]  Simulation 2 current rotation time

COL_W = 3.5               # [in] IEEE column width  (\columnwidth)
TEXT_W = 7.16             # [in] IEEE text width    (\textwidth)

THRUSTERS = default_thrusters_gunnerus3()
U_MAX = np.array([th.u_max for th in THRUSTERS])
TH_LABELS = [th.name.replace("_", " ") for th in THRUSTERS]
TH_SHORT = ["TB", "A1", "A2"]            # Tunnel Bow, Azimuth 1, Azimuth 2
AZI_IDX = [i for i, th in enumerate(THRUSTERS) if th.kind == "azimuth"]

C_ACT = "#1f77b4"        # actual response (with reference model)
C_REF = "#ff7f0e"        # reference-model output eta_d
C_NOREF = "#d62728"      # actual response without reference model
C_SP = "black"           # commanded setpoint
C_ENV = "#2ca02c"        # environmental load
C_TH = ["#2ca02c", "#1f77b4", "#9467bd"]

SHOW = False
FIGURES: list[dict] = []   # registry used to write results_part1.tex
CURRENT_SECTION = ["Set-up"]
TABLES: list[dict] = []
METRICS: dict = {}


def set_style() -> None:
    """Matplotlib style that matches an IEEE paper (Times-like, 8 pt)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
        "legend.handlelength": 1.8,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# =============================================================================
# Helpers: simulation, signals, metrics
# =============================================================================
def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def eta6(N=0.0, E=0.0, psi=0.0) -> np.ndarray:
    v = np.zeros(6)
    v[0], v[1], v[5] = N, E, psi
    return v


def run_sim(T: float, eta_cmd, *, current=None, wind=None, use_reference=True):
    """One closed-loop run with the tuned (default) subsystems."""
    cfg = SimConfig(dt=DT, T=T, method="Euler", use_reference=use_reference)
    ref = ReferenceModel(dt=cfg.dt) if use_reference else None
    sim = DPSimulator3DOF(cfg, DPController(), default_thrusters_gunnerus3(),
                          reference=ref, pkl_path=None)
    sim.reset_state()
    return sim.run(np.asarray(eta_cmd, dtype=float), current=current, wind=wind)


def window(logs, t0=0.0, t1=None) -> np.ndarray:
    t1 = logs.t[-1] if t1 is None else t1
    return (logs.t >= t0 - 1e-9) & (logs.t <= t1 + 1e-9)


def pos_err(logs, target=None) -> np.ndarray:
    """Horizontal distance to the commanded setpoint (or a fixed target)."""
    ref = logs.cmd if target is None else np.broadcast_to(target, logs.eta.shape)
    return np.hypot(logs.eta[:, 0] - ref[:, 0], logs.eta[:, 1] - ref[:, 1])


def psi_err(logs, target=None) -> np.ndarray:
    """Wrapped heading error psi - psi_cmd [rad]."""
    ref = logs.cmd if target is None else np.broadcast_to(target, logs.eta.shape)
    return wrap(logs.eta[:, 5] - ref[:, 5])


def settling_time(t, err, band, t0=0.0, t1=None) -> float:
    """Time (relative to t0) after which |err| stays inside the band.

    Returns 0 if it never leaves the band and NaN if it has not settled by t1.
    """
    t1 = t[-1] if t1 is None else t1
    sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    tt, ee = t[sel], np.abs(err[sel])
    outside = np.nonzero(ee > band)[0]
    if outside.size == 0:
        return 0.0
    k = outside[-1]
    if k >= len(tt) - 1:
        return float("nan")
    return float(tt[k + 1] - t0)


def point_segment_dist(P, A, B) -> np.ndarray:
    """Distance from points P (n,2) to the segment A-B (in N/E)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    AB = B - A
    L2 = float(AB @ AB)
    if L2 < 1e-12:
        return np.linalg.norm(P - A, axis=1)
    s = np.clip(((P - A) @ AB) / L2, 0.0, 1.0)
    return np.linalg.norm(P - (A + s[:, None] * AB), axis=1)


def thrust_metrics(logs, t0=0.0, t1=None) -> dict:
    m = window(logs, t0, t1)
    u = np.abs(logs.u[m])
    util = u / U_MAX
    return {
        "mean_thrust_kN": (u.mean(axis=0) / 1e3).tolist(),
        "peak_thrust_kN": (u.max(axis=0) / 1e3).tolist(),
        "peak_util_pct": (util.max(axis=0) * 100).tolist(),
        "sat_time_pct": float(np.mean(np.any(util >= SAT_LEVEL, axis=1)) * 100),
        # integral of sum |u| dt: a simple actuator-effort measure [MNs]
        "thrust_impulse_MNs": float(np.sum(u) * DT / 1e6),
    }


def window_metrics(logs, t0=0.0, t1=None) -> dict:
    """Station-keeping metrics on [t0, t1]."""
    m = window(logs, t0, t1)
    t = logs.t[m]
    e = pos_err(logs)[m]
    ep = np.rad2deg(np.abs(psi_err(logs)[m]))
    k = int(np.argmax(e))
    d = {
        "max_pos_m": float(e.max()), "t_max_pos_s": float(t[k]),
        "max_N_m": float(np.max(np.abs(logs.eta[m, 0] - logs.cmd[m, 0]))),
        "max_E_m": float(np.max(np.abs(logs.eta[m, 1] - logs.cmd[m, 1]))),
        "mean_pos_m": float(e.mean()), "rms_pos_m": float(np.sqrt(np.mean(e**2))),
        "max_psi_deg": float(ep.max()), "mean_psi_deg": float(ep.mean()),
        "rms_psi_deg": float(np.sqrt(np.mean(ep**2))),
        "mean_tau_thr": [float(np.mean(logs.tau_thr[m, 0]) / 1e3),
                         float(np.mean(logs.tau_thr[m, 1]) / 1e3),
                         float(np.mean(logs.tau_thr[m, 5]) / 1e3)],
        "mean_tau_wind": [float(np.mean(logs.tau_w6[m, 0]) / 1e3),
                          float(np.mean(logs.tau_w6[m, 1]) / 1e3),
                          float(np.mean(logs.tau_w6[m, 5]) / 1e3)],
    }
    d.update(thrust_metrics(logs, t0, t1))
    return d


def station_keeping_metrics(logs) -> dict:
    T = float(logs.t[-1])
    return {
        "whole": window_metrics(logs, 0.0, T),
        "steady": window_metrics(logs, T - SS_WINDOW, T),
        "settle_pos_s": settling_time(logs.t, pos_err(logs), POS_BAND),
        "settle_psi_s": settling_time(logs.t, psi_err(logs), np.deg2rad(PSI_BAND_DEG)),
        "mean_wind_speed": float(np.mean(logs.U_w)),
        "std_wind_speed": float(np.std(logs.U_w)),
    }


def maneuver_metrics(logs, start, target, t0=0.0, t1=None) -> dict:
    """Setpoint-change metrics for a move start -> target on [t0, t1]."""
    t1 = float(logs.t[-1]) if t1 is None else t1
    m = window(logs, t0, t1)
    t = logs.t
    start, target = np.asarray(start, float), np.asarray(target, float)
    e = pos_err(logs, target)
    ep = psi_err(logs, target)
    NE = logs.eta[m][:, [0, 1]]
    A, B = start[[0, 1]], target[[0, 1]]
    L = float(np.linalg.norm(B - A))

    # position overshoot along the direction of travel
    if L > 1e-6:
        dirv = (B - A) / L
        os_pos = max(0.0, float(np.max((NE - A) @ dirv - L)))
    else:
        os_pos = float("nan")
    # per-axis overshoot (N and E) along each axis' approach direction
    os_axis = []
    for i in (0, 1):
        dd = target[i] - start[i]
        os_axis.append(max(0.0, float(np.max(np.sign(dd) * (logs.eta[m, i] - target[i]))))
                       if abs(dd) > 1e-6 else float("nan"))
    # heading overshoot and "wrong-way" rotation
    dpsi = float(wrap(target[5] - start[5]))
    psi_m = logs.eta[m, 5]
    if abs(dpsi) > 1e-6:
        s = np.sign(dpsi)
        os_psi = max(0.0, float(np.rad2deg(np.max(s * wrap(psi_m - target[5])))))
        wrong = max(0.0, float(np.rad2deg(np.max(-s * wrap(psi_m - start[5])))))
    else:
        os_psi = wrong = float("nan")

    # time to reach 90 % of the commanded translation
    t90 = float("nan")
    if L > 1e-6:
        prog = (NE - A) @ ((B - A) / L)
        idx = np.nonzero(prog >= 0.9 * L)[0]
        if idx.size:
            t90 = float(t[m][idx[0]] - t0)

    nu = logs.nu[m]
    fin = window(logs, t1 - 50.0, t1)
    d = {
        "start": start[[0, 1, 5]].tolist(), "target": target[[0, 1, 5]].tolist(),
        "leg_length_m": L, "dpsi_deg": float(np.rad2deg(dpsi)),
        "overshoot_path_m": os_pos,
        "overshoot_N_m": os_axis[0], "overshoot_E_m": os_axis[1],
        "overshoot_psi_deg": os_psi, "wrong_way_psi_deg": wrong,
        "t90_s": t90,
        "settle_pos_s": settling_time(t, e, POS_BAND, t0, t1),
        "settle_psi_s": settling_time(t, ep, np.deg2rad(PSI_BAND_DEG), t0, t1),
        "max_path_dev_m": float(np.max(point_segment_dist(NE, A, B))),
        "path_length_m": float(np.sum(np.hypot(np.diff(NE[:, 0]), np.diff(NE[:, 1])))),
        "max_speed_ms": float(np.max(np.hypot(nu[:, 0], nu[:, 1]))),
        "max_yaw_rate_degs": float(np.rad2deg(np.max(np.abs(nu[:, 5])))),
        "peak_tau_d": [float(np.max(np.abs(logs.tau_d[m, 0])) / 1e3),
                       float(np.max(np.abs(logs.tau_d[m, 1])) / 1e3),
                       float(np.max(np.abs(logs.tau_d[m, 5])) / 1e3)],
        "final_pos_m": float(np.mean(e[fin])),
        "final_psi_deg": float(np.rad2deg(np.mean(np.abs(ep[fin])))),
    }
    d.update(thrust_metrics(logs, t0, t1))
    return d


# =============================================================================
# Helpers: LaTeX output
# =============================================================================
def num(x, nd=2) -> str:
    """Number in math mode (proper minus sign); '--' for NaN/None."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    x = round(float(x), nd) + 0.0          # avoids "-0.00"
    return f"${x:.{nd}f}$"


def trio(v, nd=1) -> str:
    """Three values as 'a / b / c' (e.g. TB / A1 / A2)."""
    return "/".join(num(x, nd) for x in v)


def settle_str(x) -> str:
    return "not settled" if (x is None or not np.isfinite(x)) else num(x, 1)


def write_table(name, caption, label, header, rows, colspec=None, wide=False,
                note=None) -> None:
    """Write a self-contained booktabs table. ``None`` in rows = \\midrule."""
    env = "table*" if wide else "table"
    colspec = colspec or ("l" + "c" * (len(header) - 1))
    out = [f"% Generated by resultspart1.py -- do not edit by hand",
           f"\\begin{{{env}}}[!t]",
           "\\centering",
           f"\\caption{{{caption}}}",
           f"\\label{{{label}}}",
           "\\footnotesize",
           "\\setlength{\\tabcolsep}{3pt}",
           "% shrink-to-fit: scaled down only if wider than the column/page",
           "\\sbox0{%",
           f"\\begin{{tabular}}{{{colspec}}}",
           "\\toprule",
           " & ".join(header) + " \\\\",
           "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r is None else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}",
            "\\ifdim\\wd0>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox0}"
            "\\else\\usebox0\\fi"]
    if note:
        width = "\\textwidth" if wide else "\\columnwidth"
        out += ["\\\\[2pt]", f"\\parbox{{{width}}}{{\\scriptsize {note}}}"]
    out.append(f"\\end{{{env}}}")
    (TAB_DIR / f"{name}.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    TABLES.append({"name": name, "label": label, "section": CURRENT_SECTION[0]})
    # plain-text echo for the terminal
    print(f"\n  [{name}] ({label})")
    strip = lambda s: s.replace("$", "").replace("\\,", " ").replace("\\", "")
    print("    " + " | ".join(strip(h) for h in header))
    for r in rows:
        if r is not None:
            print("    " + " | ".join(strip(c) for c in r))


def save(fig, name, caption, wide=False) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    FIGURES.append({"name": name, "caption": caption, "wide": wide,
                    "section": CURRENT_SECTION[0]})
    if not SHOW:
        plt.close(fig)
    print(f"    figure -> figures/{name}.pdf")


def write_master_tex() -> None:
    """results_part1.tex: every figure and table in a ready-to-input form."""
    lines = ["% Generated by resultspart1.py",
             "% Preamble: \\usepackage{graphicx,booktabs}",
             "%           \\graphicspath{{results_part1/figures/}}",
             "% Move the blocks around freely; each is self-contained.", ""]
    sections = list(dict.fromkeys([x["section"] for x in TABLES + FIGURES]))
    for sec in sections:
        lines += ["% " + "=" * 70, f"% {sec}", "% " + "=" * 70]
        for tab in TABLES:
            if tab["section"] == sec:
                lines.append(f"\\input{{results_part1/tables/{tab['name']}.tex}}")
        lines.append("")
        for f in FIGURES:
            if f["section"] != sec:
                continue
            env = "figure*" if f["wide"] else "figure"
            width = "\\textwidth" if f["wide"] else "\\columnwidth"
            lines += [f"\\begin{{{env}}}[!t]",
                  "  \\centering",
                  f"  \\includegraphics[width={width}]{{{f['name']}.pdf}}",
                  f"  \\caption{{{f['caption']}}}",
                  f"  \\label{{fig:{f['name']}}}",
                  f"\\end{{{env}}}", ""]
    (OUT / "results_part1.tex").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Helpers: plotting
# =============================================================================
def psi_deg(x):
    return np.rad2deg(wrap(x))


def mark_switches(axes, times, label_first=False):
    for ax in np.atleast_1d(axes):
        for i, ts in enumerate(times):
            ax.axvline(ts, color="0.5", lw=0.6, ls="--", zorder=0,
                       label="Setpoint switch" if (label_first and i == 0 and ax is axes[0]) else None)


def shade_steady(axes, T):
    for ax in np.atleast_1d(axes):
        ax.axvspan(T - SS_WINDOW, T, color="0.85", alpha=0.5, lw=0, zorder=0)


def fig_nepsi(runs, name, caption, *, setpoint_logs=None, ref_logs=None,
              width=COL_W, height=3.4, steady=False, switches=(), tmax=None,
              wide=False):
    """N(t), E(t), psi(t). runs = [(logs, label, color, linestyle), ...]."""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(width, height))
    base = runs[0][0]
    for i, (col, lab) in enumerate([(0, r"$N$ [m]"), (1, r"$E$ [m]"),
                                    (5, r"$\psi$ [deg]")]):
        f = psi_deg if col == 5 else (lambda x: x)
        for lg, label, c, ls in runs:
            ax[i].plot(lg.t, f(lg.eta[:, col]), color=c, ls=ls, label=label)
        if ref_logs is not None:
            ax[i].plot(ref_logs.t, f(ref_logs.sp[:, col]), color=C_REF, ls="--",
                       label=r"Reference $\eta_d$")
        spl = setpoint_logs if setpoint_logs is not None else base
        ax[i].plot(spl.t, f(spl.cmd[:, col]), color=C_SP, ls=":", lw=1.0,
                   label=r"Setpoint $\eta_{sp}$")
        ax[i].set_ylabel(lab)
    if steady:
        shade_steady(ax, base.t[-1])
    if switches:
        mark_switches(ax, switches)
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, base.t[-1] if tmax is None else tmax)
    ax[0].legend(loc="best", ncol=2 if len(runs) + 2 > 3 else 1)
    save(fig, name, caption, wide)


def fig_xy(runs, name, caption, *, ref_logs=None, setpoints=(), circles=(),
           arrows_every=None, width=COL_W, height=3.0, sp_labels=None):
    """North-East trajectory (East on the horizontal axis, North up)."""
    fig, ax = plt.subplots(figsize=(width, height))
    ext = max(float(np.ptp(np.concatenate([lg.eta[:, 0] for lg, *_ in runs]))),
              float(np.ptp(np.concatenate([lg.eta[:, 1] for lg, *_ in runs]))), 1.0)
    for r in circles:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.sin(th), r * np.cos(th), color="0.6", lw=0.6, ls=":")
        ax.annotate(f"{r:g} m", (r * np.sin(np.pi / 4), r * np.cos(np.pi / 4)),
                    fontsize=6, color="0.4")
    if ref_logs is not None:
        ax.plot(ref_logs.sp[:, 1], ref_logs.sp[:, 0], color=C_REF, ls="--",
                label=r"Reference $\eta_d$")
    for lg, label, c, ls in runs:
        ax.plot(lg.eta[:, 1], lg.eta[:, 0], color=c, ls=ls, label=label)
        ax.plot(lg.eta[0, 1], lg.eta[0, 0], "o", color=c, ms=3.5, mfc="white")
        ax.plot(lg.eta[-1, 1], lg.eta[-1, 0], "s", color=c, ms=3.5)
        if arrows_every:
            # one arrow every `arrows_every` s, skipped while the vessel is
            # at rest (avoids a pile of identical arrows at the setpoints)
            k, last = [], None
            for j in range(0, len(lg.t), int(round(arrows_every / DT))):
                p = lg.eta[j, [0, 1, 5]]
                if last is None or np.hypot(*(p[:2] - last[:2])) > 0.08 * ext \
                        or abs(wrap(p[2] - last[2])) > np.deg2rad(15):
                    k.append(j)
                    last = p
            k = np.array(k)
            psi = lg.eta[k, 5]
            ax.quiver(lg.eta[k, 1], lg.eta[k, 0], np.sin(psi), np.cos(psi),
                      color=c, angles="xy", scale_units="inches", scale=6,
                      width=0.004, headwidth=4, alpha=0.8, zorder=3)
    for j, sp in enumerate(setpoints):
        ax.plot(sp[1], sp[0], "x", color=C_SP, ms=6, mew=1.2,
                label=r"Setpoint $\eta_{sp}$" if j == 0 else None)
        if sp_labels:
            ax.annotate(sp_labels[j], (sp[1], sp[0]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    ax.plot([], [], "o", color="0.3", ms=3.5, mfc="white", ls="none", label="Start")
    ax.plot([], [], "s", color="0.3", ms=3.5, ls="none", label="End")
    ax.set_xlabel(r"East $E$ [m]")
    ax.set_ylabel(r"North $N$ [m]")
    ax.margins(0.06)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    save(fig, name, caption)


def circles_for(logs):
    """1 m circle always; 5 m circle only if the vessel gets anywhere near it."""
    return (1.0, 5.0) if np.max(pos_err(logs)) > 2.5 else (1.0,)


def circ_txt(logs):
    return " and ".join(f"{r:g}\\,m" for r in circles_for(logs))


def fig_thrusters(logs, name, caption, *, width=COL_W, height=2.8,
                  switches=(), wide=False):
    """Thrust per thruster and azimuth angles."""
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(width, height))
    for i, lab in enumerate(TH_LABELS):
        ax[0].plot(logs.t, logs.u[:, i] / 1e3, color=C_TH[i], label=lab)
    for i, umax in enumerate(U_MAX):      # show +-u_max only when it is approached
        if np.max(np.abs(logs.u[:, i])) > 0.5 * umax:
            for s in (1, -1):
                ax[0].axhline(s * umax / 1e3, color=C_TH[i], lw=0.6, ls=":")
    ax[0].set_ylabel(r"Thrust $u_i$ [kN]")
    ax[0].legend(loc="best", ncol=3)
    for i in AZI_IDX:
        a = psi_deg(logs.alpha[:, i]).astype(float)
        a[1:][np.abs(np.diff(a)) > 300.0] = np.nan   # +-180 deg wrap: no vertical line
        ax[1].plot(logs.t, a, color=C_TH[i], label=TH_LABELS[i])
    ax[1].set_ylabel(r"Azimuth $\alpha_i$ [deg]")
    ax[1].set_ylim(-190, 190)
    ax[1].set_yticks([-180, -90, 0, 90, 180])
    if switches:
        mark_switches(ax, switches)
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, logs.t[-1])
    save(fig, name, caption, wide)


def fig_wrench(logs, name, caption, *, wind=False, width=COL_W, height=3.6):
    """Commanded vs thruster wrench (and wind loads/speed if present)."""
    n = 4 if wind else 3
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(width, height + (0.8 if wind else 0)))
    a0 = 0
    if wind:
        ax[0].plot(logs.t, logs.U_w, color=C_ENV, label=r"$U_w$")
        ax[0].axhline(np.mean(logs.U_w), color="0.3", ls=":", lw=0.8, label="Mean")
        ax[0].set_ylabel(r"$U_w$ [m/s]")
        ax[0].legend(loc="best", ncol=2)
        a0 = 1
    for j, (col, lab) in enumerate([(0, r"$X$ [kN]"), (1, r"$Y$ [kN]"),
                                    (5, r"$N_z$ [kNm]")]):
        a = ax[a0 + j]
        a.plot(logs.t, logs.tau_thr[:, col] / 1e3, color=C_ACT, label=r"Thrusters $\tau_{thr}$")
        a.plot(logs.t, logs.tau_d[:, col] / 1e3, color=C_REF, ls="--", lw=0.8,
               label=r"Commanded $\tau_c$")
        if wind:
            a.plot(logs.t, logs.tau_w6[:, col] / 1e3, color=C_ENV, lw=0.8,
                   label=r"Wind $\tau_{wind}$")
        a.set_ylabel(lab)
    ax[a0 + 1].legend(loc="center", ncol=3 if wind else 2)
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, logs.t[-1])
    save(fig, name, caption)


def fig_velocities(runs, name, caption, *, ref_logs=None, width=COL_W,
                   height=3.4, switches=(), tmax=None, wide=False):
    """Body velocities u, v, r with reference-model velocities."""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(width, height))
    for i, (col, lab, sc) in enumerate([(0, r"$u$ [m/s]", 1.0), (1, r"$v$ [m/s]", 1.0),
                                        (5, r"$r$ [deg/s]", 180 / np.pi)]):
        for lg, label, c, ls in runs:
            ax[i].plot(lg.t, lg.nu[:, col] * sc, color=c, ls=ls, label=label)
        if ref_logs is not None:
            ax[i].plot(ref_logs.t, ref_logs.nu_ref[:, col] * sc, color=C_REF, ls="--",
                       label=r"Reference $\nu_d$")
        ax[i].set_ylabel(lab)
    if switches:
        mark_switches(ax, switches)
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, runs[0][0].t[-1] if tmax is None else tmax)
    ax[1].legend(loc="upper right", ncol=3 if wide else 1)
    save(fig, name, caption, wide)


# =============================================================================
# Simulation 1 -- station keeping
# =============================================================================
def simulation_1():
    CURRENT_SECTION[0] = "Simulation 1: station keeping"
    print("\n=== Simulation 1: station keeping ===")
    sp = np.zeros(6)
    print("  running 1a (current 0.5 m/s from east) ...")
    lg_a = run_sim(T_SIM1, sp, current=Current(0.5, np.pi / 2, semantics="from"),
                   wind=Wind())
    print("  running 1b (wind 15 m/s from east, sigma 1 m/s) ...")
    lg_b = run_sim(T_SIM1, sp, current=Current(),
                   wind=Wind(15.0, np.pi / 2, semantics="from", sigma_slow=1.0,
                             seed=WIND_SEED))
    ma, mb = station_keeping_metrics(lg_a), station_keeping_metrics(lg_b)
    METRICS["sim1a_current"], METRICS["sim1b_wind"] = ma, mb

    env_a = r"a $0.5$\,m/s current from east (no wind)"
    env_b = (r"a $15$\,m/s mean wind from east with a slowly varying component "
             r"($\sigma=1$\,m/s, no current)")
    for lg, key, env, is_wind in [(lg_a, "sim1a", env_a, False),
                                  (lg_b, "sim1b", env_b, True)]:
        tag = "1a" if key == "sim1a" else "1b"
        fig_nepsi([(lg, "Actual", C_ACT, "-")], f"{key}_time",
                  f"Simulation {tag}: station keeping at $\\eta_{{sp}}=[0,0,0]^\\top$ in "
                  f"{env}. North position, East position and heading. The shaded "
                  f"area is the last {SS_WINDOW:.0f}\\,s used for steady-state metrics.",
                  steady=True)
        fig_xy([(lg, "Vessel", C_ACT, "-")], f"{key}_xy",
               f"Simulation {tag}: $xy$-trajectory (North vs.\\ East) in {env}. "
               "Dotted circle(s) mark " + circ_txt(lg) + " from the setpoint.",
               setpoints=[np.zeros(3)], circles=circles_for(lg))
        fig_thrusters(lg, f"{key}_thrusters",
                      f"Simulation {tag}: thruster forces and azimuth angles in {env}.")
        fig_wrench(lg, f"{key}_wrench",
                   f"Simulation {tag}: commanded and produced thruster wrench"
                   + (" together with the wind load and wind speed" if is_wind else "")
                   + f" in {env}.", wind=is_wind)

    def srow(label, fa, fb):
        return [label, fa(ma), fb(mb)]
    W, S = "whole", "steady"
    rows = [
        srow("Max.\\ position error [m]", lambda m: num(m[W]["max_pos_m"]), lambda m: num(m[W]["max_pos_m"])),
        srow("\\quad at time [s]", lambda m: num(m[W]["t_max_pos_s"], 0), lambda m: num(m[W]["t_max_pos_s"], 0)),
        srow("Max.\\ $|e_N|$ / $|e_E|$ [m]",
             lambda m: f"{num(m[W]['max_N_m'])} / {num(m[W]['max_E_m'])}",
             lambda m: f"{num(m[W]['max_N_m'])} / {num(m[W]['max_E_m'])}"),
        srow("Max.\\ heading error [deg]", lambda m: num(m[W]["max_psi_deg"]), lambda m: num(m[W]["max_psi_deg"])),
        srow(f"Settling, pos.\\ ($<{POS_BAND}$\\,m) [s]", lambda m: settle_str(m["settle_pos_s"]), lambda m: settle_str(m["settle_pos_s"])),
        srow(f"Settling, head.\\ ($<{PSI_BAND_DEG:g}^\\circ$) [s]", lambda m: settle_str(m["settle_psi_s"]), lambda m: settle_str(m["settle_psi_s"])),
        None,
        srow("SS mean position error [m]", lambda m: num(m[S]["mean_pos_m"], 3), lambda m: num(m[S]["mean_pos_m"], 3)),
        srow("SS RMS position error [m]", lambda m: num(m[S]["rms_pos_m"], 3), lambda m: num(m[S]["rms_pos_m"], 3)),
        srow("SS mean heading error [deg]", lambda m: num(m[S]["mean_psi_deg"], 3), lambda m: num(m[S]["mean_psi_deg"], 3)),
        srow("SS thr.\\ $X$/$Y$/$N_z$ [kN, kNm]", lambda m: trio(m[S]["mean_tau_thr"]), lambda m: trio(m[S]["mean_tau_thr"])),
        srow("SS wind $X$/$Y$/$N_z$ [kN, kNm]", lambda m: "--", lambda m: trio(m[S]["mean_tau_wind"])),
        None,
        srow("SS mean thrust TB/A1/A2 [kN]", lambda m: trio(m[S]["mean_thrust_kN"]), lambda m: trio(m[S]["mean_thrust_kN"])),
        srow("Peak thrust TB/A1/A2 [kN]", lambda m: trio(m[W]["peak_thrust_kN"]), lambda m: trio(m[W]["peak_thrust_kN"])),
        srow("Peak util.\\ TB/A1/A2 [\\%]", lambda m: trio(m[W]["peak_util_pct"], 0), lambda m: trio(m[W]["peak_util_pct"], 0)),
        srow("Time saturated [\\%]", lambda m: num(m[W]["sat_time_pct"], 1), lambda m: num(m[W]["sat_time_pct"], 1)),
    ]
    write_table("sim1", "Simulation 1: station-keeping performance at "
                "$\\eta_{sp}=[0,0,0]^\\top$. SS = steady state (mean over the last "
                f"{SS_WINDOW:.0f}\\,s of the {T_SIM1:.0f}\\,s run). TB = Tunnel Bow, "
                "A1/A2 = Azimuth 1/2.", "tab:sim1",
                ["Metric", "1a: current", "1b: wind"], rows, colspec="lcc")


# =============================================================================
# Simulation 2 -- rotating current
# =============================================================================
def simulation_2():
    CURRENT_SECTION[0] = "Simulation 2: rotating current"
    print("\n=== Simulation 2: rotating current ===")
    print(f"  running (current 0.5 m/s, from north -> from east over {ROTATION_TIME:.0f} s) ...")
    lg = run_sim(T_SIM2, np.zeros(6),
                 current=Current(0.5, 0.0, semantics="from", beta_end=np.pi / 2,
                                 duration=ROTATION_TIME),
                 wind=Wind())
    T = float(lg.t[-1])
    m_rot = window_metrics(lg, 0.0, ROTATION_TIME)
    m_after = window_metrics(lg, ROTATION_TIME, T)
    m_ss = window_metrics(lg, T - SS_WINDOW, T)
    settle_after = settling_time(lg.t, pos_err(lg), POS_BAND, ROTATION_TIME)
    METRICS["sim2"] = {"rotation": m_rot, "after": m_after, "steady": m_ss,
                       "settle_pos_after_rotation_s": settle_after,
                       "settle_psi_after_rotation_s":
                           settling_time(lg.t, psi_err(lg), np.deg2rad(PSI_BAND_DEG), ROTATION_TIME)}

    env = (r"a $0.5$\,m/s current whose direction rotates linearly from north to "
           rf"east over {ROTATION_TIME:.0f}\,s (no wind)")
    fig_nepsi([(lg, "Actual", C_ACT, "-")], "sim2_time",
              f"Simulation 2: station keeping in {env}. The dashed line marks the end "
              "of the current rotation; the shaded area is the steady-state window.",
              steady=True, switches=(ROTATION_TIME,))
    fig_xy([(lg, "Vessel", C_ACT, "-")], "sim2_xy",
           f"Simulation 2: $xy$-trajectory in {env}. Dotted circle(s) mark "
           + circ_txt(lg) + " from the setpoint.", setpoints=[np.zeros(3)],
           circles=circles_for(lg))

    # current input
    VN, VE = lg.Uc * np.cos(lg.beta_c), lg.Uc * np.sin(lg.beta_c)
    beta_from = (np.rad2deg(lg.beta_c) + 180.0) % 360.0
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(COL_W, 2.4))
    ax[0].plot(lg.t, VN, color=C_ACT, label=r"$V_{c,N}$")
    ax[0].plot(lg.t, VE, color=C_NOREF, label=r"$V_{c,E}$")
    ax[0].plot(lg.t, lg.Uc, color="0.4", ls=":", label=r"$V_c=|\nu_c^n|$")
    ax[0].set_ylabel("Current [m/s]")
    ax[0].legend(loc="center right", ncol=1)
    ax[1].plot(lg.t, beta_from, color=C_ENV)
    ax[1].set_ylabel("Dir.\\ from [deg]".replace("\\", ""))
    ax[1].set_yticks([0, 45, 90])
    mark_switches(ax, (ROTATION_TIME,))
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, T)
    save(fig, "sim2_current",
         "Simulation 2: NED current components $V_{c,N}(t)$, $V_{c,E}(t)$ returned to "
         "the plant, and the direction the current comes from (0$^\\circ$ = north, "
         "90$^\\circ$ = east).")
    fig_thrusters(lg, "sim2_thrusters",
                  "Simulation 2: thruster forces and azimuth angles during the "
                  "rotating-current test.", switches=(ROTATION_TIME,))
    fig_wrench(lg, "sim2_wrench",
               "Simulation 2: commanded and produced thruster wrench during the "
               "rotating-current test.")

    cols = [m_rot, m_after, m_ss]
    r = lambda lab, f: [lab] + [f(m) for m in cols]
    rows = [
        r("Max.\\ position error [m]", lambda m: num(m["max_pos_m"])),
        r("\\quad at time [s]", lambda m: num(m["t_max_pos_s"], 0)),
        r("Mean / RMS pos.\\ error [m]", lambda m: f"{num(m['mean_pos_m'])} / {num(m['rms_pos_m'])}"),
        r("Max.\\ heading error [deg]", lambda m: num(m["max_psi_deg"])),
        r("Mean heading error [deg]", lambda m: num(m["mean_psi_deg"], 3)),
        None,
        r("Mean thr.\\ $X$/$Y$/$N_z$ [kN, kNm]", lambda m: trio(m["mean_tau_thr"])),
        r("Mean thrust TB/A1/A2 [kN]", lambda m: trio(m["mean_thrust_kN"])),
        r("Peak thrust TB/A1/A2 [kN]", lambda m: trio(m["peak_thrust_kN"])),
        r("Saturated time [\\%]", lambda m: num(m["sat_time_pct"], 1)),
        None,
        [f"Settling after rot.\\ ($<{POS_BAND}$\\,m) [s]", "--", settle_str(settle_after), "--"],
    ]
    write_table("sim2", "Simulation 2: station keeping in a rotating current "
                f"($0.5$\\,m/s, from north to from east over {ROTATION_TIME:.0f}\\,s). "
                f"Metrics during the rotation, after it, and in the last "
                f"{SS_WINDOW:.0f}\\,s.", "tab:sim2",
                ["Metric", f"$0$--${ROTATION_TIME:.0f}$\\,s",
                 f"${ROTATION_TIME:.0f}$--${T:.0f}$\\,s", "Last "
                 f"${SS_WINDOW:.0f}$\\,s"], rows, colspec="lccc", wide=False)


# =============================================================================
# Simulation 3 -- setpoint change with and without reference model
# =============================================================================
def simulation_3():
    CURRENT_SECTION[0] = "Simulation 3: setpoint change with/without reference model"
    print("\n=== Simulation 3: setpoint change with/without reference model ===")
    start = eta6(0, 0, 0)
    target = eta6(10.0, 10.0, 3.0 * np.pi / 2.0)
    print("  running with reference model ...")
    lg_r = run_sim(T_SIM3, target, use_reference=True)
    print("  running without reference model ...")
    lg_n = run_sim(T_SIM3, target, use_reference=False)
    mr, mn = maneuver_metrics(lg_r, start, target), maneuver_metrics(lg_n, start, target)
    METRICS["sim3_with_ref"], METRICS["sim3_without_ref"] = mr, mn

    # zoom time axes to where things happen
    t_end = max(np.nan_to_num(mr["settle_pos_s"], nan=T_SIM3),
                np.nan_to_num(mn["settle_pos_s"], nan=T_SIM3),
                np.nan_to_num(mr["settle_psi_s"], nan=T_SIM3),
                np.nan_to_num(mn["settle_psi_s"], nan=T_SIM3))
    tmax = float(min(T_SIM3, 50.0 * np.ceil(1.4 * t_end / 50.0)))

    runs = [(lg_r, "With ref.\\ model".replace("\\", ""), C_ACT, "-"),
            (lg_n, "Without ref.\\ model".replace("\\", ""), C_NOREF, "-")]
    desc = (r"step from $\eta_0=[0,0,0]^\top$ to $\eta_{sp}=[10,10,3\pi/2]^\top$ "
            r"($3\pi/2 \equiv -\pi/2$), no environmental loads")
    fig_nepsi(runs, "sim3_time",
              f"Simulation 3: {desc}. Response with and without the reference model, "
              "the reference-model output $\\eta_d$ and the setpoint.",
              ref_logs=lg_r, tmax=tmax, height=3.6)
    fig_xy(runs, "sim3_xy",
           f"Simulation 3: $xy$-trajectories for the {desc}. Arrows show the vessel "
           "heading every 20\\,s.", ref_logs=lg_r, setpoints=[target[[0, 1, 5]]],
           arrows_every=20.0, height=3.2)
    fig_velocities(runs, "sim3_velocities",
                   f"Simulation 3: body-frame velocities $u$, $v$, $r$ with and without "
                   "the reference model, together with the reference-model velocities "
                   "$\\nu_d$ (mapped to the body frame).", ref_logs=lg_r, tmax=tmax)

    # thrust comparison (one row per thruster)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(COL_W, 3.4))
    for i, lab in enumerate(TH_LABELS):
        ax[i].plot(lg_r.t, lg_r.u[:, i] / 1e3, color=C_ACT, label="With ref. model")
        ax[i].plot(lg_n.t, lg_n.u[:, i] / 1e3, color=C_NOREF, label="Without ref. model")
        for s in (1, -1):
            ax[i].axhline(s * U_MAX[i] / 1e3, color="0.4", lw=0.6, ls=":")
        ax[i].set_ylabel(f"{TH_SHORT[i]} [kN]")
    ax[0].legend(loc="best", ncol=2)
    ax[-1].set_xlabel(r"Time $t$ [s]")
    ax[-1].set_xlim(0, tmax)
    save(fig, "sim3_thrusters",
         "Simulation 3: thrust of the Tunnel Bow (TB) and Azimuth 1/2 (A1/A2) "
         "thrusters with and without the reference model (dotted: $\\pm u_{\\max}$).")

    r = lambda lab, f, nd=2: [lab, f(mr), f(mn)]
    rows = [
        r("Overshoot $N$ / $E$ [m]", lambda m: f"{num(m['overshoot_N_m'])} / {num(m['overshoot_E_m'])}"),
        r("Overshoot along path [m]", lambda m: num(m["overshoot_path_m"])),
        r("Heading overshoot [deg]", lambda m: num(m["overshoot_psi_deg"])),
        r("Rotation in wrong dir.\\ [deg]", lambda m: num(m["wrong_way_psi_deg"])),
        None,
        r("Time to 90\\,\\% of translation [s]", lambda m: num(m["t90_s"], 1)),
        r(f"Settling, pos.\\ ($<{POS_BAND}$\\,m) [s]", lambda m: settle_str(m["settle_pos_s"])),
        r(f"Settling, head.\\ ($<{PSI_BAND_DEG:g}^\\circ$) [s]", lambda m: settle_str(m["settle_psi_s"])),
        r("Final position error [m]", lambda m: num(m["final_pos_m"], 3)),
        r("Final heading error [deg]", lambda m: num(m["final_psi_deg"], 3)),
        None,
        r("Max.\\ dev.\\ from straight line [m]", lambda m: num(m["max_path_dev_m"])),
        r("Path length (straight $14.14$) [m]", lambda m: num(m["path_length_m"])),
        r("Max.\\ speed / yaw rate [m/s, deg/s]", lambda m: f"{num(m['max_speed_ms'])} / {num(m['max_yaw_rate_degs'])}"),
        None,
        r("Peak $|\\tau_c|$ $X$/$Y$/$N_z$ [kN, kNm]", lambda m: trio(m["peak_tau_d"], 0)),
        r("Peak thrust TB/A1/A2 [kN]", lambda m: trio(m["peak_thrust_kN"])),
        r("Time saturated [\\%]", lambda m: num(m["sat_time_pct"], 1)),
        r("$\\int\\sum_i|u_i|\\,dt$ [MNs]", lambda m: num(m["thrust_impulse_MNs"])),
    ]
    write_table("sim3", "Simulation 3: setpoint change from $[0,0,0]^\\top$ to "
                "$[10,10,3\\pi/2]^\\top$ without environmental loads, with and without "
                "the reference model (identical controller and allocation). Final errors "
                "are means over the last 50\\,s.", "tab:sim3",
                ["Metric", "With ref.", "Without ref."], rows, colspec="lcc")


# =============================================================================
# Simulation 4 -- four-corner test
# =============================================================================
def simulation_4():
    CURRENT_SECTION[0] = "Simulation 4: four-corner test"
    print("\n=== Simulation 4: four-corner test ===")
    corners = [eta6(0, 0, 0), eta6(50, 0, 0), eta6(50, -50, 0),
               eta6(50, -50, -np.pi / 4), eta6(0, -50, -np.pi / 4), eta6(0, 0, 0)]
    legs = corners[1:]
    T = HOLD_SIM4 * len(legs)
    n_steps = int(round(T / DT)) + 1
    per_leg = int(round(HOLD_SIM4 / DT))
    eta_cmd = np.zeros((n_steps, 6))
    for i, c in enumerate(legs):
        rows = slice(i * per_leg, n_steps if i == len(legs) - 1 else (i + 1) * per_leg)
        eta_cmd[rows] = c
    print(f"  running ({len(legs)} legs x {HOLD_SIM4:.0f} s) ...")
    lg = run_sim(T, eta_cmd, use_reference=True)

    leg_m = []
    for i in range(len(legs)):
        t0, t1 = i * HOLD_SIM4, (i + 1) * HOLD_SIM4
        leg_m.append(maneuver_metrics(lg, corners[i], corners[i + 1], t0, t1))
    METRICS["sim4_legs"] = leg_m
    METRICS["sim4_total"] = thrust_metrics(lg)

    switches = tuple(HOLD_SIM4 * k for k in range(1, len(legs)))
    run = [(lg, "Actual", C_ACT, "-")]
    fig_nepsi(run, "sim4_time",
              "Simulation 4: four-corner test without environmental loads. North, East "
              "and heading with the reference-model output $\\eta_d$ and the setpoint. "
              f"Setpoints switch every {HOLD_SIM4:.0f}\\,s (dashed lines).",
              ref_logs=lg, width=TEXT_W, height=3.4, switches=switches, wide=True)
    labels = [r"$\eta_0,\eta_5$", r"$\eta_1$", r"$\eta_2$", "", r"$\eta_3,\eta_4$"]
    sp_pts = [c[[0, 1, 5]] for c in corners[:5]]
    lab_pts = [r"$\eta_0,\eta_5$", r"$\eta_1$", r"$\eta_2,\eta_3$", "", r"$\eta_4$"]
    fig_xy(run, "sim4_xy",
           "Simulation 4: $xy$-trajectory of the four-corner test with the reference "
           "path. Arrows show the vessel heading every 50\\,s.",
           ref_logs=lg, setpoints=sp_pts, sp_labels=lab_pts, arrows_every=50.0,
           height=3.3)
    fig_velocities(run, "sim4_velocities",
                   "Simulation 4: body-frame velocities $u$, $v$, $r$ and the "
                   "reference-model velocities $\\nu_d$ (mapped to the body frame).",
                   ref_logs=lg, width=TEXT_W, height=3.2, switches=switches, wide=True)
    fig_thrusters(lg, "sim4_thrusters",
                  "Simulation 4: thruster forces (dotted: $\\pm u_{\\max}$ of the Tunnel Bow "
                  "thruster) and azimuth angles during the four-corner test.", width=TEXT_W, height=2.6,
                  switches=switches, wide=True)

    def sp_str(c):
        return f"$({c[0]:g},\\,{c[1]:g},\\,{np.rad2deg(c[5]):g}^\\circ)$"
    rows = []
    for i, m in enumerate(leg_m):
        rows.append([
            f"$\\eta_{{{i}}}\\!\\rightarrow\\!\\eta_{{{i + 1}}}$", sp_str(legs[i]),
            settle_str(m["settle_pos_s"]), settle_str(m["settle_psi_s"]),
            num(m["overshoot_path_m"]), num(m["overshoot_psi_deg"]),
            num(m["max_path_dev_m"]), num(m["final_pos_m"], 3),
            num(m["final_psi_deg"], 3), num(max(m["peak_util_pct"]), 0),
        ])
    write_table("sim4", "Simulation 4: four-corner test, metrics per leg "
                f"({HOLD_SIM4:.0f}\\,s hold per setpoint, time measured from the "
                f"setpoint switch). Settling bands: {POS_BAND}\\,m and "
                f"{PSI_BAND_DEG:g}$^\\circ$. OS = overshoot along the direction of travel; "
                "path dev.\\ = max.\\ distance from the straight line between setpoints; "
                "final errors are means over the last 50\\,s of each leg; util.\\ = "
                "peak thrust relative to $u_{\\max}$ (worst thruster).", "tab:sim4",
                ["Leg", "Setpoint $(N,E,\\psi)$", "$t_{s,\\mathrm{pos}}$ [s]",
                 "$t_{s,\\psi}$ [s]", "OS [m]", "OS$_\\psi$ [deg]", "Path dev.\\ [m]",
                 "$\\bar e_{\\mathrm{pos}}$ [m]", "$\\bar e_{\\psi}$ [deg]", "Util.\\ [\\%]"],
                rows, colspec="llcccccccc", wide=True)


# =============================================================================
# Overview table of the simulation set-up
# =============================================================================
def setup_table():
    rows = [
        ["1a", "$[0,0,0]^\\top$", "Current $0.5$\\,m/s from east", "On", f"${T_SIM1:.0f}$"],
        ["1b", "$[0,0,0]^\\top$", "Wind $15$\\,m/s from east, $\\sigma=1$\\,m/s", "On", f"${T_SIM1:.0f}$"],
        ["2", "$[0,0,0]^\\top$", f"Current $0.5$\\,m/s, N$\\to$E in {ROTATION_TIME:.0f}\\,s", "On", f"${T_SIM2:.0f}$"],
        ["3", "$[10,10,3\\pi/2]^\\top$", "None", "On / Off", f"${T_SIM3:.0f}$"],
        ["4", "Four corners", "None", "On", f"${HOLD_SIM4 * 5:.0f}$"],
    ]
    write_table("sim_setup", "Simulation set-up. All runs use forward Euler with "
                f"$h={DT}$\\,s, ideal thrusters (Part 1 default), start at "
                "$\\eta_0=[0,0,0]^\\top$, $\\nu_0=0$, and the same controller and "
                "thrust allocation.", "tab:sim_setup",
                ["Sim.", "Setpoint", "Environment", "Ref.", "$T$ [s]"], rows,
                colspec="lllcc")


# =============================================================================
def main():
    global SHOW
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--only", type=int, choices=[1, 2, 3, 4], help="run one simulation")
    p.add_argument("--show", action="store_true", help="show figures on screen")
    args = p.parse_args()
    SHOW = args.show

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    sims = {1: simulation_1, 2: simulation_2, 3: simulation_3, 4: simulation_4}
    setup_table()
    for k, fn in sims.items():
        if args.only is None or args.only == k:
            fn()

    write_master_tex()
    (OUT / "metrics.json").write_text(json.dumps(METRICS, indent=2), encoding="utf-8")
    print(f"\nDone. Results written to {OUT}")
    print("  figures/  -> PDF (for LaTeX) + PNG (preview)")
    print("  tables/   -> booktabs .tex tables")
    print("  results_part1.tex -> \\input this in the Simulation results section")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
