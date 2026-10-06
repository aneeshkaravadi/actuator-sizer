"""Regenerate every figure and number in the README.   python examples/make_figures.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from actsizer import actuator as A
from actsizer import lift, model
from actsizer.cad import actuator_housing

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
results: dict = {}
J = model.JOINTS


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


(ROOT / "models" / "sagittal_humanoid.xml").write_text(model.build_mjcf())

# ------------------------------------------------------------------ 1. fastest balanced lift
def fastest_balanced(style, lo=0.6, hi=4.0):
    if not lift.simulate_lift(style, duration=hi).balanced():
        return None
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if lift.simulate_lift(style, duration=mid).balanced() else (mid, hi)
    return hi


results["fastest_balanced_lift_s"] = {s: fastest_balanced(s) for s in ("squat", "stoop")}
lifts = {s: lift.simulate_lift(s, duration=round(results["fastest_balanced_lift_s"][s] + 0.05, 2)) for s in ("squat", "stoop")}
results["lift_peaks_both_sides"] = {s: r.peak() for s, r in lifts.items()}

fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
for col, (s, r) in enumerate(lifts.items()):
    for i, j in enumerate(J):
        axes[0, col].plot(r.t, r.tau[:, i] / 2, label=j)
        axes[1, col].plot(r.t, r.power[:, i] / 2, label=j)
    axes[0, col].set_title(f"{s} lift, 25 kg tote, {r.t[-1]:.2f} s (fastest that stays balanced)", fontsize=9)
    axes[1, col].set_xlabel("time (s)")
axes[0, 0].set_ylabel("joint torque per side (N m)")
axes[1, 0].set_ylabel("joint power per side (W)")
axes[0, 0].legend(fontsize=8)
save(fig, "lift_torques.png")

# posture strip
m, d = model.load()
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, (s, r) in zip(axes, lifts.items()):
    for k in np.linspace(0, len(r.t) - 1, 6).astype(int):
        d.qpos[:] = r.q[k]
        mujoco.mj_kinematics(m, d)
        names = ["shank", "thigh", "trunk", "upper_arm", "forearm", "tote"]
        pts = [np.array([0.0, 0.08])] + [d.body(n).xpos[[0, 2]] for n in names[1:]] + [d.site("grip").xpos[[0, 2]]]
        pts = np.array(pts)
        a = 0.25 + 0.75 * k / (len(r.t) - 1)
        ax.plot(pts[:3, 0], pts[:3, 1], "-o", color="C0", alpha=a, ms=3)
        ax.plot(pts[2:6, 0], pts[2:6, 1], "-o", color="C1", alpha=a, ms=3)
        ax.plot(*pts[-1], "s", color="C3", alpha=a, ms=7)
    ax.plot(r.cop_x, np.zeros_like(r.cop_x), color="k", lw=4, alpha=0.4, label="center of pressure path")
    ax.axvspan(*lift.FOOT_X, ymax=0.03, color="g", alpha=0.3, label="foot")
    ax.set_aspect("equal")
    ax.set_title(f"{s}", fontsize=10)
    ax.set_xlim(-0.5, 0.8)
    ax.legend(fontsize=7, loc="upper left")
save(fig, "postures.png")

# ------------------------------------------------------------------ 2. gear-ratio trade at the hip
r = lifts["stoop"]
qdd = np.gradient(r.qd, r.t, axis=0)
hip = J.index("hip")
tau_h, qd_h, qdd_h = r.tau[:, hip] / 2, r.qd[:, hip], qdd[:, hip]
SPEED_FLOOR = 6.0  # rad/s at the hip for fast motions (step recovery, walking); an explicit assumption
fig, ax = plt.subplots(figsize=(7, 4.3))
ax2 = ax.twinx()
results["hip_ratio_trade"] = {}
for mot, c in zip(A.MOTORS, ("C0", "C1", "C2")):
    ratios, loss, ok = A.sweep_ratio(mot, tau_h, qd_h, qdd_h)
    n_speed = mot.w_max / SPEED_FLOOR * 0.8  # keep 20% margin for torque at that speed
    ax.plot(ratios, loss, color=c, label=f"{mot.name.split()[-1]}: Km {mot.km:.2f} N m/sqrt(W)")
    ax.axvline(n_speed, color=c, ls=":", lw=1)
    ax2.plot(ratios, mot.j_rotor * ratios**2, color=c, ls="--", lw=1)
    usable = ok & (ratios <= n_speed)
    if usable.any():
        k = np.argmin(np.where(usable, loss, np.inf))
        results["hip_ratio_trade"][mot.name] = {"ratio": round(float(ratios[k]), 1), "mean_copper_W": round(float(loss[k]), 2),
                                                "reflected_inertia_kgm2": round(float(mot.j_rotor * ratios[k] ** 2), 3)}
        ax.plot(ratios[k], loss[k], "o", color=c)
    else:
        results["hip_ratio_trade"][mot.name] = None
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("gear ratio N")
ax.set_ylabel("mean copper loss during lift (W, solid)")
ax2.set_ylabel("reflected rotor inertia N^2 J (kg m^2, dashed)")
ax2.set_yscale("log")
ax.set_title("Hip, one side: higher N cuts heat but stiffens and slows the joint\n(dotted = speed limit for a 6 rad/s requirement)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
save(fig, "hip_ratio_trade.png")

# ------------------------------------------------------------------ 3. tote shuttle: which joint overheats first?
mot = A.MOTORS[1]
choice = {"ankle": 120, "knee": 120, "hip": results["hip_ratio_trade"][mot.name]["ratio"], "shoulder": 120, "elbow": 120}
m_empty, d_empty = model.load(model.Body(tote_mass=0.0))
q_end = r.q[-1]
d.qpos[:], d.qvel[:], d.qacc[:] = q_end, 0, 0
mujoco.mj_inverse(m, d)
hold_tau = d.qfrc_inverse.copy() / 2
d_empty.qpos[:], d_empty.qvel[:], d_empty.qacc[:] = q_end * np.array([1, 1, 1, 0, 0]), 0, 0
mujoco.mj_inverse(m_empty, d_empty)
empty_tau = d_empty.qfrc_inverse.copy() / 2
lift_T = r.t[-1]
RATE = 120  # totes per hour: one every 30 s


def steady_T(j_idx, ratio, rate):
    cycle = 3600.0 / rate
    walk = max(cycle - 2 * lift_T, 0.0) / 2  # carry loaded half the remaining time, return empty the other half
    dr = A.Drive(mot, ratio)
    e_lift = np.trapezoid(dr.copper_loss(r.tau[:, j_idx] / 2, qdd[:, j_idx]), r.t)
    p_avg = (2 * e_lift + walk * dr.copper_loss(hold_tau[j_idx]) + walk * dr.copper_loss(empty_tau[j_idx])) / cycle
    return A.steady_winding_temperature(mot, p_avg)


ratios = np.geomspace(4, 120, 60)
fig, ax = plt.subplots(figsize=(7, 4.2))
results["min_ratio_for_120_totes_per_hour"] = {}
for i, j in enumerate(J):
    T = np.array([steady_T(i, n, RATE) for n in ratios])
    ax.plot(ratios, np.minimum(T, 400), label=j)
    ok = np.where(T <= mot.t_winding_max)[0]
    results["min_ratio_for_120_totes_per_hour"][j] = round(float(ratios[ok[0]]), 1) if len(ok) else None
ax.axhline(mot.t_winding_max, color="k", ls=":", lw=1)
ax.text(4.3, mot.t_winding_max + 4, "winding limit 120 C", fontsize=8)
ax.set_xscale("log")
ax.set_ylim(25, 250)
ax.set_xlabel("gear ratio N (same 'M' motor at every joint)")
ax.set_ylabel("steady winding temperature (C)")
ax.set_title(f"Tote shuttle at {RATE}/hour: below what gear ratio does each joint overheat?", fontsize=9)
ax.legend(fontsize=8)
save(fig, "shuttle_thermal.png")
results["hold_torque_per_side_Nm"] = dict(zip(J, np.round(hold_tau, 1).tolist()))

# ------------------------------------------------------------------ 4. series elastic actuator: impact
j_link = 1.5  # kg m^2, rough trunk + arms about the hip
K_ENV = 2e5  # N m/rad: a stiff obstacle (rigid shelf, floor)
ks = np.geomspace(100, 1e6, 50)
fig, ax = plt.subplots(figsize=(6.5, 4))
results["sea_impact"] = {}
for n_hip, c in ((20, "C0"), (50, "C1"), (120, "C2")):
    j_ref = mot.j_rotor * n_hip**2
    peaks = np.array([A.impact_peak_torque(j_link, j_ref, k, K_ENV, 1.0) for k in ks])
    ax.loglog(ks, peaks, color=c, label=f"N = {n_hip} (reflected rotor {j_ref:.2f} kg m^2)")
    results["sea_impact"][f"N{n_hip}"] = {"rigid_Nm": round(float(peaks[-1])), "k_3000_Nm": round(float(np.interp(3000, ks, peaks)))}
ax.set_xlabel("series stiffness k (N m/rad)")
ax.set_ylabel("peak gearbox torque (N m)")
ax.set_title("Hip hits a stiff obstacle at 1 rad/s: the spring protects the gearbox,\nand matters more the higher the gear ratio", fontsize=9)
ax.legend(fontsize=8)
save(fig, "sea_impact.png")
results["sea_static_deflection_rad_at_peak_hip_torque"] = {k: round(float(np.abs(tau_h).max() / k), 3) for k in (1000, 3000, 10000)}

# ------------------------------------------------------------------ 5. air core vs iron core
i = np.linspace(0, 120, 300)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for core in (A.IRON_CORE, A.AIR_CORE):
    axes[0].plot(i, core.torque(i), label=core.name)
axes[0].set_xlabel("current (A)")
axes[0].set_ylabel("motor torque (N m)")
axes[0].legend(fontsize=8)
w = np.linspace(0, 600, 200)
for core in (A.IRON_CORE, A.AIR_CORE):
    axes[1].plot(w * 60 / (2 * np.pi), core.iron_loss(w), label=core.name)
axes[1].set_xlabel("motor speed (rpm)")
axes[1].set_ylabel("core loss (W)")
fig.suptitle("Air-core trades low-current torque for linearity and zero core loss (illustrative parameters)", fontsize=10)
save(fig, "aircore_vs_iron.png")

# ------------------------------------------------------------------ 6. CAD
actuator_housing(ROOT / "cad" / "hip_actuator_envelope.step", motor_od_mm=90, stack_mm=35, gearbox_mm=40)
results["cad"] = "cad/hip_actuator_envelope.step"
(ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({k: v for k, v in results.items() if k != "lift_peaks_both_sides"}, indent=2, default=float))
