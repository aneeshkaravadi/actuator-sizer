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
from actsizer import recovery as R
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

# ------------------------------------------------------------------ 1b. where the hip speed requirement comes from
pend = R.pendulum_from_model()
DESIGN_PUSH, DESIGN_STEP = 1.5, 0.6  # m/s of center-of-mass speed from the push; longest recovery step (m)
pushes = np.linspace(0.3, 1.8, 61)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
results["push_recovery"] = {"com_height_m": round(pend.z0, 3), "hip_height_m": round(pend.leg, 3),
                            "largest_push_ankles_alone_m_s": round(pend.toe * pend.w, 2), "by_step_cap": {}}
for cap, c in ((0.5, "C3"), (0.6, "C1"), (0.8, "C2"), (np.inf, "C0")):
    sp = np.array([R.required_hip_speed(pend, v, cap)[0] for v in pushes])
    label = "any step the leg can reach" if np.isinf(cap) else f"step at most {cap:.1f} m"
    ax.plot(pushes, np.where(np.isfinite(sp), sp, np.nan), color=c, label=label)
    results["push_recovery"]["by_step_cap"][label] = {f"{v:.2f}": round(float(R.required_hip_speed(pend, v, cap)[0]), 2)
                                                      for v in (1.0, 1.25, 1.5)}
SPEED_FLOOR, t_swing, step_len = R.required_hip_speed(pend, DESIGN_PUSH, DESIGN_STEP)
results["push_recovery"]["design_case"] = {"push_m_s": DESIGN_PUSH, "max_step_m": DESIGN_STEP,
                                           "impulse_Ns": round(DESIGN_PUSH * float(model.load(model.Body(tote_mass=0.0))[0].body_mass[1:].sum())),
                                           "hip_speed_rad_s": round(SPEED_FLOOR, 2), "swing_s": round(t_swing, 3),
                                           "step_m": round(step_len, 3)}
ax.axvline(pend.toe * pend.w, color="k", ls=":", lw=1)
ax.text(pend.toe * pend.w + 0.02, 0.3, "ankles alone\ncan stop it", fontsize=8)
ax.plot(DESIGN_PUSH, SPEED_FLOOR, "ko")
ax.annotate(f"design case: {SPEED_FLOOR:.1f} rad/s", (DESIGN_PUSH, SPEED_FLOOR), xytext=(0.75, 6.2), fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8})
ax.set_ylim(0, 8)
ax.set_xlabel("center-of-mass speed right after the push (m/s)")
ax.set_ylabel("hip speed needed for one recovery step (rad/s)")
ax.set_title("One-step push recovery (linear inverted pendulum, capture point):\nthe shorter the step has to be, the faster the hip", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
save(fig, "push_recovery.png")

# ------------------------------------------------------------------ 2. gear-ratio trade at the hip
r = lifts["stoop"]
qdd = np.gradient(r.qd, r.t, axis=0)
hip = J.index("hip")
tau_h, qd_h, qdd_h = r.tau[:, hip] / 2, r.qd[:, hip], qdd[:, hip]
fig, ax = plt.subplots(figsize=(7, 4.3))
ax2 = ax.twinx()
results["hip_ratio_trade"] = {}
for mot, c in zip(A.MOTORS, ("C0", "C1", "C2")):
    ratios, loss, ok = A.sweep_ratio(mot, tau_h, qd_h, qdd_h, gearbox=A.PLANETARY)
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
ax.set_title(f"Hip, one side, planetary gearbox: higher N cuts heat but stiffens and slows the joint\n"
             f"(dotted = speed limit for the {SPEED_FLOOR:.1f} rad/s a push-recovery step needs)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
save(fig, "hip_ratio_trade.png")

# ------------------------------------------------------------------ 2b. gearbox efficiency: winding heat vs total heat
mot_m = A.MOTORS[1]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
results["gearbox"] = {}
for gb, c in ((A.PLANETARY, "C0"), (A.STRAIN_WAVE, "C3")):
    lo, hi = gb.ratio_range
    ns = np.geomspace(max(5, lo), min(300, hi), 160)
    copper, gear = [], []
    for n in ns:
        dr = A.Drive(mot_m, n, gearbox=gb)
        copper.append(np.mean(dr.copper_loss(tau_h, qdd_h, qd_h)))
        gear.append(np.mean(dr.gearbox_loss(tau_h, qd_h)))
    copper, gear = np.array(copper), np.array(gear)
    total = copper + gear
    k = int(np.argmin(total))
    axes[0].plot(ns, copper, color=c, label=f"{gb.name.split(' (')[0]}: winding (copper)")
    axes[0].plot(ns, total, color=c, ls="--", label=f"{gb.name.split(' (')[0]}: winding + gearbox")
    axes[0].plot(ns[k], total[k], "o", color=c)
    results["gearbox"][gb.name] = {"min_total_heat_ratio": round(float(ns[k]), 1), "min_total_heat_W": round(float(total[k]), 2),
                                   "copper_W_at_that_ratio": round(float(copper[k]), 2), "gearbox_W_at_that_ratio": round(float(gear[k]), 2)}
for gb in (A.PLANETARY, A.STRAIN_WAVE):
    dr = A.Drive(mot_m, 50.0, gearbox=gb)
    results["gearbox"][gb.name]["at_50_to_1"] = {"copper_W": round(float(np.mean(dr.copper_loss(tau_h, qdd_h, qd_h))), 2),
                                                 "gearbox_W": round(float(np.mean(dr.gearbox_loss(tau_h, qd_h))), 2),
                                                 "efficiency": round(gb.efficiency(50.0), 3)}
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("gear ratio N")
axes[0].set_ylabel("mean heat during the lift, hip, one side (W)")
axes[0].set_title("planetary efficiency steps down at 10:1 and 100:1;\ncounting the gearbox's own heat, the total stops falling past about 100:1", fontsize=9)
axes[0].legend(fontsize=7)
ns = np.geomspace(5, 300, 300)
d.qpos[:] = r.q[0]  # hip-to-hands distance in the pick posture, where a stoop reaches furthest
mujoco.mj_kinematics(m, d)
reach = float(np.linalg.norm(d.site("grip").xpos[[0, 2]] - d.body("trunk").xpos[[0, 2]]))
play = np.array([A.PLANETARY.backlash_arcmin(n) for n in ns])
axes[1].plot(ns, play, color="C0", label="planetary (10 arcmin per stage, assumed)")
axes[1].axhline(0.0, color="C3", label="strain wave (essentially none)")
axes[1].set_xscale("log")
axes[1].set_xlabel("gear ratio N")
axes[1].set_ylabel("backlash at the joint (arcmin)")
ax2 = axes[1].twinx()
ax2.set_ylabel(f"hand play from hip backlash at {reach:.2f} m reach (mm)")
axes[1].set_ylim(-0.5, 15)
ax2.set_ylim(np.radians(-0.5 / 60) * reach * 1000, np.radians(15 / 60) * reach * 1000)
axes[1].set_title("backlash is set by the last stage, so it barely grows with N", fontsize=9)
axes[1].legend(fontsize=7, loc="lower right")
fig.suptitle("Hip, 'M' motor, stoop lift: gearbox efficiency and backlash against ratio", fontsize=10)
save(fig, "gearbox_tradeoffs.png")
results["gearbox"]["hip_reach_at_pick_m"] = round(reach, 3)
results["gearbox"]["hand_play_mm_planetary_at_50_to_1"] = round(float(np.radians(A.PLANETARY.backlash_arcmin(50) / 60) * reach * 1000), 1)

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
    dr = A.Drive(mot, ratio, gearbox=A.PLANETARY)
    tau_j, qd_j, qdd_j = r.tau[:, j_idx] / 2, r.qd[:, j_idx], qdd[:, j_idx]
    e_lift = np.trapezoid(dr.copper_loss(tau_j, qdd_j, qd_j), r.t)
    # Lowering is the lift played backwards: with no damping the torques are the same at each posture,
    # but the speeds flip sign, so the load drives the gearbox and friction helps.
    e_lower = np.trapezoid(dr.copper_loss(tau_j, qdd_j, -qd_j), r.t)
    p_avg = (e_lift + e_lower + walk * dr.copper_loss(hold_tau[j_idx]) + walk * dr.copper_loss(empty_tau[j_idx])) / cycle
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
