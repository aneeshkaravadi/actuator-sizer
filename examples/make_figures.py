"""Regenerate every figure and number in the README.   python examples/make_figures.py"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from actsizer import actuator as A
from actsizer import asym, gait, lift, model
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
def balanced_quasi_static(r):
    """My first balance check: the floor force taken as the weight and the ankle height left out."""
    cop = -r.tau[:, 0] / (model.load()[0].body_mass.sum() * 9.81)
    return bool(np.all((cop > lift.FOOT_X[0]) & (cop < lift.FOOT_X[1])))


def fastest_balanced(style, lo=0.6, hi=8.0, check=lambda r: r.balanced()):
    if not check(lift.simulate_lift(style, duration=hi)):
        return None
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if check(lift.simulate_lift(style, duration=mid)) else (mid, hi)
    return hi


results["fastest_balanced_lift_s"] = {s: fastest_balanced(s) for s in ("squat", "stoop")}
results["fastest_balanced_lift_s_with_my_first_cop"] = {s: fastest_balanced(s, check=balanced_quasi_static) for s in ("squat", "stoop")}
lifts = {s: lift.simulate_lift(s, duration=round(results["fastest_balanced_lift_s"][s] + 0.05, 2)) for s in ("squat", "stoop")}
results["lift_peaks_both_sides"] = {s: r.peak() for s, r in lifts.items()}
M_ROBOT_G = model.load()[0].body_mass.sum() * 9.81
results["analyzed_lifts"] = {s: {"duration_s": round(float(r.t[-1]), 2), "balanced": r.balanced(),
                                 "cop_range_m": [round(float(r.cop_x.min()), 3), round(float(r.cop_x.max()), 3)],
                                 "floor_force_over_weight": [round(float(r.f_floor[:, 1].min() / M_ROBOT_G), 2),
                                                             round(float(r.f_floor[:, 1].max() / M_ROBOT_G), 2)],
                                 "largest_cop_change_from_my_first_formula_mm": round(float(1000 * np.abs(
                                     r.cop_x + r.tau[:, 0] / M_ROBOT_G).max()), 1),
                                 "hip_peak_per_side_Nm": round(float(np.abs(r.tau[:, 2]).max() / 2), 1),
                                 "hip_peak_power_per_side_W": round(float(np.abs(r.power[:, 2]).max() / 2))}
                             for s, r in lifts.items()}
# squat against stoop at the same duration, so speed doesn't muddy the comparison
same = {}
for T in (lifts["stoop"].t[-1], lifts["squat"].t[-1]):
    pk = {s: float(np.abs(lift.simulate_lift(s, duration=float(T)).tau[:, 2]).max()) for s in ("squat", "stoop")}
    same[f"{T:.2f} s"] = round(pk["squat"] / pk["stoop"], 2)
results["squat_over_stoop_hip_peak_at_equal_duration"] = same

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
quick = R.required_hip_speed(pend, DESIGN_PUSH, 0.5, swing_times=np.linspace(0.03, 0.8, 1541))
results["push_recovery"]["half_meter_step_if_swings_under_80_ms_were_allowed"] = {
    "hip_speed_rad_s": round(quick[0], 2), "swing_s": round(quick[1], 3), "min_swing_assumed_s": R.MIN_SWING}
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
    ax.plot(ratios, loss, color=c, label=f"{' '.join(mot.name.split()[-3:])}: Km {mot.km:.2f} N m/sqrt(W)")
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
    axes[0].plot(ns, copper, color=c, label=f"{gb.name.split(' (')[0]}: winding (copper)")
    axes[0].plot(ns, total, color=c, ls="--", label=f"{gb.name.split(' (')[0]}: winding + gearbox")
    most = np.where(gear > copper)[0]
    results["gearbox"][gb.name] = {"sweep": [round(float(ns[0]), 1), round(float(ns[-1]), 1)],
                                   "total_W_at_sweep_end": round(float(total[-1]), 2),
                                   "total_still_falling_at_sweep_end": bool(total[-1] < total[-2]),
                                   "gearbox_makes_most_of_the_heat_from": round(float(ns[most[0]]), 1) if len(most) else None}
for n_step in (100.0, 101.0):
    dr = A.Drive(mot_m, n_step, gearbox=A.PLANETARY)
    results["gearbox"][A.PLANETARY.name][f"total_W_at_{n_step:.0f}_to_1"] = round(
        float(np.mean(dr.copper_loss(tau_h, qdd_h, qd_h)) + np.mean(dr.gearbox_loss(tau_h, qd_h))), 2)
for gb in (A.PLANETARY, A.STRAIN_WAVE):
    dr = A.Drive(mot_m, 50.0, gearbox=gb)
    results["gearbox"][gb.name]["at_50_to_1"] = {"copper_W": round(float(np.mean(dr.copper_loss(tau_h, qdd_h, qd_h))), 2),
                                                 "gearbox_W": round(float(np.mean(dr.gearbox_loss(tau_h, qd_h))), 2),
                                                 "efficiency": round(gb.efficiency(50.0), 3)}
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("gear ratio N")
axes[0].set_ylabel("mean heat during the lift, hip, one side (W)")
axes[0].set_title("planetary efficiency steps down at 10:1 and 100:1; counting the gearbox's own heat,\n"
                  "the total jumps at each new stage but keeps falling", fontsize=9)
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
fig.suptitle(f"Hip, {mot_m.name}, stoop lift: gearbox efficiency and backlash against ratio", fontsize=10)
save(fig, "gearbox_tradeoffs.png")
results["gearbox"]["hip_reach_at_pick_m"] = round(reach, 3)
results["gearbox"]["hand_play_mm_planetary_at_50_to_1"] = round(float(np.radians(A.PLANETARY.backlash_arcmin(50) / 60) * reach * 1000), 1)

# ------------------------------------------------------------------ 2c. do maxon's thermal numbers hang together?
# maxon's nominal current is the continuous current that takes the winding to its limit at 25 C ambient,
# at the nominal speed. Iron losses (maxon: a friction torque of Kt x no-load current) are put through the
# whole thermal resistance like copper; the last column puts them on the housing only, as a bracket.
results["maxon_rated_point_winding_C"] = {}
for dsheet in A.MAXON_HT:
    mm_ = dsheet.motor()
    p_cu = dsheet.nominal_current**2 * mm_.r
    p_fe = mm_.t_iron * dsheet.nominal_speed_rpm * 2 * np.pi / 60
    a_cu = 0.00393
    housing_only = 25 + (mm_.r_th * p_cu + dsheet.r_th_housing_ambient * p_fe) / (1 - mm_.r_th * p_cu * a_cu)
    results["maxon_rated_point_winding_C"][dsheet.name] = {
        "copper_only": round(dsheet.rated_winding_temperature(iron=False), 1),
        "with_iron_loss": round(dsheet.rated_winding_temperature(), 1),
        "iron_loss_on_housing_only": round(float(housing_only), 1),
        "copper_W": round(p_cu, 1), "iron_W": round(p_fe, 1), "maxon_limit": dsheet.max_winding_temp}

# ------------------------------------------------------------------ 3. tote shuttle: which joint overheats first?
# One cycle: lift, carry the tote, lower it, walk back empty. The carry and the walk back are the standing
# posture's torques (from MuJoCo) plus, at the leg joints, a human stride's swing in moment (Fukuchi et al.
# 2018 gait data, scaled to the mass the legs carry; see gait.py). walking=False is my first version,
# where both walks were just standing still.
mot = A.MOTORS[1]
STRIDE = gait.load_stride()
LEGS = ("ankle", "knee", "hip")
BODY = model.Body()
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
WALK_T = (3600.0 / RATE - 2 * lift_T) / 2  # each way


@functools.lru_cache(maxsize=None)
def walk_load(j_idx, mass, scale=1.0):
    """(swing torque, speed, acceleration) over one stride at a leg joint, per side; None for the arms."""
    if J[j_idx] not in LEGS:
        return None
    _, tau, qd, qdd = STRIDE.joint(J[j_idx], mass)
    return scale * tau, qd, qdd


def walk_scale(j_idx, which):
    """Scale the mean stride's swing to the smallest or largest person's (per-person rms over mean-curve rms)."""
    if J[j_idx] not in LEGS or which == "mean":
        return 1.0
    lo, _, hi = STRIDE.zero_mean_rms_range[J[j_idx]]
    _, tau, _, _ = STRIDE.joint(J[j_idx], 1.0)
    return (lo if which == "min" else hi) / float(np.sqrt(np.mean(tau**2)))


def cycle_parts(j_idx, ratio, motor=mot, walking=True, kt_drop=0.0, people="mean"):
    """Average copper and iron loss (W) in each phase, and whether every point is within the motor's limits."""
    dr = A.Drive(motor, ratio, gearbox=A.PLANETARY, kt_drop=kt_drop)
    tau_j, qd_j, qdd_j = r.tau[:, j_idx] / 2, r.qd[:, j_idx], qdd[:, j_idx]
    parts = {"lift": (np.trapezoid(dr.copper_loss(tau_j, qdd_j, qd_j), r.t) / lift_T,
                      np.trapezoid(dr.iron_loss(qd_j), r.t) / lift_T),
             # Lowering is the lift played backwards: with no damping the torques are the same at each
             # posture, but the speeds flip sign, so the load drives the gearbox and friction helps.
             "lower": (np.trapezoid(dr.copper_loss(tau_j, qdd_j, -qd_j), r.t) / lift_T,
                       np.trapezoid(dr.iron_loss(qd_j), r.t) / lift_T)}
    ok = dr.feasible(tau_j, qd_j, qdd_j)
    for phase, hold, mass in (("carry", hold_tau[j_idx], BODY.mass + BODY.tote_mass), ("walk back", empty_tau[j_idx], BODY.mass)):
        w = walk_load(j_idx, mass, walk_scale(j_idx, people)) if walking else None
        if w is None:
            parts[phase] = (float(dr.copper_loss(hold)), 0.0)
        else:
            tau_w, qd_w, qdd_w = w
            parts[phase] = (float(np.mean(dr.copper_loss(hold + tau_w, qdd_w, qd_w))), float(np.mean(dr.iron_loss(qd_w))))
            ok = ok and dr.feasible(hold + tau_w, qd_w, qdd_w)
    return parts, ok


def evaluate(j_idx, ratio, motor=mot, **kw):
    """Steady winding temperature (C) over the cycle, and whether the drive can do every move."""
    parts, ok = cycle_parts(j_idx, ratio, motor, **kw)
    dur = {"lift": lift_T, "lower": lift_T, "carry": WALK_T, "walk back": WALK_T}
    cycle = sum(dur.values())
    p_cu = sum(parts[k][0] * dur[k] for k in dur) / cycle
    p_fe = sum(parts[k][1] * dur[k] for k in dur) / cycle
    return A.steady_winding_temperature(motor, p_cu, p_other_w=p_fe), ok


def steady_T(j_idx, ratio, motor=mot, **kw):
    return evaluate(j_idx, ratio, motor, **kw)[0]


LIMIT_GRID = np.geomspace(4.0, 300.0, 160)


def first_crossing(ok, grid_ok):
    """Smallest ratio where ok(N) holds, from a scan on LIMIT_GRID refined by bisection."""
    if not grid_ok.any():
        return None
    k = int(np.argmax(grid_ok))
    if k == 0:
        return float(LIMIT_GRID[0])
    a, b = LIMIT_GRID[k - 1], LIMIT_GRID[k]
    for _ in range(30):
        mid = 0.5 * (a + b)
        a, b = (a, mid) if ok(mid) else (mid, b)
    return float(b)


def min_ratio(ok):
    return first_crossing(ok, np.array([ok(x) for x in LIMIT_GRID]))


def joint_limit(j_idx, motor=mot, **kw):
    """Minimum ratio for the joint, which limit sets it, and the heat-only and torque/speed-only minimums."""
    ev = [evaluate(j_idx, x, motor, **kw) for x in LIMIT_GRID]
    cool = np.array([T <= motor.t_winding_max for T, _ in ev])
    able = np.array([ok for _, ok in ev])
    heat = first_crossing(lambda x: evaluate(j_idx, x, motor, **kw)[0] <= motor.t_winding_max, cool)
    drive = first_crossing(lambda x: evaluate(j_idx, x, motor, **kw)[1], able)
    both = first_crossing(lambda x: (lambda T, ok: T <= motor.t_winding_max and ok)(*evaluate(j_idx, x, motor, **kw)), cool & able)
    binding = "heat" if heat is not None and both is not None and abs(both - heat) < 1e-6 * both else "torque or speed"
    return both, binding, heat, drive


ratios = np.geomspace(4, 120, 120)
fig, ax = plt.subplots(figsize=(7, 4.2))
results["shuttle"] = {"rate_per_hour": RATE, "lift_s": round(float(lift_T), 2), "walk_each_way_s": round(float(WALK_T), 2),
                      "stride_s": STRIDE.stride_time, "mass_carried_kg": BODY.mass + BODY.tote_mass}
squat_cycle = 2 * lifts["squat"].t[-1] + 2 * WALK_T  # same walks, squat lifts instead of stoops
results["shuttle"]["cycle_with_squat_lifts_s"] = round(float(squat_cycle), 1)
results["shuttle"]["totes_per_hour_with_squat_lifts"] = round(float(3600 / squat_cycle))
results["min_ratio_for_120_totes_per_hour"] = {}
results["min_ratio_for_120_totes_per_hour_walking_as_standing"] = {}
for i, j in enumerate(J):
    T = np.array([steady_T(i, n) for n in ratios])
    T0 = np.array([steady_T(i, n, walking=False) for n in ratios])
    line, = ax.plot(ratios, np.minimum(T, 400), label=j)
    if j in LEGS:
        ax.plot(ratios, np.minimum(T0, 400), color=line.get_color(), ls=":", lw=1)
    results["min_ratio_for_120_totes_per_hour"][j] = round(joint_limit(i)[0], 1)
    results["min_ratio_for_120_totes_per_hour_walking_as_standing"][j] = round(joint_limit(i, walking=False)[0], 1)
ax.axhline(mot.t_winding_max, color="k", ls=":", lw=1)
ax.text(4.3, mot.t_winding_max + 4, f"winding limit {mot.t_winding_max:.0f} C", fontsize=8)
ax.set_xscale("log")
ax.set_ylim(25, 250)
ax.set_xlabel(f"gear ratio N (the same {mot.name} at every joint)")
ax.set_ylabel("steady winding temperature (C)")
ax.set_title(f"Tote shuttle at {RATE}/hour, walking loads from human gait data (dotted: walking treated as standing)\n"
             "below what gear ratio does each joint overheat?", fontsize=9)
ax.legend(fontsize=8)
save(fig, "shuttle_thermal.png")
results["hold_torque_per_side_Nm"] = dict(zip(J, np.round(hold_tau, 1).tolist()))

# hip heat budget per cycle (J), HT 76 M at its minimum ratio
hip_i = J.index("hip")
n_hip = results["min_ratio_for_120_totes_per_hour"]["hip"]
parts, _ = cycle_parts(hip_i, n_hip)
dur = {"lift": lift_T, "lower": lift_T, "carry": WALK_T, "walk back": WALK_T}
results["hip_heat_per_cycle_J"] = {"ratio": n_hip, **{k: round(float((parts[k][0] + parts[k][1]) * dur[k])) for k in dur}}
parts0, _ = cycle_parts(hip_i, n_hip, walking=False)
results["hip_heat_per_cycle_J_walking_as_standing"] = {k: round(float((parts0[k][0] + parts0[k][1]) * dur[k])) for k in dur}

# every kit, both mountings maxon gives, walking in or not; and which limit sets the hip's minimum
results["shuttle_min_ratio_by_motor"] = {}
results["hip_window_by_motor"] = {}
for dsheet in A.MAXON_HT:
    row = {}
    for mounting, label in (("plate", "rated mounting (housing on a plate)"), ("housing", "housing, no plate")):
        motor = dsheet.motor(mounting)
        cases = {"walking": {}, "walking as standing": {}}
        for i, j in enumerate(J):
            lim = joint_limit(i, motor)
            cases["walking"][j] = round(lim[0], 1) if lim[0] else None
            cases["walking as standing"][j] = round(joint_limit(i, motor, walking=False)[0], 1)
            if j == "hip":
                both, binding, heat, drive = lim
        cases["hip limit set by"] = binding
        cases["hip heat-only minimum"] = round(heat, 1)
        cases["hip torque-and-speed minimum"] = round(drive, 1)
        row[label] = cases
        n_max = motor.w_max / SPEED_FLOOR * 0.8
        results.setdefault("hip_max_ratio_for_push_recovery", {})[f"maxon {dsheet.name}"] = round(n_max, 1)
        hi_ok = both is not None and both <= n_max
        results["hip_window_by_motor"].setdefault(f"maxon {dsheet.name} ({dsheet.weight_g:.0f} g)", {})[label] = (
            [round(both, 1), round(n_max, 1)] if hi_ok else None)
    results["shuttle_min_ratio_by_motor"][f"maxon {dsheet.name} ({dsheet.weight_g:.0f} g)"] = row

# sensitivities for the HT 76 M hip: people who walk with smaller or bigger swings, and torque per amp falling at high torque
results["hip_min_ratio_sensitivity_HT76M"] = {
    "gait swing of the smallest / mean / largest person": [round(joint_limit(hip_i, people=w)[0], 1) for w in ("min", "mean", "max")],
    "torque per amp falling 0 / 20 / 40% from 2x nominal torque to stall": [round(joint_limit(hip_i, kt_drop=k)[0], 1) for k in (0.0, 0.2, 0.4)],
}
tm = np.abs(A.Drive(mot, n_hip, gearbox=A.PLANETARY).motor_torque(r.tau[:, hip_i] / 2, qdd[:, hip_i]))
results["hip_motor_torque_at_min_ratio"] = {"peak_over_nominal": round(float(tm.max() / mot.t_nominal), 2),
                                            "peak_over_stall": round(float(tm.max() / (mot.kt * mot.i_max)), 2)}

# ------------------------------------------------------------------ 3b. the winding's ripple within each cycle
# The steady analysis above uses each cycle's average loss. With the two-node model the winding (time
# constant under a minute) swings within every 30 s cycle, so check the peak against the limit too.
# Within a walk the losses are averaged over each 1.06 s stride, 50 times shorter than the winding's time constant.
DT_TH = 0.05


def cycle_loss(ratio, j_idx=hip_i):
    """Copper and iron loss through one cycle, sampled every DT_TH."""
    dr = A.Drive(mot, ratio, gearbox=A.PLANETARY)
    tau_j, qd_j, qdd_j = r.tau[:, j_idx] / 2, r.qd[:, j_idx], qdd[:, j_idx]
    t_move = np.arange(0.0, lift_T, DT_TH)
    n_walk = round(WALK_T / DT_TH)
    parts, _ = cycle_parts(j_idx, ratio)
    cu = np.concatenate([np.interp(t_move, r.t, dr.copper_loss(tau_j, qdd_j, qd_j)), np.full(n_walk, parts["carry"][0]),
                         np.interp(t_move, r.t, dr.copper_loss(tau_j, qdd_j, -qd_j)), np.full(n_walk, parts["walk back"][0])])
    fe = np.concatenate([np.interp(t_move, r.t, dr.iron_loss(qd_j)), np.full(n_walk, parts["carry"][1]),
                         np.interp(t_move, r.t, dr.iron_loss(qd_j)), np.full(n_walk, parts["walk back"][1])])
    return cu, fe


def last_cycle(ratio, cycles=300):
    cu, fe = cycle_loss(ratio)
    return A.winding_temperature(mot, np.tile(cu, cycles), DT_TH, p_other_w=np.tile(fe, cycles))[-len(cu):]


lo, hi = 20.0, 120.0  # hip ratio where the PEAK winding temperature reaches the limit
for _ in range(30):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if last_cycle(mid).max() > mot.t_winding_max else (lo, mid)
peak_limit = hi
avg_limit = min_ratio(lambda x: steady_T(hip_i, x) <= mot.t_winding_max)
trace = last_cycle(peak_limit)
results["hip_thermal_ripple"] = {"motor": mot.name, "limit_from_average": round(avg_limit, 1), "limit_from_peak": round(peak_limit, 1),
                                 "swing_K_at_peak_limit": round(float(trace.max() - trace.min()), 1),
                                 "peak_above_cycle_mean_K": round(float(trace.max() - trace.mean()), 1),
                                 "hottest_moment_s_into_cycle": round(float(np.argmax(trace) * DT_TH), 1)}
fig, ax = plt.subplots(figsize=(7, 3.6))
tt = np.arange(len(trace)) * DT_TH
ax.plot(tt, trace, color="C2")
ax.axhline(mot.t_winding_max, color="k", ls=":", lw=1)
ax.axhline(trace.mean(), color="C2", ls="--", lw=1, label=f"cycle mean {trace.mean():.0f} C")
phases = ((0.0, "lift"), (lift_T, "carry"), (lift_T + WALK_T, "lower"), (2 * lift_T + WALK_T, "walk back"))
for k, (start, label) in enumerate(phases):
    ax.axvline(start, color="gray", lw=0.6)
    ax.text(start + 0.3, trace.min() + (0.5 if k % 2 == 0 else 2.0), label, fontsize=7, color="gray")
ax.set_xlabel(f"time within one {3600 / RATE:.0f} s cycle (s)")
ax.set_ylabel("hip winding temperature (C)")
ax.set_title(f"{mot.name}, hip at {peak_limit:.1f}:1 after hours of shuttling: the winding swings "
             f"{trace.max() - trace.min():.0f} K\nwithin every cycle, so its peak sets the limit, not its average", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
save(fig, "hip_winding_cycle.png")

# ------------------------------------------------------------------ 3c. one-handed and twisting lifts
T_LIFT = round(results["fastest_balanced_lift_s"]["stoop"] + 0.05, 2)
two = lifts["stoop"]
one = asym.simulate_asym_lift((0.35, asym.SHOULDER_Y, 0.25), (0.30, asym.SHOULDER_Y, 0.95), duration=T_LIFT)
twist = asym.simulate_asym_lift((0.10, 0.38, 0.30), (0.30, asym.SHOULDER_Y, 0.95), duration=T_LIFT)
two_pk = {j: float(np.abs(two.tau[:, i]).max()) for i, j in enumerate(J)}
per_actuator = {  # per actuator: legs and hip pitch shared by two sides, the loaded arm alone
    "two-handed": {"hip pitch": two_pk["hip"] / 2, "hip roll": 0.0, "trunk twist": 0.0,
                   "shoulder": two_pk["shoulder"] / 2, "elbow": two_pk["elbow"] / 2},
}
for name, lf in (("one-handed", one), ("one-handed, from the side", twist)):
    pk = lf.peak()
    per_actuator[name] = {"hip pitch": pk["hip"] / 2, "hip roll": pk["hip_roll"], "trunk twist": pk["trunk_yaw"],
                          "shoulder": pk["shoulder"], "elbow": pk["elbow"]}
results["asymmetric_lifts_peak_Nm"] = {k: {j: round(v, 1) for j, v in row.items()} for k, row in per_actuator.items()}
results["asymmetric_lifts_note"] = "hip roll and trunk twist are both sides together; the rest per actuator"
TWIST_DEG = float(np.degrees(abs(twist.q[0, asym.JOINTS_ASYM.index("trunk_yaw")])))
results["side_pick_trunk_twist_deg"] = round(TWIST_DEG)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
jn = list(per_actuator["two-handed"])
xx = np.arange(len(jn))
for k, (name, row) in enumerate(per_actuator.items()):
    axes[0].bar(xx + (k - 1) * 0.27, [row[j] for j in jn], 0.27, label=name)
axes[0].set_xticks(xx, jn, fontsize=8)
axes[0].set_ylabel("peak torque (N m)")
axes[0].set_title("25 kg tote, same pick and place heights", fontsize=9)
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)
for j, c in (("hip_roll", "C1"), ("trunk_yaw", "C2"), ("shoulder", "C0")):
    i = asym.JOINTS_ASYM.index(j)
    axes[1].plot(twist.t, twist.tau[:, i], color=c, label=j.replace("_", " ").replace("yaw", "twist"))
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("joint torque (N m)")
axes[1].set_title(f"one-handed, picked from the side ({TWIST_DEG:.0f} deg of trunk twist)", fontsize=9)
axes[1].legend(fontsize=8)
fig.suptitle("Lifting with one hand doubles the loaded arm's torque and adds sideways and twisting loads", fontsize=10)
save(fig, "asymmetric_lifts.png")

# ------------------------------------------------------------------ 4. series elastic actuator: impact
# The link is everything above the hip: the hip's own entry in the model's mass matrix (the other joints
# held), per side, from inverse dynamics with a unit hip acceleration minus the same with none.


def hip_link_inertia(mm, dd, q):
    f = []
    for acc in (0.0, 1.0):
        dd.qpos[:], dd.qvel[:], dd.qacc[:] = q, 0.0, 0.0
        dd.qacc[J.index("hip")] = acc
        mujoco.mj_inverse(mm, dd)
        f.append(dd.qfrc_inverse[J.index("hip")])
    return float(f[1] - f[0]) / 2


link_by_posture = {"standing, no tote": hip_link_inertia(m_empty, d_empty, np.zeros(5)),
                   "carrying the tote (end of the lift)": hip_link_inertia(m, d, r.q[-1]),
                   "stooped over the tote (start of the lift)": hip_link_inertia(m, d, r.q[0])}
j_link = link_by_posture["carrying the tote (end of the lift)"]  # walking into a shelf with the tote
K_ENV = 2e5  # N m/rad: a stiff obstacle (rigid shelf, floor)
ks = np.geomspace(100, 1e6, 50)
fig, ax = plt.subplots(figsize=(6.5, 4))
results["sea_impact"] = {"link_inertia_per_side_kgm2": {k: round(v, 2) for k, v in link_by_posture.items()},
                         "figure_uses": "carrying the tote (end of the lift)"}
for n_hip, c in ((20, "C0"), (50, "C1"), (120, "C2")):
    j_ref = mot.j_rotor * n_hip**2
    peaks = np.array([A.impact_peak_torque(j_link, j_ref, k, K_ENV, 1.0) for k in ks])
    rigid = 1.0 * j_ref * np.sqrt(K_ENV / (j_link + j_ref))  # no spring at all (DERIVATIONS section 6)
    ax.loglog(ks, peaks, color=c, label=f"N = {n_hip} (reflected rotor {j_ref:.2f} kg m^2)")
    ax.axhline(rigid, color=c, ls=":", lw=1)
    worse = np.where(peaks >= rigid)[0]
    results["sea_impact"][f"N{n_hip}"] = {"rigid_Nm": round(float(rigid)), "k_1e6_Nm": round(float(peaks[-1])),
                                          "k_3000_Nm": round(A.impact_peak_torque(j_link, j_ref, 3000.0, K_ENV, 1.0)),
                                          "spring_helps_only_below_k": round(float(ks[worse[0]]), -1) if len(worse) else None,
                                          "worst_k": round(float(ks[np.argmax(peaks)]), -2), "worst_Nm": round(float(peaks.max()))}
    for name, jl in link_by_posture.items():
        results["sea_impact"].setdefault("rigid_vs_k3000_Nm_by_posture", {}).setdefault(name, {})[f"N{n_hip}"] = [
            round(float(j_ref * np.sqrt(K_ENV / (jl + j_ref)))), round(A.impact_peak_torque(jl, j_ref, 3000.0, K_ENV, 1.0))]
ax.set_xlabel("series stiffness k (N m/rad)")
ax.set_ylabel("peak gearbox torque (N m)")
ax.set_title("Hip, carrying the tote, hits a stiff obstacle at 1 rad/s: only a soft spring protects the gearbox,\n"
             "and a medium one makes the shock worse than none (dotted: no spring)", fontsize=9)
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
