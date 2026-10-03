import mujoco
import numpy as np
import pytest

from actsizer import actuator as A
from actsizer import lift, model

G = 9.81


def test_static_shoulder_torque_matches_hand_calculation():
    """Arms straight out in front, trunk upright: shoulder torque = sum of m g x about the shoulder."""
    b = model.Body()
    m, d = model.load(b)
    d.qpos[:] = [0, 0, 0, -np.pi / 2, 0]
    d.qvel[:] = d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    Lu, mu, cu = b.seg(b.upper_arm)
    Lf, mf, cf = b.seg(b.forearm)
    grip = Lf + b.grip_offset
    # tote COM sits 0.10 m "below" the grip in the forearm frame, i.e. further forward when the arm is level
    moment = mu * cu + mf * (Lu + cf) + b.tote_mass * (Lu + grip + 0.10)
    assert abs(d.qfrc_inverse[3]) == pytest.approx(moment * G, rel=1e-6)


def test_standing_straight_needs_no_hip_or_knee_torque():
    m, d = model.load(model.Body(tote_mass=0.0))
    d.qpos[:] = d.qvel[:] = d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    assert np.allclose(d.qfrc_inverse, 0, atol=1e-9)


def test_min_jerk_boundary_conditions():
    q0, q1 = np.zeros(5), np.ones(5)
    t = np.array([0.0, 1.0])
    q, qd, qdd = lift.min_jerk(q0, q1, 1.0, t)
    assert np.allclose(q[0], q0) and np.allclose(q[-1], q1)
    assert np.allclose(qd, 0) and np.allclose(qdd, 0)


def test_lift_reaches_targets_and_stoop_is_balanced():
    r = lift.simulate_lift("stoop", duration=1.6)
    m, d = model.load()
    assert np.allclose(model.grip_position(m, d, r.q[0]), [0.35, 0.25], atol=0.02)
    assert np.allclose(model.grip_position(m, d, r.q[-1]), [0.30, 0.95], atol=0.02)
    assert r.balanced()


def test_lift_work_equals_potential_energy_gain():
    """Slow lift: net joint work = rise in potential energy (no friction, start and end at rest)."""
    r = lift.simulate_lift("stoop", duration=3.0, n=1201)
    m, d = model.load()

    def pe(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        return float((m.body_mass[1:] * d.xipos[1:, 2]).sum() * G)

    work = np.trapezoid(r.power.sum(axis=1), r.t)
    assert work == pytest.approx(pe(r.q[-1]) - pe(r.q[0]), rel=0.01)


def test_inertia_matched_ratio_minimizes_loss_for_pure_inertia():
    mot = A.MOTORS[1]
    J_load = 0.5
    t = np.linspace(0, 1, 400)
    qdd = np.sin(2 * np.pi * t) * 20
    tau = J_load * qdd
    ratios = np.geomspace(5, 300, 400)
    loss = [np.mean(A.Drive(mot, n, efficiency=1.0).copper_loss(tau, qdd)) for n in ratios]
    assert ratios[int(np.argmin(loss))] == pytest.approx(A.inertia_matched_ratio(J_load, mot.j_rotor), rel=0.03)


def test_steady_winding_temperature_matches_transient():
    mot = A.MOTORS[0]
    T = A.winding_temperature(mot, np.full(200000, 30.0), dt=0.5)
    assert T[-1] == pytest.approx(A.steady_winding_temperature(mot, 30.0), abs=0.2)


def test_rigid_impact_matches_closed_form():
    j_l, j_r, k_env, w0 = 1.5, 0.5, 2e5, 1.0
    peak = A.impact_peak_torque(j_l, j_r, 1e8, k_env, w0)
    # rigid: total inertia decelerated by k_env; rotor share of the contact torque
    expected = w0 * np.sqrt(k_env * (j_l + j_r)) * j_r / (j_l + j_r)
    assert peak == pytest.approx(expected, rel=0.05)


def test_feasibility_respects_speed_line():
    mot = A.MOTORS[1]
    dr = A.Drive(mot, 100)
    assert dr.feasible(np.array([10.0]), np.array([1.0]))
    assert not dr.feasible(np.array([10.0]), np.array([mot.w_max / 100 * 1.01]))


def test_air_core_is_linear_and_iron_core_saturates():
    i = np.array([10.0, 100.0])
    air, iron = A.AIR_CORE.torque(i), A.IRON_CORE.torque(i)
    assert air[1] / air[0] == pytest.approx(10.0)
    assert iron[1] / iron[0] < 8.0
