import mujoco
import numpy as np
import pytest

from actsizer import actuator as A
from actsizer import lift, model
from actsizer import recovery as R

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


def test_planetary_efficiency_steps_down_with_each_stage():
    pg = A.PLANETARY
    assert [pg.stages(n) for n in (5, 10, 10.5, 100, 101)] == [1, 1, 2, 2, 3]
    assert pg.efficiency(50) == pytest.approx(0.97**2)
    assert A.STRAIN_WAVE.stages(30) == A.STRAIN_WAVE.stages(160) == 1
    # two 10:1 stages: the first stage's play reaches the output divided by 10
    assert pg.backlash_arcmin(100) == pytest.approx(10 + 10 / 10)


def test_backdriven_efficiency_and_self_locking():
    assert A.backdriven_efficiency(0.9) == pytest.approx(2 - 1 / 0.9)
    assert A.backdriven_efficiency(0.5) == pytest.approx(0.0)  # below this the gearbox self-locks
    dr = A.Drive(A.MOTORS[1], 50, gearbox=A.PLANETARY)
    eta = 0.97**2
    raise_ = dr.motor_torque(100.0, 0.0, 1.0)  # lifting: the motor drives the load
    lower = dr.motor_torque(100.0, 0.0, -1.0)  # lowering the same load: the load drives the motor
    assert raise_ == pytest.approx(100 / (50 * eta)) and lower == pytest.approx(100 * (2 - 1 / eta) / 50)


def test_gearbox_power_balance_both_ways():
    """Motor shaft power (through the gears) = joint power + heat made in the gearbox, raising or lowering."""
    rng = np.random.default_rng(1)
    tau, qd = rng.normal(0, 80, 200), rng.normal(0, 2, 200)
    for gb in (A.PLANETARY, A.STRAIN_WAVE):
        dr = A.Drive(A.MOTORS[1], 60, gearbox=gb)
        shaft = dr.motor_torque(tau, 0.0, qd) * dr.ratio * qd
        assert shaft == pytest.approx(tau * qd + dr.gearbox_loss(tau, qd), rel=1e-12, abs=1e-12)
        assert np.all(dr.gearbox_loss(tau, qd) >= 0)


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


# ---------------------------------------------------------------- push recovery

def test_pendulum_closed_form_matches_integration():
    from scipy.integrate import solve_ivp
    pend = R.Pendulum(z0=1.0, leg=0.9)
    sol = solve_ivp(lambda t, y: [y[1], pend.w**2 * (y[0] - 0.19)], (0, 0.6), [0.0, 0.8], rtol=1e-11, atol=1e-12,
                    dense_output=True)
    t = np.linspace(0, 0.6, 7)
    x, v = pend.com(0.8, t, 0.19)
    assert x == pytest.approx(sol.sol(t)[0], abs=1e-8) and v == pytest.approx(sol.sol(t)[1], abs=1e-8)


def test_center_of_pressure_on_the_capture_point_brings_the_robot_to_rest():
    pend = R.Pendulum(z0=1.0, leg=0.9)
    xi = pend.capture_point(0.0, 0.8)
    x, v = pend.com(0.8, 5.0, xi)
    assert x == pytest.approx(xi, abs=1e-6) and v == pytest.approx(0.0, abs=1e-6)
    x_short, _ = pend.com(0.8, 5.0, xi - 0.01)  # a centimeter short and it runs away
    assert x_short - xi > 1.0


def test_small_pushes_need_no_step_and_bigger_pushes_or_shorter_steps_need_a_faster_hip():
    pend = R.pendulum_from_model()
    assert R.required_hip_speed(pend, 0.4)[0] == 0.0  # the ankles can stop this one
    speeds = [R.required_hip_speed(pend, v)[0] for v in (0.8, 1.0, 1.2, 1.4)]
    assert np.all(np.diff(speeds) > 0)
    capped = [R.required_hip_speed(pend, 1.25, cap)[0] for cap in (np.inf, 0.8, 0.6, 0.5)]
    assert np.all(np.diff(capped) >= 0)
    assert R.required_hip_speed(pend, 1.5, max_step=0.4)[0] == np.inf


def test_min_jerk_peak_speed_is_fifteen_eighths_of_the_average():
    t = np.linspace(0, 1, 100001)
    _, qd, _ = lift.min_jerk(np.zeros(1), np.ones(1), 1.0, t)
    assert qd.max() == pytest.approx(R.MIN_JERK_PEAK, rel=1e-6)


# ---------------------------------------------------------------- datasheet motors

@pytest.mark.parametrize("d", A.MAXON_HT, ids=lambda d: d.part)
def test_maxon_datasheet_values_are_self_consistent(d):
    """Cross-checks between independently listed numbers, which would catch a mistyped value."""
    assert d.torque_constant_mnm_a == pytest.approx(60_000 / (2 * np.pi * d.speed_constant_rpm_v), rel=0.01)
    assert 0.98 < d.no_load_speed_rpm / (d.speed_constant_rpm_v * d.nominal_voltage) <= 1.0
    assert d.stall_current == pytest.approx(d.nominal_voltage / d.terminal_resistance, rel=0.01)
    # the listed stall torque sits far below Kt times the stall current: the iron saturates
    assert d.stall_torque_mnm < 0.5 * d.torque_constant_mnm_a * d.stall_current


def test_two_node_winding_model_settles_on_the_closed_form():
    mot = A.MOTORS[1]
    T = A.winding_temperature(mot, np.full(40000, 25.0), dt=0.5)
    assert T[-1] == pytest.approx(A.steady_winding_temperature(mot, 25.0), abs=0.2)


def test_two_node_winding_first_heats_like_its_own_small_mass():
    """For the first seconds the winding barely feels the housing: dT/dt is about P / C_winding."""
    mot = A.MOTORS[0]
    T = A.winding_temperature(mot, np.full(40, 50.0), dt=0.05)  # 2 s, far below its 37.8 s time constant
    rate = 50.0 * (1 + 0.00393 * (30.0 - 25.0)) / mot.c_winding
    assert (T[-1] - 30.0) / 2.0 == pytest.approx(rate, rel=0.05)
