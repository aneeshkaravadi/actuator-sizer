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
    r = lift.simulate_lift("stoop", duration=2.0)
    m, d = model.load()
    assert np.allclose(model.grip_position(m, d, r.q[0]), [0.35, 0.25], atol=0.02)
    assert np.allclose(model.grip_position(m, d, r.q[-1]), [0.30, 0.95], atol=0.02)
    assert r.balanced()


def test_center_of_pressure_matches_the_multibody_zmp_from_kinematics_alone():
    """The CoP from the ankle torque and floor force has to match the zero-moment point worked out
    from nothing but how each body moves (Newton-Euler for the whole body about a point on the floor),
    with z measured from the floor,
        x_zmp = [sum m (x (g + z'') - z x'') - sum I_y theta''] / sum m (g + z'').
    Body accelerations come from finite differences of the planned motion, not from MuJoCo's dynamics."""
    T = 1.5
    r = lift.simulate_lift("stoop", duration=T, n=61)
    m, d = model.load()
    bodies = range(1, m.nbody)
    assert np.allclose(m.body_iquat[1:], [1, 0, 0, 0])  # inertia axes line up with the body frames
    chain = {b: [m.jnt_qposadr[j] for j in range(m.njnt) if _is_ancestor(m, m.jnt_bodyid[j], b)] for b in bodies}

    def state(t):
        q, _, qdd = lift.min_jerk(r.q[0], r.q[-1], T, np.atleast_1d(t))
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(m, d)
        return d.xipos[1:, [0, 2]].copy(), qdd[0]

    h = 1e-4
    for k in range(5, 56, 5):
        t = r.t[k]
        (p0, qdd), (pm, _), (pp, _) = state(t), state(t - h), state(t + h)
        acc = (pp - 2 * p0 + pm) / h**2
        mass = m.body_mass[1:]
        alpha = np.array([qdd[chain[b]].sum() for b in bodies])  # planar: a body's pitch is the sum of the joints below it
        num = np.sum(mass * (p0[:, 0] * (G + acc[:, 1]) - p0[:, 1] * acc[:, 0])) - np.sum(m.body_inertia[1:, 1] * alpha)
        x_zmp = num / np.sum(mass * (G + acc[:, 1]))
        assert r.cop_x[k] == pytest.approx(x_zmp, abs=2e-4)


def _is_ancestor(m, a, b):
    while b > 0:
        if b == a:
            return True
        b = m.body_parentid[b]
    return False


def test_center_of_pressure_of_a_still_posture_is_under_the_center_of_mass():
    m, d = model.load()
    q = np.array([0.1, -0.3, 0.6, -0.4, -0.2])
    tau = np.zeros(5)
    d.qpos[:], d.qvel[:], d.qacc[:] = q, 0, 0
    mujoco.mj_inverse(m, d)
    mujoco.mj_rnePostConstraint(m, d)
    fx, fz = d.cfrc_int[1][[3, 5]]
    tau[:] = d.qfrc_inverse
    assert fx == pytest.approx(0.0, abs=1e-9) and fz == pytest.approx(m.body_mass.sum() * G, rel=1e-9)
    assert lift.center_of_pressure(tau[0], fx, fz, lift.ankle_height(m, d)) == pytest.approx(model.com_x(m, d, q), abs=1e-9)


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


def test_backdriven_efficiency_agrees_with_an_inclined_plane_with_coulomb_friction():
    """A gear mesh is a wedge. Pushing a load W up a slope of angle lam with friction angle phi takes
    a force W tan(lam + phi), and holding it while it slides back down takes W tan(lam - phi), so
        eta_forward = tan(lam) / tan(lam + phi),   eta_back = tan(lam - phi) / tan(lam).
    For mesh-like friction (mu of a few percent) that matches 2 - 1/eta to second order in mu, and
    both say a drive that is about 50% efficient or worse can't be backdriven at all."""
    for lam in np.radians([10.0, 20.0, 35.0, 50.0]):
        for mu in (0.01, 0.02, 0.04):
            phi = np.arctan(mu)
            eta_f, eta_b = np.tan(lam) / np.tan(lam + phi), np.tan(lam - phi) / np.tan(lam)
            assert A.backdriven_efficiency(eta_f) == pytest.approx(eta_b, abs=6 * mu**2)
    lam = np.radians(3.0)  # self-locking: the slope is exactly as steep as the friction angle
    assert A.backdriven_efficiency(np.tan(lam) / np.tan(2 * lam)) == pytest.approx(0.0, abs=0.01)


def test_drive_torque_raising_and_lowering():
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
    """Cross-checks between independently listed numbers, which would catch a mistyped value.
    The relations are maxon's own (catalog 2026/27, pp. 90-91)."""
    assert d.torque_constant_mnm_a == pytest.approx(60_000 / (2 * np.pi * d.speed_constant_rpm_v), rel=0.01)
    assert 0.98 < d.no_load_speed_rpm / (d.speed_constant_rpm_v * d.nominal_voltage) <= 1.0
    assert d.stall_current == pytest.approx(d.nominal_voltage / d.terminal_resistance, rel=0.01)
    # speed/torque gradient = 30000/pi R / kM^2, and mechanical time constant = R J / kM^2 (rotor inertia)
    kt = d.torque_constant_mnm_a / 1000
    assert d.speed_torque_gradient_rpm_mnm == pytest.approx(d.terminal_resistance / kt**2 * 60 / (2 * np.pi) / 1000, rel=0.02)
    assert d.mechanical_time_constant_ms == pytest.approx(d.terminal_resistance * d.rotor_inertia_gcm2 * 1e-7 / kt**2 * 1e3, rel=0.02)
    # max. efficiency = (1 - sqrt(I0 / I_A))^2 (no-load current)
    assert d.max_efficiency == pytest.approx((1 - np.sqrt(d.no_load_current / d.stall_current)) ** 2, abs=0.006)
    # maxon's general rule is stall torque = kM x starting current; for these kits it lists far less
    assert d.stall_torque_mnm < 0.5 * d.torque_constant_mnm_a * d.stall_current


@pytest.mark.parametrize("d", A.MAXON_HT, ids=lambda d: d.part)
def test_maxon_rated_current_brings_the_winding_to_its_limit(d):
    """maxon's nominal current is the continuous current that takes the winding to its maximum
    temperature at 25 C ambient (catalog p. 92). Running the thermal model at that current and the
    nominal speed is an independent check on the thermal resistances, once the iron losses are in:
    within 10 K of 155 C, 8% of the 130 K rise (maxon's product page and catalog differ by 8% on
    R_th1 alone). The misses are -2, +6 and +9 K. Copper loss alone comes out 47 to 56 K short."""
    assert d.rated_winding_temperature() == pytest.approx(d.max_winding_temp, abs=10.0)
    assert d.rated_winding_temperature(iron=False) < d.max_winding_temp - 40.0


def test_two_node_winding_model_settles_on_the_closed_form():
    mot = A.MOTORS[1]
    T = A.winding_temperature(mot, np.full(40000, 25.0), dt=0.5)
    assert T[-1] == pytest.approx(A.steady_winding_temperature(mot, 25.0), abs=0.2)


def test_two_node_winding_with_iron_loss_settles_on_the_closed_form():
    mot = A.MOTORS[2]
    T = A.winding_temperature(mot, np.full(40000, 20.0), dt=0.5, p_other_w=8.0)
    assert T[-1] == pytest.approx(A.steady_winding_temperature(mot, 20.0, p_other_w=8.0), abs=0.2)


def test_two_node_winding_first_heats_like_its_own_small_mass():
    """For the first seconds the winding barely feels the housing: dT/dt is about P / C_winding."""
    mot = A.MOTORS[0]
    T = A.winding_temperature(mot, np.full(40, 50.0), dt=0.05)  # 2 s, far below its 37.8 s time constant
    rate = 50.0 * (1 + 0.00393 * (30.0 - 25.0)) / mot.c_winding
    assert (T[-1] - 30.0) / 2.0 == pytest.approx(rate, rel=0.05)


def test_two_node_winding_under_a_repeating_load_averages_to_the_steady_value():
    """A 30 s cycle (60 W for 3 s, 10 W for the rest): the winding ripples, but its mean over a cycle
    settles near the closed-form temperature for the average loss (the copper's temperature
    coefficient makes it slightly nonlinear)."""
    mot = A.MOTORS[1]
    dt = 0.05
    cycle = np.r_[np.full(60, 60.0), np.full(540, 10.0)]
    T = A.winding_temperature(mot, np.tile(cycle, 300), dt)[-len(cycle):]
    assert T.mean() == pytest.approx(A.steady_winding_temperature(mot, float(cycle.mean())), abs=1.0)
    assert T.max() - T.min() > 1.0  # it really does ripple


# ---------------------------------------------------------------- one-handed and twisting lifts

def test_one_handed_static_moments_match_hand_calculations():
    """Arms hang (or the loaded one hangs plumb), so the arms' own weights cancel side to side and only
    the tote is off center: tote weight times the shoulder offset, as hip roll standing up and as
    trunk twist when bent 90 degrees forward (the trunk's axis is then horizontal)."""
    from actsizer import asym
    b = model.Body()
    m, d = asym.load_asym(b)
    expected = b.tote_mass * G * asym.SHOULDER_Y
    names = asym.JOINTS_ASYM
    d.qpos[:], d.qvel[:], d.qacc[:] = 0.0, 0.0, 0.0
    mujoco.mj_inverse(m, d)
    assert abs(d.qfrc_inverse[names.index("hip_roll")]) == pytest.approx(expected, rel=1e-6)
    q = np.zeros(7)
    q[names.index("hip")], q[names.index("shoulder")] = np.pi / 2, -np.pi / 2  # bent over, arm plumb
    d.qpos[:] = q
    mujoco.mj_inverse(m, d)
    assert abs(d.qfrc_inverse[names.index("trunk_yaw")]) == pytest.approx(expected, rel=1e-6)


def test_twisting_lift_work_equals_potential_energy_gain():
    from actsizer import asym
    r = asym.simulate_asym_lift((0.10, 0.38, 0.30), (0.30, 0.20, 0.95), duration=3.0, n=1201)
    m, d = asym.load_asym()

    def pe(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        return float((m.body_mass[1:] * d.xipos[1:, 2]).sum() * G)

    work = np.trapezoid((r.tau * r.qd).sum(axis=1), r.t)
    assert work == pytest.approx(pe(r.q[-1]) - pe(r.q[0]), rel=0.01)


# ---------------------------------------------------------------- human walking loads

def test_walking_data_carry_the_body_weight_over_a_stride():
    """Walking at a steady speed, the floor holds the body up on average: one leg's vertical force,
    averaged over its stride, is half the weight. This checks that the fixture really is per kg of
    body mass and covers one whole stride (it is the dataset's own force-plate data)."""
    from actsizer import gait
    s = gait.load_stride()
    assert np.mean(s.data["grf_vertical"][:-1]) == pytest.approx(G / 2, rel=0.03)
    assert 0.9 < s.stride_time < 1.2  # a comfortable adult stride


def test_walking_moments_keep_their_physical_meaning_in_the_robot_convention():
    """Early stance the hip pushes the trunk back (extensor) and late stance the ankle pushes off
    (plantarflexor). In the MuJoCo model a negative hip torque holds a forward-leaning trunk up and a
    negative ankle torque holds a forward-leaning body up, so both must come out negative."""
    from actsizer import gait
    s = gait.load_stride()
    robot = {}
    for j in ("hip", "ankle"):
        _, swing, _, _ = s.joint(j, 1.0)
        mean = gait.TO_ROBOT[j][0] * s.data[gait.COLUMNS[j][0]][:-1].mean()
        robot[j] = swing + mean  # the whole moment, in the robot's signs
    assert robot["hip"][2:12].mean() < -0.3 and robot["ankle"][40:55].mean() < -1.0
    m, d = model.load()
    for q, j in (([0, 0, 0.5, 0, 0], 2), ([0.2, 0, 0, 0, 0], 0)):
        d.qpos[:], d.qvel[:], d.qacc[:] = q, 0, 0
        mujoco.mj_inverse(m, d)
        assert d.qfrc_inverse[j] < 0
    # angles too: a flexed hip at heel strike, and a bent knee (negative in the robot) in swing
    assert s.data["hip_angle_flx"][0] > 20 and gait.TO_ROBOT["knee"][1] * s.data["knee_angle_flx"][70] < -30


def test_walking_speeds_integrate_back_to_the_angles():
    from actsizer import gait
    s = gait.load_stride()
    for j in ("hip", "knee", "ankle"):
        t, _, qd, qdd = s.joint(j, 1.0)
        ang = gait.TO_ROBOT[j][1] * np.radians(s.data[gait.COLUMNS[j][1]][:-1])
        dt = t[1] - t[0]
        rebuilt = ang[0] + np.concatenate([[0], np.cumsum(0.5 * (qd[1:] + qd[:-1]) * dt)])
        assert rebuilt == pytest.approx(ang, abs=0.01)
        assert np.mean(qd) == pytest.approx(0.0, abs=1e-9) and np.mean(qdd) == pytest.approx(0.0, abs=1e-9)


def test_walking_swing_adds_its_mean_square_to_the_holding_loss():
    """Forward-driven, the copper loss of (hold + zero-mean swing) averages to the hold's loss plus the
    swing's mean square: the cross term vanishes over a stride."""
    from actsizer import gait
    _, tau, _, _ = gait.load_stride().joint("hip", 98.0)
    dr = A.Drive(A.MOTORS[1], 40.0, efficiency=0.94)
    hold = -50.0
    expected = (hold**2 + np.mean(tau**2)) * dr.motor.r / (dr.motor.kt * 40.0 * 0.94) ** 2
    assert np.mean(dr.copper_loss(hold + tau)) == pytest.approx(expected, rel=1e-9)
