"""Motor + gearbox sizing, winding temperature, series elasticity, air-core vs iron-core.

MOTORS are three real frameless BLDC kits, maxon's EC frameless HT 60 M, 76 M and
90 M, built from the values on maxon's product pages (MAXON_HT, read 2026-10-06).
The illustrative motors this started with are kept as ILLUSTRATIVE_MOTORS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Motor:
    name: str
    kt: float  # N m / A
    r: float  # ohm (winding, line-to-line equivalent for the torque-producing current)
    j_rotor: float  # kg m^2
    w_max: float  # rad/s, no-load speed at bus voltage
    i_max: float  # A, peak current (driver / demagnetization limit)
    r_th: float  # K/W, winding to ambient
    c_th: float  # J/K, winding + stator thermal mass
    mass: float  # kg
    t_winding_max: float = 120.0  # C
    # With these two, the thermal model has two nodes: the winding (c_winding) behind
    # r_th_wh, then the housing (c_th) behind the rest of r_th to ambient.
    r_th_wh: float | None = None  # K/W, winding to housing
    c_winding: float | None = None  # J/K

    @property
    def km(self) -> float:
        return self.kt / np.sqrt(self.r)


ILLUSTRATIVE_MOTORS = [
    Motor("illustrative frameless S", 0.10, 0.20, 2.0e-5, 600.0, 30.0, 1.2, 150.0, 0.6),
    Motor("illustrative frameless M", 0.20, 0.12, 1.2e-4, 400.0, 40.0, 0.8, 300.0, 1.3),
    Motor("illustrative frameless L", 0.35, 0.08, 4.5e-4, 280.0, 50.0, 0.5, 600.0, 2.6),
]


@dataclass(frozen=True)
class MaxonDatasheet:
    """A frameless kit's values exactly as maxon lists them (units as published)."""

    name: str
    part: str
    nominal_voltage: float  # V
    no_load_speed_rpm: float
    nominal_torque_mnm: float  # max. continuous
    nominal_current: float  # A, max. continuous
    stall_torque_mnm: float
    stall_current: float  # A
    terminal_resistance: float  # ohm, phase to phase
    torque_constant_mnm_a: float
    speed_constant_rpm_v: float
    rotor_inertia_gcm2: float
    r_th_housing_ambient: float  # K/W
    r_th_winding_housing: float  # K/W
    tau_winding: float  # s, thermal time constant of the winding
    tau_motor: float  # s, thermal time constant of the motor
    max_winding_temp: float  # C
    weight_g: float

    def motor(self) -> Motor:
        """The kit as a Motor.

        The peak current is capped at the listed stall torque over Kt, not at the listed
        stall current: the stall current is just V/R, and the listed stall torque is far
        below Kt times it, because the iron saturates long before.
        """
        kt = self.torque_constant_mnm_a / 1000.0
        return Motor(f"maxon {self.name}", kt=kt, r=self.terminal_resistance, j_rotor=self.rotor_inertia_gcm2 * 1e-7,
                     w_max=self.no_load_speed_rpm * 2 * np.pi / 60, i_max=self.stall_torque_mnm / 1000.0 / kt,
                     r_th=self.r_th_winding_housing + self.r_th_housing_ambient,
                     c_th=self.tau_motor / self.r_th_housing_ambient, mass=self.weight_g / 1000.0,
                     t_winding_max=self.max_winding_temp, r_th_wh=self.r_th_winding_housing,
                     c_winding=self.tau_winding / self.r_th_winding_housing)


# From maxon's product pages (maxongroup.com/maxon/view/product/<part>), read 2026-10-06.
MAXON_HT = [
    MaxonDatasheet("EC frameless HT 60 M", "936176", 48, 4640, 612, 5.76, 3460, 96.9, 0.495, 98, 97.4, 177,
                   2.04, 1.65, 37.8, 231, 155, 226),
    MaxonDatasheet("EC frameless HT 76 M", "934928", 48, 3750, 1270, 9.65, 7000, 223, 0.215, 122, 78.6, 461,
                   1.5, 1.37, 54, 1320, 155, 376),
    MaxonDatasheet("EC frameless HT 90 M", "937612", 48, 2490, 2750, 13.9, 10600, 321, 0.149, 183, 52.2, 1200,
                   1.05, 1.12, 77.7, 341, 155, 649),
]
MOTORS = [d.motor() for d in MAXON_HT]


@dataclass(frozen=True)
class Gearbox:
    """Efficiency and backlash against ratio for one kind of gearbox.

    A planetary gearbox needs a new stage for every ``max_stage_ratio`` of reduction,
    and each stage multiplies the efficiency by ``stage_efficiency``, so efficiency
    steps down as the ratio grows. A strain-wave (harmonic) gearbox does its whole
    ratio range in one stage at a lower, roughly constant efficiency.
    The values in the presets below are typical-range assumptions, not a product's.
    """

    name: str
    stage_efficiency: float
    max_stage_ratio: float = 10.0
    stage_backlash_arcmin: float = 0.0  # play of one stage, measured at that stage's output
    ratio_range: tuple[float, float] = (1.0, np.inf)

    def stages(self, ratio: float) -> int:
        if self.max_stage_ratio == np.inf:
            return 1
        return max(1, int(np.ceil(np.log(ratio) / np.log(self.max_stage_ratio) - 1e-9)))

    def efficiency(self, ratio: float) -> float:
        return self.stage_efficiency ** self.stages(ratio)

    def backlash_arcmin(self, ratio: float) -> float:
        """Play at the output. Each upstream stage's play is divided by the ratio of the stages after it."""
        k = self.stages(ratio)
        stage_ratio = ratio ** (1.0 / k)
        return float(sum(self.stage_backlash_arcmin / stage_ratio**i for i in range(k)))


PLANETARY = Gearbox("planetary (assumed 0.97 per stage, up to 10:1 per stage)", 0.97, 10.0, 10.0)
STRAIN_WAVE = Gearbox("strain wave (assumed 0.75, one stage)", 0.75, np.inf, 0.0, (30.0, 160.0))


def backdriven_efficiency(eta: float) -> float:
    """Efficiency when the load drives the motor, for a gear train whose losses are Coulomb friction.

    Friction always opposes the motion, so running backwards it subtracts where it used to add:
    eta_back = 2 - 1/eta. Below eta = 0.5 that is zero or less: the gearbox self-locks.
    """
    return 2.0 - 1.0 / eta


@dataclass
class Drive:
    motor: Motor
    ratio: float
    efficiency: float | None = None  # forward efficiency; None takes it from the gearbox, or 0.90
    gearbox: Gearbox | None = None

    @property
    def eta(self) -> float:
        if self.efficiency is not None:
            return self.efficiency
        return self.gearbox.efficiency(self.ratio) if self.gearbox else 0.90

    def motor_torque(self, tau_joint, qdd_joint=0.0, qd_joint=None):
        """Motor torque to deliver tau_joint while also accelerating its own rotor (reflected N^2 J).

        Without ``qd_joint`` the gearbox is always taken as driven forward, which is conservative.
        With it, wherever the load is driving the motor (tau * qd < 0, like lowering a tote) the
        friction helps instead, and the torque scales with the backdriven efficiency.
        """
        tau = np.asarray(tau_joint, float)
        through = tau / (self.ratio * self.eta)
        if qd_joint is not None:
            back = tau * np.asarray(qd_joint, float) < 0
            through = np.where(back, tau * backdriven_efficiency(self.eta) / self.ratio, through)
        return through + self.motor.j_rotor * self.ratio * np.asarray(qdd_joint, float)

    def current(self, tau_joint, qdd_joint=0.0, qd_joint=None):
        return self.motor_torque(tau_joint, qdd_joint, qd_joint) / self.motor.kt

    def copper_loss(self, tau_joint, qdd_joint=0.0, qd_joint=None):
        return self.current(tau_joint, qdd_joint, qd_joint) ** 2 * self.motor.r

    def gearbox_loss(self, tau_joint, qd_joint):
        """Heat made by friction in the gearbox itself (not in the winding), W."""
        p = np.asarray(tau_joint, float) * np.asarray(qd_joint, float)
        return np.where(p >= 0, p * (1.0 / self.eta - 1.0), -p * (1.0 - backdriven_efficiency(self.eta)))

    def feasible(self, tau_joint, qd_joint, qdd_joint=0.0) -> bool:
        """Every point under the peak-current limit and the voltage-limited speed line.

        With back-EMF constant Ke = Kt (SI units), V = Ke w + I R gives the
        speed available at motor torque t:  w <= w_noload - t R / Kt^2.
        """
        m = self.motor
        tm = np.abs(self.motor_torque(tau_joint, qdd_joint))
        wm = np.abs(qd_joint) * self.ratio
        return bool(np.all(tm <= m.kt * m.i_max) and np.all(wm <= m.w_max - tm * m.r / m.kt**2))


def sweep_ratio(motor: Motor, tau, qd, qdd, ratios=None, gearbox: Gearbox | None = None):
    """Mean copper loss and feasibility across gear ratios for one joint's trajectory."""
    ratios = np.geomspace(5, 300, 200) if ratios is None else ratios
    loss, ok = [], []
    for n in ratios:
        dr = Drive(motor, n, gearbox=gearbox)
        loss.append(np.mean(dr.copper_loss(tau, qdd, qd if gearbox else None)))
        ok.append(dr.feasible(tau, qd, qdd))
    return ratios, np.array(loss), np.array(ok)


def best_ratio(motor, tau, qd, qdd):
    r, loss, ok = sweep_ratio(motor, tau, qd, qdd)
    if not ok.any():
        return None, None
    i = np.argmin(np.where(ok, loss, np.inf))
    return float(r[i]), float(loss[i])


def inertia_matched_ratio(j_load: float, j_rotor: float) -> float:
    """For a pure inertial load, copper loss is minimized at N = sqrt(J_load / J_rotor)."""
    return float(np.sqrt(j_load / j_rotor))


# ---------------------------------------------------------------- winding temperature

def winding_temperature(motor: Motor, p_loss_w: np.ndarray, dt: float, t_amb: float = 30.0, alpha_cu: float = 0.00393):
    """Winding temperature, with copper resistance rising with temperature.

    One node:   C dT/dt = P (1 + a (T - 25)) - (T - T_amb) / R_th
    Two nodes (when the motor gives r_th_wh and c_winding, as maxon's data do):
        C_w dT_w/dt = P (1 + a (T_w - 25)) - (T_w - T_h) / R_wh
        C_h dT_h/dt = (T_w - T_h) / R_wh - (T_h - T_amb) / R_ha,    R_ha = R_th - R_wh
    P is the copper loss at the 25 C resistance.
    """
    T = np.empty(len(p_loss_w))
    if motor.r_th_wh is None:
        t = t_amb
        for i, p in enumerate(p_loss_w):
            t += dt * (p * (1 + alpha_cu * (t - 25.0)) - (t - t_amb) / motor.r_th) / motor.c_th
            T[i] = t
        return T
    r_wh, r_ha = motor.r_th_wh, motor.r_th - motor.r_th_wh
    tw = th = t_amb
    for i, p in enumerate(p_loss_w):
        q_wh = (tw - th) / r_wh
        tw += dt * (p * (1 + alpha_cu * (tw - 25.0)) - q_wh) / motor.c_winding
        th += dt * (q_wh - (th - t_amb) / r_ha) / motor.c_th
        T[i] = tw
    return T


def steady_winding_temperature(motor: Motor, p_avg_w: float, t_amb: float = 30.0, alpha_cu: float = 0.00393) -> float:
    """Closed form of the model above at steady state (cycle-averaged loss)."""
    # T - T_amb = R_th * P (1 + a (T - 25))  ->  linear in T
    a, R, P = alpha_cu, motor.r_th, p_avg_w
    denom = 1 - R * P * a
    if denom <= 0:
        return np.inf  # thermal runaway: resistance grows faster than cooling
    return (t_amb + R * P * (1 - 25 * a)) / denom


# ---------------------------------------------------------------- series elasticity

def impact_peak_torque(j_link: float, j_reflected: float, k_series: float, k_env: float, w0: float,
                       t_end: float = 0.08) -> float:
    """Peak transmission torque when a moving link hits a stiff obstacle.

    Rotor (reflected inertia N^2 J_m) -- spring k_series -- link (J_link) -- contact k_env.
    Both start at speed w0; the motor applies no torque during the short event.
    A very large k_series is a rigid gearbox; a small one is a series elastic actuator.
    """
    def f(t, y):
        th_r, w_r, th_l, w_l = y
        tau_s = k_series * (th_r - th_l)
        tau_c = k_env * th_l if th_l > 0 else 0.0
        return [w_r, -tau_s / j_reflected, w_l, (tau_s - tau_c) / j_link]

    wn = np.sqrt(max(k_series, k_env) / min(j_link, j_reflected))
    sol = solve_ivp(f, (0, t_end), [0, w0, 0, w0], max_step=0.05 / wn, rtol=1e-8, atol=1e-10)
    return float(np.max(np.abs(k_series * (sol.y[0] - sol.y[2]))))


# ---------------------------------------------------------------- air-core vs iron-core

@dataclass(frozen=True)
class CoreModel:
    name: str
    kt0: float  # N m/A at low current
    i_sat: float  # A, current where iron starts to saturate (inf for air core)
    k_iron: float  # W / (rad/s)^1.5, Steinmetz-style iron loss coefficient at rated flux

    def torque(self, i):
        """Iron saturates: past i_sat the incremental torque constant falls to 30% of kt0."""
        i = np.asarray(i, float)
        if np.isinf(self.i_sat):
            return self.kt0 * i
        return self.kt0 * (0.7 * self.i_sat * np.tanh(i / self.i_sat) + 0.3 * i)

    def iron_loss(self, w):
        return self.k_iron * np.abs(np.asarray(w, float)) ** 1.5


IRON_CORE = CoreModel("iron-core radial (illustrative)", kt0=0.20, i_sat=30.0, k_iron=0.012)
AIR_CORE = CoreModel("air-core axial PCB stator (illustrative)", kt0=0.15, i_sat=np.inf, k_iron=0.0)
