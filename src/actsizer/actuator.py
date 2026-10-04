"""Motor + gearbox sizing, winding temperature, series elasticity, air-core vs iron-core.

Motor parameters below are ILLUSTRATIVE frameless BLDC values chosen to span a
realistic range (motor constant Km = Kt/sqrt(R) from ~0.2 to ~1.2 N m/sqrt(W)).
For a real design, swap in datasheet numbers; every function takes a Motor.
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

    @property
    def km(self) -> float:
        return self.kt / np.sqrt(self.r)


MOTORS = [
    Motor("illustrative frameless S", 0.10, 0.20, 2.0e-5, 600.0, 30.0, 1.2, 150.0, 0.6),
    Motor("illustrative frameless M", 0.20, 0.12, 1.2e-4, 400.0, 40.0, 0.8, 300.0, 1.3),
    Motor("illustrative frameless L", 0.35, 0.08, 4.5e-4, 280.0, 50.0, 0.5, 600.0, 2.6),
]


@dataclass
class Drive:
    motor: Motor
    ratio: float
    efficiency: float = 0.90  # gearbox

    def motor_torque(self, tau_joint, qdd_joint=0.0):
        """Motor torque to deliver tau_joint while also accelerating its own rotor (reflected N^2 J)."""
        return tau_joint / (self.ratio * self.efficiency) + self.motor.j_rotor * self.ratio * qdd_joint

    def current(self, tau_joint, qdd_joint=0.0):
        return self.motor_torque(tau_joint, qdd_joint) / self.motor.kt

    def copper_loss(self, tau_joint, qdd_joint=0.0):
        return self.current(tau_joint, qdd_joint) ** 2 * self.motor.r

    def feasible(self, tau_joint, qd_joint, qdd_joint=0.0) -> bool:
        """Every point under the peak-current limit and the voltage-limited speed line.

        With back-EMF constant Ke = Kt (SI units), V = Ke w + I R gives the
        speed available at motor torque t:  w <= w_noload - t R / Kt^2.
        """
        m = self.motor
        tm = np.abs(self.motor_torque(tau_joint, qdd_joint))
        wm = np.abs(qd_joint) * self.ratio
        return bool(np.all(tm <= m.kt * m.i_max) and np.all(wm <= m.w_max - tm * m.r / m.kt**2))


def sweep_ratio(motor: Motor, tau, qd, qdd, ratios=np.geomspace(5, 300, 200)):
    """RMS copper loss and feasibility across gear ratios for one joint's trajectory."""
    loss, ok = [], []
    for n in ratios:
        dr = Drive(motor, n)
        loss.append(np.mean(dr.copper_loss(tau, qdd)))
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
    """First-order thermal model with copper resistance rising with temperature.

    C dT/dt = P_loss * (1 + a (T - 25)) / (1 + a (T_ref - 25))  -  (T - T_amb) / R_th,
    where P_loss was computed at the 25 C resistance.
    """
    T = np.empty(len(p_loss_w))
    t = t_amb
    for i, p in enumerate(p_loss_w):
        t += dt * (p * (1 + alpha_cu * (t - 25.0)) - (t - t_amb) / motor.r_th) / motor.c_th
        T[i] = t
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
