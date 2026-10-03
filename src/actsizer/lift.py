"""Lift trajectories and inverse dynamics.

A lift is defined by two postures found with inverse kinematics (grip point
at the tote handle on the floor, then at the end height) and a minimum-jerk
blend between them in joint space. MuJoCo's inverse dynamics (mj_inverse)
then gives the torque each joint must supply at every instant.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .model import JOINTS, Body, com_x, grip_position, load

FOOT_X = (-0.07, 0.19)  # m, heel and toe relative to the ankle: where the centre of pressure must stay


def solve_posture(m, d, grip_xz, style: str, q0=None) -> np.ndarray:
    """IK for the grip point, regularized toward a lifting style.

    'squat': keep the trunk upright-ish, bend the knees.
    'stoop': keep the legs straight-ish, bend at the hip.
    The balance constraint keeps the whole-body centre of mass over the feet.
    """
    if style == "squat":
        prefer, w = np.array([0.35, -1.2, 0.9, 0.0, 0.0]), np.array([0.3, 0.3, 0.6, 0.05, 0.05])
    elif style == "stoop":
        prefer, w = np.array([0.05, -0.1, 1.4, 0.0, 0.0]), np.array([0.6, 0.6, 0.1, 0.05, 0.05])
    else:  # standing carry
        prefer, w = np.zeros(5), np.array([0.3, 0.3, 0.3, 0.02, 0.02])
    mid_foot = 0.5 * (FOOT_X[0] + FOOT_X[1])
    q0 = prefer if q0 is None else q0

    def resid(q):
        g = grip_position(m, d, q) - grip_xz
        bal = com_x(m, d, q) - mid_foot
        return np.concatenate([10 * g, [30 * bal], w * (q - prefer)])

    lo = np.array([-0.6, -2.4, -0.5, -2.8, -2.4])
    hi = np.array([0.8, 0.0, 2.2, 1.0, 0.0])
    return least_squares(resid, np.clip(q0, lo + 1e-3, hi - 1e-3), bounds=(lo, hi)).x


def min_jerk(q0, q1, T, t):
    s = np.clip(t / T, 0, 1)
    shape = 10 * s**3 - 15 * s**4 + 6 * s**5
    dshape = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    ddshape = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    dq = q1 - q0
    return q0 + np.outer(shape, dq), np.outer(dshape, dq), np.outer(ddshape, dq)


@dataclass
class LiftResult:
    t: np.ndarray
    q: np.ndarray  # (n, 5)
    qd: np.ndarray
    tau: np.ndarray  # joint torque, both sides together (N m)
    cop_x: np.ndarray  # centre of pressure under the feet (m from ankle)
    style: str

    @property
    def power(self) -> np.ndarray:
        return self.tau * self.qd

    def peak(self) -> dict:
        return {j: {"peak_torque_Nm": float(np.abs(self.tau[:, i]).max()),
                    "rms_torque_Nm": float(np.sqrt(np.mean(self.tau[:, i] ** 2))),
                    "peak_speed_rad_s": float(np.abs(self.qd[:, i]).max()),
                    "peak_power_W": float(np.abs(self.power[:, i]).max())} for i, j in enumerate(JOINTS)}

    def balanced(self) -> bool:
        return bool(np.all((self.cop_x > FOOT_X[0]) & (self.cop_x < FOOT_X[1])))


def simulate_lift(style: str = "squat", body: Body = Body(), duration: float = 1.5,
                  pick_xz=(0.35, 0.25), place_xz=(0.30, 0.95), armature=None, n: int = 301) -> LiftResult:
    m, d = load(body, armature)
    q_start = solve_posture(m, d, np.array(pick_xz), style)
    q_end = solve_posture(m, d, np.array(place_xz), "stand", q_start)
    t = np.linspace(0, duration, n)
    Q, Qd, Qdd = min_jerk(q_start, q_end, duration, t)
    tau = np.zeros_like(Q)
    cop = np.zeros(n)
    total_mass = m.body_mass.sum()
    for k in range(n):
        d.qpos[:], d.qvel[:], d.qacc[:] = Q[k], Qd[k], Qdd[k]
        mujoco.mj_inverse(m, d)
        tau[k] = d.qfrc_inverse
        # The floor supplies the ankle torque through the foot, so the centre of
        # pressure sits at x = tau_ankle / F_z ahead of the ankle (a positive torque
        # about +y pitches the body forward... and must be reacted by pressure toward
        # the toe). F_z is taken quasi-statically as the total weight.
        cop[k] = -tau[k, 0] / (total_mass * 9.81)
    return LiftResult(t, Q, Qd, tau, cop, style)
