"""One-handed and twisting lifts: the sagittal model plus the joints they need.

The legs stay lumped (both legs as one chain, as in model.py), but the upper
body is no longer symmetric. Three hinges at the hip let the trunk lean
sideways (hip_roll, about x), bend forward (hip, about y) and turn (trunk_yaw,
about z). One arm, at a sideways shoulder offset, carries the tote with a
shoulder and an elbow; the other hangs at the opposite offset with no joints.
Torques at the hip and trunk are for both sides together, and the arm's are
for the one loaded arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .lift import FOOT_X, min_jerk
from .model import Body, _capsule_inertia

JOINTS_ASYM = ["ankle", "knee", "hip_roll", "hip", "trunk_yaw", "shoulder", "elbow"]
SHOULDER_Y = 0.20  # m, half the shoulder width of a 1.73 m adult (biacromial breadth ~0.4 m)
STANCE_Y = 0.08  # m, how far sideways the center of mass may sit from midway between the feet


def build_asym_mjcf(b: Body | None = None, shoulder_y: float = SHOULDER_Y) -> str:
    b = Body() if b is None else b
    Ls, ms, cs = b.seg(b.shank)
    Lt, mt, ct = b.seg(b.thigh)
    Lk, mk, ck = b.seg(b.trunk)
    Lu, mu2, cu = b.seg(b.upper_arm)
    Lf, mf2, cf = b.seg(b.forearm)
    mu, mf = mu2 / 2, mf2 / 2  # Body's arm segments are both arms together
    grip = Lf + b.grip_offset
    arm_com = (mu * cu + mf * (Lu + cf)) / (mu + mf)  # free arm hanging straight, COM below the shoulder
    return f"""<mujoco model="humanoid_one_handed_lift">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001"><flag contact="disable"/></option>
  <worldbody>
    <body name="shank" pos="0 0 0.08">
      <joint name="ankle" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {cs:.5f}" mass="{ms:.4f}" diaginertia="{_capsule_inertia(ms, Ls, 0.05)}"/>
      <body name="thigh" pos="0 0 {Ls:.5f}">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 {ct:.5f}" mass="{mt:.4f}" diaginertia="{_capsule_inertia(mt, Lt, 0.06)}"/>
        <body name="trunk" pos="0 0 {Lt:.5f}">
          <joint name="hip_roll" type="hinge" axis="1 0 0"/>
          <joint name="hip" type="hinge" axis="0 1 0"/>
          <joint name="trunk_yaw" type="hinge" axis="0 0 1"/>
          <inertial pos="0 0 {ck:.5f}" mass="{mk:.4f}" diaginertia="{_capsule_inertia(mk, Lk, 0.12)}"/>
          <body name="free_arm" pos="0 {-shoulder_y:.4f} {Lk:.5f}">
            <inertial pos="0 0 {-arm_com:.5f}" mass="{mu + mf:.4f}" diaginertia="0.02 0.02 0.002"/>
          </body>
          <body name="upper_arm" pos="0 {shoulder_y:.4f} {Lk:.5f}">
            <joint name="shoulder" type="hinge" axis="0 1 0"/>
            <inertial pos="0 0 {-cu:.5f}" mass="{mu:.4f}" diaginertia="{_capsule_inertia(mu, Lu)}"/>
            <body name="forearm" pos="0 0 {-Lu:.5f}">
              <joint name="elbow" type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 {-cf:.5f}" mass="{mf:.4f}" diaginertia="{_capsule_inertia(mf, Lf)}"/>
              <body name="tote" pos="0 0 {-grip:.5f}">
                <site name="grip" pos="0 0 0" size="0.02"/>
                <inertial pos="0.0 0 -0.10" mass="{b.tote_mass:.3f}" diaginertia="0.30 0.25 0.20"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def load_asym(b: Body | None = None, shoulder_y: float = SHOULDER_Y):
    m = mujoco.MjModel.from_xml_string(build_asym_mjcf(b, shoulder_y))
    return m, mujoco.MjData(m)


def _grip(m, d, q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site("grip").xpos.copy()


def _com_xy(m, d, q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    w = m.body_mass[1:]
    return (d.xipos[1:, :2] * w[:, None]).sum(axis=0) / w.sum()


def solve_asym_posture(m, d, grip_xyz, style: str = "stoop", q0=None) -> np.ndarray:
    """IK: reach the grip point, keep the center of mass over the middle of the feet, prefer a style.

    'stoop' prefers straight legs and a bent hip, 'stand' an upright body.
    """
    if style == "stoop":
        prefer, w = np.array([0.05, -0.1, 0.0, 1.4, 0.0, 0.0, 0.0]), np.array([0.6, 0.6, 0.3, 0.1, 0.05, 0.05, 0.05])
    else:
        prefer, w = np.zeros(7), np.array([0.3, 0.3, 0.3, 0.3, 0.05, 0.02, 0.02])
    mid_foot = 0.5 * (FOOT_X[0] + FOOT_X[1])

    def resid(q):
        com = _com_xy(m, d, q)
        # Fore-aft the center of mass has to sit over the middle of the foot, like the sagittal model.
        # Sideways the feet are apart, so anywhere within STANCE_Y of the middle is balanced.
        side = np.sign(com[1]) * max(abs(com[1]) - STANCE_Y, 0.0)
        return np.concatenate([10 * (_grip(m, d, q) - grip_xyz), [30 * (com[0] - mid_foot), 30 * side], w * (q - prefer)])

    lo = np.array([-0.6, -2.4, -0.5, -0.5, -1.2, -2.8, -2.4])
    hi = np.array([0.8, 0.0, 0.5, 2.2, 1.2, 1.0, 0.0])
    q0 = prefer if q0 is None else q0
    return least_squares(resid, np.clip(q0, lo + 1e-3, hi - 1e-3), bounds=(lo, hi)).x


@dataclass
class AsymLift:
    t: np.ndarray
    q: np.ndarray  # (n, 7)
    qd: np.ndarray
    tau: np.ndarray  # joint torques, (n, 7)

    def peak(self) -> dict:
        return {j: float(np.abs(self.tau[:, i]).max()) for i, j in enumerate(JOINTS_ASYM)}


def simulate_asym_lift(pick_xyz, place_xyz, b: Body | None = None, duration: float = 1.5, n: int = 301) -> AsymLift:
    """Minimum-jerk lift between two IK postures, with MuJoCo inverse dynamics at every instant."""
    m, d = load_asym(b)
    q_start = solve_asym_posture(m, d, np.asarray(pick_xyz, float), "stoop")
    q_end = solve_asym_posture(m, d, np.asarray(place_xyz, float), "stand")
    t = np.linspace(0, duration, n)
    Q, Qd, Qdd = min_jerk(q_start, q_end, duration, t)
    tau = np.zeros_like(Q)
    for k in range(n):
        d.qpos[:], d.qvel[:], d.qacc[:] = Q[k], Qd[k], Qdd[k]
        mujoco.mj_inverse(m, d)
        tau[k] = d.qfrc_inverse
    return AsymLift(t, Q, Qd, tau)
