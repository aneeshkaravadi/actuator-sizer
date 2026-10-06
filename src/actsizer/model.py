"""Sagittal-plane humanoid for a two-handed tote lift, built as a MuJoCo model.

Both legs and both arms move together in a symmetric lift, so each pair is
lumped into one chain. Joint torques reported here are therefore for BOTH
sides together; divide by two for one actuator.

Segment lengths, masses and center-of-mass locations are scaled from human
anthropometry (Winter, "Biomechanics and Motor Control of Human Movement",
Table 4.1), which is where human-scale humanoids also land.

Chain (feet bolted to the floor):  ankle -> shank -> knee -> thigh -> hip ->
trunk -> shoulder -> upper arm -> elbow -> forearm + hand (+ tote).
All joints rotate about +y; angles are relative to the parent segment, with
zero meaning straight up (standing tall, arms hanging straight down).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

JOINTS = ["ankle", "knee", "hip", "shoulder", "elbow"]


@dataclass
class Body:
    height: float = 1.73  # m
    mass: float = 73.0  # kg
    tote_mass: float = 25.0  # kg, held in both hands

    # (length fraction of height, mass fraction of body mass, COM fraction from proximal end)
    shank = (0.246, 2 * 0.0465, 0.433)
    thigh = (0.245, 2 * 0.100, 0.433)
    trunk = (0.288, 0.578, 0.60)  # hip to shoulder; includes head and neck
    upper_arm = (0.186, 2 * 0.028, 0.436)
    forearm = (0.146, 2 * 0.022, 0.682)  # forearm + hand, COM measured along the forearm
    grip_offset = 0.07  # m, wrist to grip point

    def seg(self, s):
        L, m, c = s
        return L * self.height, m * self.mass, c * L * self.height


def _capsule_inertia(m, L, r=0.04):
    """Rough slender-rod inertia about the COM, for the rotation axis (y)."""
    Iyy = m * L**2 / 12 + m * r**2 / 4
    return f"{m * r**2 / 2:.6f} {Iyy:.6f} {Iyy:.6f}"


def build_mjcf(b: Body = Body(), armature: dict[str, float] | None = None) -> str:
    """Return MJCF XML. ``armature`` adds reflected rotor inertia N^2 J_m per joint (kg m^2)."""
    arm = {j: 0.0 for j in JOINTS} | (armature or {})
    Ls, ms, cs = b.seg(b.shank)
    Lt, mt, ct = b.seg(b.thigh)
    Lk, mk, ck = b.seg(b.trunk)
    Lu, mu, cu = b.seg(b.upper_arm)
    Lf, mf, cf = b.seg(b.forearm)
    grip = Lf + b.grip_offset

    tote = f"""
          <body name="tote" pos="0 0 {-grip:.5f}">
            <site name="grip" pos="0 0 0" size="0.02"/>
            <inertial pos="0.0 0 -0.10" mass="{b.tote_mass:.3f}" diaginertia="0.30 0.25 0.20"/>
            <geom type="box" pos="0 0 -0.10" size="0.20 0.15 0.10" rgba="0.85 0.45 0.2 0.6"/>
          </body>"""
    forearm = f"""
        <body name="forearm" pos="0 0 {-Lu:.5f}">
          <joint name="elbow" type="hinge" axis="0 1 0" armature="{arm['elbow']:.6f}"/>
          <inertial pos="0 0 {-cf:.5f}" mass="{mf:.4f}" diaginertia="{_capsule_inertia(mf, Lf)}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-grip:.5f}" size="0.03" rgba="0.6 0.65 0.75 1"/>
          {tote}
        </body>"""
    upper = f"""
      <body name="upper_arm" pos="0 0 {Lk:.5f}">
        <joint name="shoulder" type="hinge" axis="0 1 0" armature="{arm['shoulder']:.6f}"/>
        <inertial pos="0 0 {-cu:.5f}" mass="{mu:.4f}" diaginertia="{_capsule_inertia(mu, Lu)}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-Lu:.5f}" size="0.035" rgba="0.6 0.65 0.75 1"/>
        {forearm}
      </body>"""
    trunk = f"""
    <body name="trunk" pos="0 0 {Lt:.5f}">
      <joint name="hip" type="hinge" axis="0 1 0" armature="{arm['hip']:.6f}"/>
      <inertial pos="0 0 {ck:.5f}" mass="{mk:.4f}" diaginertia="{_capsule_inertia(mk, Lk, 0.12)}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {Lk:.5f}" size="0.12" rgba="0.5 0.55 0.65 1"/>
      {upper}
    </body>"""
    thigh = f"""
    <body name="thigh" pos="0 0 {Ls:.5f}">
      <joint name="knee" type="hinge" axis="0 1 0" armature="{arm['knee']:.6f}"/>
      <inertial pos="0 0 {ct:.5f}" mass="{mt:.4f}" diaginertia="{_capsule_inertia(mt, Lt, 0.06)}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {Lt:.5f}" size="0.06" rgba="0.6 0.65 0.75 1"/>
      {trunk}
    </body>"""
    return f"""<mujoco model="sagittal_humanoid_tote_lift">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001"><flag contact="disable"/></option>
  <worldbody>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <geom type="box" pos="0.06 0 0.02" size="0.13 0.08 0.02" rgba="0.3 0.3 0.3 1"/>
    <body name="shank" pos="0 0 0.08">
      <joint name="ankle" type="hinge" axis="0 1 0" armature="{arm['ankle']:.6f}"/>
      <inertial pos="0 0 {cs:.5f}" mass="{ms:.4f}" diaginertia="{_capsule_inertia(ms, Ls, 0.05)}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {Ls:.5f}" size="0.05" rgba="0.6 0.65 0.75 1"/>
      {thigh}
    </body>
  </worldbody>
</mujoco>"""


def load(b: Body = Body(), armature=None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(build_mjcf(b, armature))
    return m, mujoco.MjData(m)


def grip_position(m, d, q) -> np.ndarray:
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site("grip").xpos[[0, 2]].copy()  # (x forward, z up)


def com_x(m, d, q) -> float:
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    total = m.body_mass[1:].sum()
    return float((d.xipos[1:, 0] * m.body_mass[1:]).sum() / total)
