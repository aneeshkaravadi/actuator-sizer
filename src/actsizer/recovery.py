"""One-step push recovery, to get the hip speed a humanoid needs from a task instead of a guess.

The robot is a linear inverted pendulum (Kajita et al., IROS 2001): the center of
mass at constant height z0, pushed around by the center of pressure p under the
foot:

    x'' = w^2 (x - p),    w = sqrt(g / z0)

The capture point (Pratt et al., Humanoids 2006) is  xi = x + x'/w.  With p held
fixed, it runs away exponentially:  xi(t) = p + (xi(0) - p) e^(w t).  If the push
leaves xi inside the foot, the ankles alone can stop the robot. If not, it has to
step, and the new foot has to land where xi has got to by then. The swing leg
covers that in a minimum-jerk motion of the hip, which peaks at 15/8 of its
average speed, so a quick step needs a fast hip and a slow step needs a long one.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .lift import FOOT_X
from .model import Body, load

MIN_JERK_PEAK = 15.0 / 8.0  # peak over average speed of a minimum-jerk move


@dataclass(frozen=True)
class Pendulum:
    z0: float  # m, center-of-mass height
    leg: float  # m, hip height above the floor: the swing leg's length
    heel: float = FOOT_X[0]  # m, relative to the ankle
    toe: float = FOOT_X[1]
    g: float = 9.81

    @property
    def w(self) -> float:
        return float(np.sqrt(self.g / self.z0))

    def com(self, v0: float, t, p: float, x0: float = 0.0):
        """Center-of-mass position and speed at time t with the center of pressure held at p."""
        t = np.asarray(t, float)
        w = self.w
        x = p + (x0 - p) * np.cosh(w * t) + v0 / w * np.sinh(w * t)
        v = (x0 - p) * w * np.sinh(w * t) + v0 * np.cosh(w * t)
        return x, v

    def capture_point(self, x, v):
        return x + v / self.w

    def ankles_suffice(self, v0: float) -> bool:
        """The push leaves the capture point inside the foot (the robot starts over its ankle)."""
        return self.capture_point(0.0, v0) <= self.toe


def pendulum_from_model(body: Body | None = None) -> Pendulum:
    """Center-of-mass and hip heights of the MuJoCo model standing straight (no tote by default)."""
    m, d = load(Body(tote_mass=0.0) if body is None else body)
    d.qpos[:] = 0.0
    mujoco.mj_kinematics(m, d)
    z = float((d.xipos[1:, 2] * m.body_mass[1:]).sum() / m.body_mass[1:].sum())
    return Pendulum(z0=z, leg=float(d.body("trunk").xpos[2]))


def step(pend: Pendulum, v0: float, swing_time: float, reaction: float = 0.0):
    """What one recovery step takes if the foot lands ``reaction + swing_time`` after the push.

    Until touchdown the center of pressure stays at the toe (the ankles push as hard as they
    can). The new foot is placed so the capture point lands mid-foot. Returns the peak hip
    speed (rad/s), the step length (m, ankle to ankle) and the swing leg's angle at touchdown,
    with the speed and angle NaN where the foothold is farther from the hip than the leg reaches.
    """
    t_land = reaction + swing_time
    x, v = pend.com(v0, t_land, pend.toe)
    foot = pend.capture_point(x, v) - 0.5 * (pend.heel + pend.toe)
    reach = (foot - x) / pend.leg  # hip over the center of mass
    angle = np.where(np.abs(reach) <= 1.0, np.arcsin(np.clip(reach, -1.0, 1.0)), np.nan)
    return MIN_JERK_PEAK * angle / swing_time, foot, angle


def required_hip_speed(pend: Pendulum, v0: float, max_step: float = np.inf, reaction: float = 0.0,
                       swing_times=None) -> tuple[float, float, float]:
    """Slowest hip that still recovers from a push of v0 (m/s) in one step no longer than max_step.

    Returns (hip speed rad/s, swing time s, step length m). (0, 0, 0) if the ankles are enough,
    and (inf, nan, nan) if no single step that short can do it.
    """
    if pend.ankles_suffice(v0):
        return 0.0, 0.0, 0.0
    swing_times = np.linspace(0.08, 0.8, 1441) if swing_times is None else swing_times
    speed, foot, _ = step(pend, v0, swing_times, reaction)
    ok = (foot <= max_step) & np.isfinite(speed) & (speed > 0)
    if not ok.any():
        return np.inf, np.nan, np.nan
    k = int(np.argmin(np.where(ok, speed, np.inf)))
    return float(speed[k]), float(swing_times[k]), float(foot[k])
