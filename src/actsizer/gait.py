"""Human walking loads for the shuttle's carry and walk-back phases.

These are human-derived, not robot data: one mean stride of comfortable-speed overground
walking from Fukuchi, Fukuchi & Duarte's public dataset (PeerJ 2018, doi:10.7717/peerj.4640,
data doi:10.6084/m9.figshare.5722711, CC BY 4.0), reduced to data/wbds_comfortable_walking.csv.
The moments are per kg of body mass, so they are scaled to the mass the robot's legs carry.

Only the swing of each moment over a stride is used (its stride mean removed). It is added
on top of the standing-posture torque the MuJoCo model already gives for the carry, so the
tote's forward offset is counted once.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np

# dataset sign -> robot sign, for (moment, angle). Robot: hip + pitches the trunk forward (flexion),
# knee - bends it (flexion), ankle + tips the shank forward (dorsiflexion); torques follow the angles.
# Dataset: hip moment and angle flexion +, knee moment extension + and angle flexion +,
# ankle moment plantarflexion + and angle dorsiflexion +.
TO_ROBOT = {"hip": (1.0, 1.0), "knee": (1.0, -1.0), "ankle": (-1.0, 1.0)}
COLUMNS = {"hip": ("hip_moment_flx", "hip_angle_flx"), "knee": ("knee_moment_ext", "knee_angle_flx"),
           "ankle": ("ankle_moment_pf", "ankle_angle_df")}


@dataclass(frozen=True)
class Stride:
    pct: np.ndarray  # 0..100, % of the gait cycle
    data: dict  # column -> values, as in the file (dataset signs, N m/kg and degrees)
    stride_time: float  # s
    zero_mean_rms_range: dict  # joint -> (min, mean, max) over the 42 people, N m/kg

    def joint(self, name: str, mass: float):
        """One stride at one joint, in the robot's sign convention, for a body of ``mass`` kg.

        Returns (t, tau_swing, qd, qdd) on the file's 100 samples per stride (its 101st is the next
        heel strike): time (s), the moment's swing about its stride mean (N m, per leg), and the
        joint speed and acceleration (rad/s, rad/s^2), differentiated as a periodic Fourier series.
        """
        mcol, acol = COLUMNS[name]
        sm, sa = TO_ROBOT[name]
        moment = self.data[mcol][:-1]
        angle = sa * np.radians(self.data[acol][:-1])
        n = len(angle)
        w = 2 * np.pi * np.fft.rfftfreq(n, d=self.stride_time / n)
        spec = np.fft.rfft(angle)
        if n % 2 == 0:
            spec[-1] = 0.0  # the Nyquist term has no well-defined derivative
        qd = np.fft.irfft(1j * w * spec, n)
        qdd = np.fft.irfft(-(w**2) * spec, n)
        return np.arange(n) * self.stride_time / n, sm * mass * (moment - moment.mean()), qd, qdd


def load_stride() -> Stride:
    text = (files("actsizer") / "data" / "wbds_comfortable_walking.csv").read_text()
    lines = text.splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    rows = [ln for ln in lines if ln and not ln.startswith("#")]
    keys = rows[0].split(",")
    arr = np.array([[float(v) for v in r.split(",")] for r in rows[1:]])
    data = {k: arr[:, i] for i, k in enumerate(keys)}
    stride = next(ln for ln in head if "stride_time_s:" in ln)
    rms_line = next(ln for ln in head if "per_subject_zero_mean_rms" in ln).split(":", 1)[1]
    rms = {}
    for part in rms_line.split(";"):
        name, lo, mid, hi = part.split()
        rms[name] = (float(lo), float(mid), float(hi))
    return Stride(data["gait_pct"], data, float(stride.split(":")[1].split()[0]), rms)
