# actuator-sizer

**From a task to an actuator spec for a humanoid: MuJoCo inverse dynamics of a 25 kg tote lift, then gear-ratio trades, winding temperature, series elasticity, and a CAD envelope.**

[![tests](https://github.com/aneeshkaravadi/actuator-sizer/actions/workflows/ci.yml/badge.svg)](https://github.com/aneeshkaravadi/actuator-sizer/actions/workflows/ci.yml)

<img src="docs/figures/postures.png" width="80%">

## What this shows

The model is a human-scale sagittal-plane humanoid (1.73 m, 73 kg, segment data scaled from Winter's anthropometry). It lifts a 25 kg tote from a floor-level handle to waist height. MuJoCo's inverse dynamics gives every joint's torque, speed and power, and the centre of pressure under the feet decides whether the lift stays balanced.

| Per-side peak (25 kg tote) | ankle | knee | hip | shoulder | elbow |
|---|---|---|---|---|---|
| **stoop**, 1.52 s: torque (N m) | 87 | 119 | **221** | 67 | 45 |
| stoop: power (W) | 26 | 27 | **395** | 69 | 16 |
| **squat**, 3.92 s: torque (N m) | 33 | 81 | 103 | 65 | 45 |
| squat: power (W) | 8 | 86 | 83 | 18 | 10 |

### Finding 1: balance, not torque, sets how fast a squat lift can go

The fastest stoop lift that keeps the centre of pressure on the foot takes **1.47 s**. The squat lift needs **3.87 s**: rising quickly out of a deep squat throws the centre of pressure behind the heel.

The squat halves peak hip torque, but costs 2.6× the cycle time. For a robot, "lift with your legs" is a throughput trade, not a free win.

![Lift torques](docs/figures/lift_torques.png)

### Finding 2: gear ratio is a trade between heat, speed and impact, not a single optimum

For a slow, gravity-dominated lift, copper loss keeps falling as gear ratio rises. What caps the ratio is:
- the speed a joint needs for fast motions (an explicit 6 rad/s hip requirement here)
- the reflected rotor inertia $N^2 J$ that the joint carries into every collision

<img src="docs/figures/hip_ratio_trade.png" width="49%"> <img src="docs/figures/shuttle_thermal.png" width="49%">

The right plot repeats a tote shuttle (120 totes/hour: lift, carry, lower, return). With the same mid-size motor at every joint, the winding passes 120 °C below these ratios:

| hip | shoulder | knee | elbow | ankle |
|---|---|---|---|---|
| 12.7 | 10.1 | 8.5 | 7.1 | 6.3 |

The hip and the shoulder (holding the tote out in front) cook first. Quasi-direct-drive ratios (6–10) need a bigger motor at those joints.

### Finding 3: a series spring protects the gearbox, but only when reflected inertia is large

Here the hip hits a stiff obstacle at 1 rad/s.

| Gear ratio | Gearbox torque spike, rigid | With a 3000 N m/rad spring |
|---|---|---|
| N = 120 | 470 N m | 98 N m |
| N = 50 | 117 N m | 54 N m |
| N = 20 | 18 N m | 21 N m |

At N = 20 the spring slightly *raises* the spike, because the rotor is already light.

The price of the spring is deflection under load: 0.074 rad at 3000 N m/rad, at the hip's peak lift torque.

<img src="docs/figures/sea_impact.png" width="60%">

### Also included

- **Air-core vs iron-core motors** (illustrative parameters): an iron core saturates at high current and has speed-dependent core loss, while an air core (e.g. a PCB axial-flux stator) is linear with no core loss. ([figure](docs/figures/aircore_vs_iron.png))
- **CAD envelope** for the hip actuator (motor bay, gearbox bay, output flange bolt circle, mounting ears): [`cad/hip_actuator_envelope.step`](cad/hip_actuator_envelope.step).
- **The MuJoCo model itself**: [`models/sagittal_humanoid.xml`](models/sagittal_humanoid.xml).

## Checks

10 tests in CI, including:
- the shoulder torque with arms level matches a hand calculation to 1e-6
- standing straight needs zero hip and knee torque
- **total joint work over a lift equals the potential-energy gain within 1%**
- the copper-loss-optimal ratio for a pure inertia matches $N^* = \sqrt{J_\text{load}/J_\text{rotor}}$
- the rigid-impact spike matches its closed form
- the transient winding model converges to the closed-form steady state

## Quick start

```bash
git clone https://github.com/aneeshkaravadi/actuator-sizer && cd actuator-sizer
pip install -e ".[dev,cad]"
pytest -q
python examples/make_figures.py
```

```python
from actsizer import lift, actuator as A
r = lift.simulate_lift("stoop", duration=1.6)          # MuJoCo inverse dynamics
print(r.peak()["hip"], r.balanced())
n, loss = A.best_ratio(A.MOTORS[1], r.tau[:, 2] / 2, r.qd[:, 2], 0 * r.qd[:, 2])
```

> [!NOTE]
> The motor parameters are **illustrative** frameless-BLDC values spanning a realistic range of motor constant. Swap in datasheet values for a real design. Torques are for the lumped symmetric model, so divide by 2 per side (the tables above already do).

Derivations are in [DERIVATIONS.md](DERIVATIONS.md).

## About

Built by **Aneesh Karavadi**, an engineering student at the University of North Texas (TAMS), with CAD in SolidWorks, Fusion 360 and Onshape and a background in competition physics. I used **Claude Code** as a pair programmer. The modelling choices and conclusions are mine to defend.
