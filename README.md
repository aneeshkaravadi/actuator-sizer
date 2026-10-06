# actuator-sizer

[![tests](https://github.com/aneeshkaravadi/actuator-sizer/actions/workflows/ci.yml/badge.svg)](https://github.com/aneeshkaravadi/actuator-sizer/actions/workflows/ci.yml)

How big does a humanoid robot's hip motor need to be if its job is moving 25 kg totes all day? This repo works it out from the task backwards: simulate the lift, get the torque every joint needs, then pick motors and gear ratios and see what overheats.

<img src="docs/figures/postures.png" width="75%">

## The model

It's a human-sized robot (1.73 m, 73 kg) in MuJoCo, simplified to a side view with both legs and both arms lumped together, since a two-handed lift is symmetric. Segment lengths and masses come from human anthropometry tables (Winter), which is roughly where human-scale humanoids end up anyway. The robot plans a pick posture and a finish posture with inverse kinematics, blends between them with a minimum-jerk profile, and MuJoCo's inverse dynamics tells me the torque at each joint at every instant. The feet are bolted down, so I check balance separately by tracking where the center of pressure lands under the foot.

The first sanity check I trusted: over a whole lift, the total work done by all the joints matches the gain in potential energy to within 1%.

## What one lift needs

Per side, for a stoop lift in 1.5 s, the hip is doing almost all the work: 221 N·m peak and almost 400 W. A squat lift cuts hip torque roughly in half, which is the "lift with your legs" advice working the way you'd expect.

![Joint torques](docs/figures/lift_torques.png)

## Balance turned out to be the real limit

This was the surprise. When I tried to speed up the squat lift, the center of pressure ran off the back of the heel, because rising quickly out of a deep squat throws the body backwards. The fastest squat that stays balanced takes 3.9 s, against 1.5 s for the stoop. So for a robot with feet this size, lifting with your legs costs you 2.6 times the cycle time, which is a pretty big deal if the whole point is throughput.

## Picking a gear ratio

I expected to find an optimal gear ratio and got something else. For a slow lift that's mostly fighting gravity, motor heat keeps going down the higher you make the ratio, forever. What actually stops you is everything else a high ratio costs: the joint can't move fast anymore, and the motor's rotor inertia gets multiplied by N², which you feel in every collision. So the plot on the left is a trade space, not an answer, and the dotted lines are where a 6 rad/s speed requirement cuts you off.

<img src="docs/figures/hip_ratio_trade.png" width="49%"> <img src="docs/figures/shuttle_thermal.png" width="49%">

On the right, the same mid-size motor at every joint does 120 totes an hour (lift, carry, lower, walk back). The hip overheats first, below about 12:1, and the shoulder holding the tote out in front is next at about 9.5:1. That's interesting because the low ratios that make a robot backdrivable and safe around people (6 to 10:1) are right where these two joints run out of thermal margin.

## Do series springs help?

I modeled the hip running into something stiff at 1 rad/s, with a spring between the gearbox and the leg. At a 120:1 ratio a 3000 N·m/rad spring cuts the shock on the gearbox from 470 to 98 N·m, but at 20:1 it actually makes it slightly worse, because there's not much rotor inertia to protect against in the first place. The cost of the spring is 0.074 rad of deflection at peak hip torque.

<img src="docs/figures/sea_impact.png" width="55%">

There's also a quick air-core vs iron-core motor comparison ([figure](docs/figures/aircore_vs_iron.png)) and a CAD envelope for the hip actuator: motor bay, gearbox bay and output flange ([STEP](cad/hip_actuator_envelope.step)).

## What the gearbox costs

I first used a constant 90% gearbox efficiency. A real planetary gearbox adds a stage for every 10:1 or so, and each stage costs a few percent, so I made efficiency step down with the number of stages (0.97 per stage, an assumption) and counted the gearbox's own friction heat too. The winding heat still keeps falling with ratio, but the total doesn't. Past about 100:1 the gearbox makes most of the heat. The total is 18.4 W at 100:1, jumps to 24.5 W once a third stage is needed, and is still 18 W at 300:1.

Lowering a tote runs the gearbox backwards, with the load driving the motor, and then friction helps hold the load instead of fighting the motor. For friction-type losses the backwards efficiency is $2 - 1/\eta$, so the motor needs less torque going down than coming up. The shuttle numbers above include that.

A strain-wave gearbox gets a high ratio in one stage with essentially no backlash, but at an assumed 75% efficiency it makes 2.6 times the heat of a two-stage planetary at 50:1 (104 W against 40 W), most of it in the gearbox. Backlash barely depends on the ratio, because each stage's play reaches the output divided by the ratio of the stages after it, so the last stage sets it. At 50:1 that's about 11 arcmin, which lets the hands wander about 3 mm at the hip's 0.92 m reach.

<img src="docs/figures/gearbox_tradeoffs.png" width="95%">

<!-- TODO(Aneesh): replace or add next to the generated envelope with your own SolidWorks model, e.g.
<img src="docs/photos/hip_actuator_solidworks.png" width="60%">
and put the native file in cad/solidworks/
-->

## Things I got wrong first

- My first inverse dynamics run said the hip needed about 40,000 N·m. The body and the tote were sinking into the floor in the stoop posture and MuJoCo was adding contact forces. Turning contacts off (the feet are fixed anyway) fixed it.
- My "squat" inverse kinematics kept quietly turning into a stoop, because it started from a standing pose and found the nearest answer. Seeding it from a squat posture fixed that.
- Both lifts originally put the center of pressure past the toes, until I made balance a much stronger term in the posture solver and moved the tote closer to the body.
- My first thermal study used a 120:1 ratio everywhere and nothing came close to overheating, which told me nothing. Sweeping the ratio is what made it useful.

## Running it

```bash
pip install -e ".[dev,cad]"
pytest -q
python examples/make_figures.py
```

The motor parameters are illustrative values spread across a realistic range, not catalogue parts, so swap in a datasheet for a real design. The physics is written out in [DERIVATIONS.md](DERIVATIONS.md). Open questions and next steps are in [issues](https://github.com/aneeshkaravadi/actuator-sizer/issues).

---

Aneesh Karavadi, dual-enrolled engineering student at UNT through TAMS. I do most of my CAD in SolidWorks, Fusion and Onshape. I used Claude Code to write a lot of the implementation, but the questions, the checks and the conclusions are mine.
