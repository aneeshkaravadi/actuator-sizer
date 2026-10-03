# For the Persona AI team

Shipyard work means heavy, repetitive handling, so the part of this repo that matters is the link from a task to actuator heat and balance:

- MuJoCo inverse dynamics of a 25 kg lift, with a centre-of-pressure balance check. The fastest balanced lift depends strongly on lifting style (1.5 s stoop vs 3.9 s squat). ([figure](figures/postures.png))
- A repeated-duty thermal model: which joint's winding hits 120 °C first, and below what gear ratio. ([figure](figures/shuttle_thermal.png))
- A series-elastic impact study: when a spring protects the gearbox and when it doesn't. ([figure](figures/sea_impact.png))

**A 3-week project I could do for you, remote:** extend the model to a heavier industrial payload and a carry-while-walking duty cycle, and produce a per-joint thermal and torque envelope for a shipyard task set.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
