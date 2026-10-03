# For the Apptronik team

Apollo moves totes, so this repo starts from that task and works back to actuator requirements:

1. **Task to joint requirements.** MuJoCo inverse dynamics of a 25 kg tote lift gives per-joint torque, speed and power, plus a centre-of-pressure balance check. The stoop lift needs 221 N m and 395 W peak at the hip, per side. ([figure](figures/lift_torques.png))
2. **Balance limits throughput.** The fastest balanced squat lift (3.9 s) takes 2.6× as long as a stoop lift (1.5 s), because rising out of a deep squat pushes the CoP behind the heel.
3. **Gear ratio as a trade, not an optimum.** Copper loss against reflected inertia, capped by a speed requirement. A tote-shuttle thermal study shows which joints overheat first at low ratios: the hip, then the shoulder holding the tote. ([figure](figures/shuttle_thermal.png))
4. **Series elasticity.** For a stiff impact, a 3000 N m/rad spring cuts the gearbox spike from 470 to 98 N m at N = 120, but does nothing useful at N = 20. ([figure](figures/sea_impact.png))

**What it doesn't do:** 3D whole-body dynamics, walking, real Apollo parameters, or real motor datasheets (the motors are illustrative).

**A 3-week project I could do for you, remote:** run your task library (or public proxies) through the same pipeline to produce a per-joint requirement envelope and a thermal duty map for a chosen actuator set. It shows which joint limits sustained throughput, and by how much a motor or ratio change would move it.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
