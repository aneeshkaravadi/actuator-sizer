# 60-second demo video script

| Time | Show | Say |
|---|---|---|
| 0–10 s | `postures.png` | "actuator-sizer: I take a humanoid task, lifting a 25 kg tote, and work back to what every motor and gearbox has to do." |
| 10–22 s | `lift_torques.png` + `pytest -q` | "MuJoCo inverse dynamics gives the torque at every joint. The test suite checks that total joint work equals the potential-energy gain within 1%." |
| 22–32 s | postures again, CoP bar | "Balance turned out to be the real speed limit: a squat lift has to be 2.6 times slower than a stoop, or the centre of pressure leaves the foot." |
| 32–45 s | `shuttle_thermal.png` | "Repeating the task 120 times an hour shows which motor overheats first, and below what gear ratio." |
| 45–55 s | `sea_impact.png`, then the STEP file in Fusion | "A series spring cuts the gearbox shock by almost 5x at high ratio, but does nothing at low ratio. And the actuator envelope goes straight to CAD." |
| 55–60 s | repo URL | "Code's on GitHub. I'm Aneesh." |
