# For the Infinitum team

Two parts of this repo connect to air-core motors:

1. **Air-core vs iron-core model** (`actuator.py`). An iron core saturates (its incremental torque constant drops past the knee current) and has speed-dependent core loss. An air-core PCB stator is linear with no core loss. Parameters are illustrative. ([figure](figures/aircore_vs_iron.png))
2. **Duty-cycle-driven sizing.** The repo turns a real task (a humanoid tote lift) into motor torque, speed and copper loss, then into winding temperature. That is the same pipeline you would use to show where an air-core motor's linearity and zero core loss pay off.

**A 3-week project I could do for you, remote:** take datasheet curves for one of your motors and an iron-core competitor, run both through measured or standard duty cycles (HVAC fan, pump, robot joint), and produce an efficiency and thermal comparison report with the code.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
