# For the Agility Robotics team

Your legs grew out of compliant, spring-based designs, so start with the series-elasticity result:

- **Impact study.** A two-inertia model (reflected rotor, series spring, link) hits a stiff obstacle at 1 rad/s.

  | Gear ratio | Rigid spike | With a 3000 N m/rad spring |
  |---|---|---|
  | N = 120 | 470 N m | 98 N m |
  | N = 20 | 18 N m | 21 N m |

  At N = 20 the spring slightly *adds* to the spike, because the reflected rotor is already light. It checks against the closed-form rigid limit. ([figure](figures/sea_impact.png))
- **Task-driven sizing.** MuJoCo inverse dynamics of a tote lift (Digit's core job) gives per-joint torque and power, and a balance-limited minimum lift time.

**A 3-week project I could do for you, remote:** a stiffness-selection study for one joint. It sweeps series stiffness against impact load, force-control bandwidth and deflection for a given duty cycle, then outputs a recommended stiffness range with the trade curves.

— Aneesh Karavadi · aneesh.karavadi@gmail.com
