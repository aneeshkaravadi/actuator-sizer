# Derivations

## 1. Inverse dynamics (`lift.simulate_lift`)

The equations of motion of a rigid multibody are

$$M(q)\,\ddot q + c(q,\dot q) + g(q) = \tau$$

where $M$ is the mass matrix, $c$ holds the Coriolis and centrifugal terms, and $g$ is gravity. Given a planned $q(t)$ we know $\dot q$ and $\ddot q$, so the required torque follows directly. MuJoCo's `mj_inverse` evaluates the left side with the recursive Newton–Euler algorithm.

Contacts are disabled. The feet are bolted to the world, and balance is checked separately (section 3).

**Check:** over a lift that starts and ends at rest, $\int \tau\cdot\dot q\,dt = \Delta PE$ to within 1% (`test_lift_work_equals_potential_energy_gain`).

## 2. Trajectory

The start and end postures come from inverse kinematics (`scipy.optimize.least_squares`). The objective has three parts:

| term | weight | purpose |
|---|---|---|
| grip-point error | heavy | reach the tote handle |
| whole-body COM offset from mid-foot | very heavy | stay balanced |
| distance from a "style" posture | light | squat: bent knees, upright trunk; stoop: straight legs, bent hip |

Between the two postures each joint follows a minimum-jerk profile:

$$q(s) = q_0 + (q_1-q_0)(10s^3 - 15s^4 + 6s^5), \qquad s = t/T$$

Velocity and acceleration are zero at both ends, so the motion starts and stops smoothly.

## 3. Balance: center of pressure

The feet are fixed at the ankle, so the floor must supply the ankle torque through a shift of the center of pressure (CoP):

$$x_\text{CoP} = -\tau_\text{ankle} / F_z, \qquad F_z \approx m_\text{total}\,g$$

The lift is **balanced** only if $x_\text{CoP}$ stays between heel and toe at every instant. A static forward lean puts the CoP exactly under the center of mass, which is a sign check.

The fastest balanced lift is found by bisection on the duration $T$.

## 4. Motor and gearbox

**Reflecting torque through the gearbox.** The joint torque divides by the ratio and efficiency. The rotor also has to be accelerated, and it spins $N$ times faster than the joint:

$$\tau_m = \frac{\tau_j}{N\eta} + J_m N \ddot q_j$$

**Copper loss** is $P_{cu} = I^2 R = \tau_m^2 / K_m^2$, with the motor constant $K_m = K_t/\sqrt R$. $K_m$ is the single best figure of merit for "torque per watt of heat".

**Feasibility** has two limits:
- peak current: $|\tau_m| \le K_t I_\text{max}$
- voltage: with $V = K_e\omega + IR$ and $K_e = K_t$ in SI units, the reachable speed is $\omega_m \le \omega_0 - \tau_m R/K_t^2$

**Optimal ratio for a pure inertia.** If $\tau_j = J_L\ddot q$, then $\tau_m = \ddot q\,(J_L/N + J_m N)$. Setting $d/dN = 0$ gives $N^* = \sqrt{J_L/J_m}$: the reflected rotor inertia then equals the load inertia (`test_inertia_matched_ratio_minimizes_loss_for_pure_inertia`).

**Gravity-dominated tasks.** These have no such optimum, since the loss just keeps falling as $1/N^2$. The ratio is then capped by the speed needed for *other* tasks and by impact and backdrivability ($N^2 J_m$). That is why the README shows a trade space rather than a single answer.

## 5. Winding temperature

The winding is modeled as a single thermal mass:

$$C\,\dot T = P_{cu}\,[1 + \alpha(T - 25)] - \frac{T - T_\text{amb}}{R_{th}}$$

The resistance of copper rises with temperature ($\alpha = 0.00393$/K). At steady state, with the cycle-averaged loss $P$:

$$T = \frac{T_\text{amb} + R_{th}P(1 - 25\alpha)}{1 - R_{th}P\alpha}$$

This goes to infinity as $R_{th}P\alpha \to 1$: **thermal runaway**, where heating grows faster than cooling.

## 6. Series elastic actuator impact

Model it as two inertias joined by the series spring $k$:
- the rotor, reflected through the gearbox: $N^2 J_m$
- the link: $J_L$

The link hits an obstacle of stiffness $k_e$, and the motor applies no torque during the millisecond-scale event. The quantity that breaks gear teeth is the spring (gearbox) torque.

**Rigid limit** ($k \to \infty$). Both inertias decelerate together, so the contact torque peaks at $\omega_0\sqrt{k_e(J_L + N^2J_m)}$. The gearbox carries the rotor's share of that:

$$\tau_\text{gear} = \omega_0\, N^2J_m\sqrt{\frac{k_e}{J_L + N^2J_m}}$$

**Compliant case.** A soft spring decouples the rotor for the first fraction of a period, so the gearbox sees much less.

**Cost of the spring.** It deflects $\tau/k$ under load, which costs position accuracy and lowers the frequency of the rotor–spring resonance, $\sqrt{k/N^2J_m}$.

## Limitations

- Sagittal plane only, with left and right sides lumped and the feet bolted down.
- Contact forces with the tote and floor are not resolved.
- Gearbox efficiency is a constant, and friction and backlash are not modeled.
- The motors are illustrative, not catalogue parts.
- The CoP uses a quasi-static vertical load.
