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

**Gearbox efficiency against ratio (`actuator.Gearbox`).** A planetary gearbox does at most about 10:1 per stage, so a ratio $N$ needs $k = \lceil \log N/\log 10 \rceil$ stages, and with per-stage efficiency $\eta_s$ (0.97 assumed) the gearbox runs at $\eta = \eta_s^k$. That steps down at 10:1 and 100:1. A strain-wave gearbox does 30 to 160:1 in one stage at a lower, roughly constant efficiency (0.75 assumed).

**Backdriving.** When the load drives the motor ($\tau_j \dot q_j < 0$, like lowering a tote), friction still opposes the motion. If the losses are Coulomb friction, a fraction $1 - \eta$ of the forward input power is lost. Running backwards, the same friction torque subtracts instead of adds, which gives a backwards efficiency $\eta_b = 2 - 1/\eta$. So $\tau_m = \tau_j\eta_b/N$ going down against $\tau_j/(N\eta)$ going up. Below $\eta = 0.5$, $\eta_b \le 0$ and the gearbox self-locks. The heat made in the gearbox is $P_j(1/\eta - 1)$ forward and $|P_j|(1 - \eta_b)$ backward, so motor shaft power always equals joint power plus gearbox heat (`test_gearbox_power_balance_both_ways`). Static holding is taken at the forward value, which is conservative.

**Lowering.** Played backwards, a trajectory has the same $q$ and $\ddot q$ at each posture and $\dot q$ flipped. The Coriolis terms are quadratic in $\dot q$, so with no damping the joint torques are identical, and only the direction of power flow changes. The shuttle's lowering phase reuses the lift's torques with $\dot q \to -\dot q$.

**Backlash.** If each stage has play $b$ at its own output, the play of stage $i$ (counting back from the output) reaches the output divided by the ratio of the $i$ stages after it, $b_\text{out} = \sum_i b/n_s^i$. With 10 arcmin per stage (assumed), two 10:1 stages give 11 arcmin. The last stage sets it, so it barely grows with $N$.

**Gravity-dominated tasks.** These have no such optimum, since the loss just keeps falling as $1/N^2$. The ratio is then capped by the speed needed for *other* tasks and by impact and backdrivability ($N^2 J_m$). That is why the README shows a trade space rather than a single answer.

## 5. Winding temperature

The winding is modeled as a single thermal mass:

$$C\,\dot T = P_{cu}\,[1 + \alpha(T - 25)] - \frac{T - T_\text{amb}}{R_{th}}$$

The resistance of copper rises with temperature ($\alpha = 0.00393$/K). At steady state, with the cycle-averaged loss $P$:

$$T = \frac{T_\text{amb} + R_{th}P(1 - 25\alpha)}{1 - R_{th}P\alpha}$$

This goes to infinity as $R_{th}P\alpha \to 1$: **thermal runaway**, where heating grows faster than cooling.

**Two nodes.** maxon's datasheets give two thermal resistances (winding to housing, $R_{wh}$, and housing to ambient, $R_{ha}$) and two time constants. They describe a winding with heat capacity $C_w = \tau_w/R_{wh}$ sitting on a housing with $C_h = \tau_m/R_{ha}$:

$$C_w\dot T_w = P\,[1 + \alpha(T_w - 25)] - \frac{T_w - T_h}{R_{wh}}, \qquad C_h\dot T_h = \frac{T_w - T_h}{R_{wh}} - \frac{T_h - T_\text{amb}}{R_{ha}}$$

In steady state the two resistances are just in series, so the closed form above holds with $R_{th} = R_{wh} + R_{ha}$. In the first seconds the winding barely feels the housing and heats at about $P/C_w$, which is much faster than the one-node model says (both checked in `tests/test_actsizer.py`).

**Within a cycle.** The shuttle analysis uses each cycle's average loss. Running the two-node model through the actual 30 s cycle (lift, loaded carry, lower, empty walk back) shows the hip winding swinging 14 K, with its peak 6 K above the cycle mean. The mean itself matches the steady value for the average loss within a degree (`test_two_node_winding_under_a_repeating_load_averages_to_the_steady_value`), so the peak-based limit only moves from 42.5 to 43:1.

**Datasheet values.** The tests check them against each other: $K_t = 60/(2\pi K_v)$ in SI, the no-load speed just under $K_v V$, and the stall current equal to $V/R$. The listed stall torque is far below $K_t I_\text{stall}$, because the iron saturates, so the current is capped at $\tau_\text{stall}/K_t$ instead of $I_\text{stall}$.

## 6. Series elastic actuator impact

Model it as two inertias joined by the series spring $k$:
- the rotor, reflected through the gearbox: $N^2 J_m$
- the link: $J_L$

The link hits an obstacle of stiffness $k_e$, and the motor applies no torque during the millisecond-scale event. The quantity that breaks gear teeth is the spring (gearbox) torque.

**Rigid limit** ($k \to \infty$). Both inertias decelerate together, so the contact torque peaks at $\omega_0\sqrt{k_e(J_L + N^2J_m)}$. The gearbox carries the rotor's share of that:

$$\tau_\text{gear} = \omega_0\, N^2J_m\sqrt{\frac{k_e}{J_L + N^2J_m}}$$

**Compliant case.** A soft spring decouples the rotor for the first fraction of a period, so the gearbox sees much less.

**Cost of the spring.** It deflects $\tau/k$ under load, which costs position accuracy and lowers the frequency of the rotor–spring resonance, $\sqrt{k/N^2J_m}$.

## 7. Push recovery and the hip speed (`recovery.py`)

**Linear inverted pendulum** (Kajita et al. 2001). With the center of mass held at height $z_0$ and the center of pressure at $p$:

$$\ddot x = \omega^2 (x - p), \qquad \omega = \sqrt{g/z_0}$$

$$x(t) = p + (x_0 - p)\cosh\omega t + \frac{\dot x_0}{\omega}\sinh\omega t$$

$z_0 = 1.02$ m and the hip height (the swing leg's length, 0.93 m) come from the MuJoCo model standing straight.

**Capture point** (Pratt et al. 2006). $\xi = x + \dot x/\omega$ obeys $\dot\xi = \omega(\xi - p)$, so with $p$ fixed it runs away exponentially. If the center of pressure sits exactly on $\xi$, the robot comes to rest over it. Starting over the ankle, the ankles alone can stop a push if $\xi_0 = \dot x_0/\omega$ is still inside the foot, so $\dot x_0 \le \omega\, x_\text{toe} = 0.59$ m/s.

**One step.** Harder pushes need a step. Until touchdown at time $T$ the center of pressure stays at the toe, and the new foot goes where $\xi(T)$ lands mid-foot. With the hip above the center of mass, the swing leg reaches touchdown at angle $\theta = \arcsin\big((x_\text{foot} - x(T))/L\big)$, and it isn't reachable at all if that argument passes 1. A minimum-jerk swing peaks at 15/8 of its average speed, so the hip needs $\dot\theta_\text{max} = \tfrac{15}{8}\,\theta/T$. Short swings need fast hips and long swings need long strides, so for a given push and maximum stride the code searches $T$ for the slowest hip that works (`required_hip_speed`).

**Checks:** the closed form against numerical integration. The center of pressure on the capture point brings the robot to rest, and 1 cm short it runs away. The required speed rises with the push and as the allowed stride shrinks. The minimum-jerk peak is 15/8 of the average (`tests/test_actsizer.py`).

## Limitations

- Sagittal plane only, with left and right sides lumped and the feet bolted down.
- Contact forces with the tote and floor are not resolved.
- Gearbox efficiency comes from a stage count with assumed per-stage values and pure Coulomb friction, with no speed- or load-dependent losses. Backlash is reported but not fed into the dynamics.
- The motors are maxon frameless kits, but their thermal resistance to ambient is for maxon's reference mounting, and a real joint housing would differ.
- The CoP uses a quasi-static vertical load.
- Push recovery uses a point-mass pendulum at constant height, with no reaction delay and no swing-leg dynamics, so its speeds are a lower bound.
