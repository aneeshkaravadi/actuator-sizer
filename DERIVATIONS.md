# Derivations

## 1. Inverse dynamics (`lift.simulate_lift`)

The equations of motion of a rigid multibody are

$$M(q)\,\ddot q + c(q,\dot q) + g(q) = \tau$$

where $M$ is the mass matrix, $c$ holds the Coriolis and centrifugal terms, and $g$ is gravity. Given a planned $q(t)$ we know $\dot q$ and $\ddot q$, so the required torque follows directly. MuJoCo's `mj_inverse` evaluates the left side with the recursive Newton–Euler algorithm.

Contacts are disabled. The feet are bolted to the world, and balance is checked separately (section 3).

**Check:** over a lift that starts and ends at rest, $\int \tau\cdot\dot q\,dt = \Delta PE$ to within 1% (`test_lift_work_equals_potential_energy_gain`).

**Segments.** Lengths are fractions of height and masses fractions of body mass from Dempster's data as tabulated by Winter (*Biomechanics and Motor Control of Human Movement*, Table 4.1, as copied in BMClab's `data/BSP_DempsterWinter.txt` on github.com/BMClab/BMC): shank 0.246 H and 0.0465 M per leg, thigh 0.245 H and 0.100 M, trunk with head and neck 0.288 H (hip to shoulder) and 0.578 M, upper arm 0.186 H and 0.028 M, forearm and hand 0.146 H and 0.022 M. Winter gives each center of mass as a fraction of the length from the segment's proximal end (knee for the shank, hip for the thigh and trunk, shoulder and elbow for the arm): 0.433, 0.433, 0.66, 0.436 and 0.682. The legs here are built from the ankle up, so their centers of mass sit $1 - 0.433$ of the way along from the ankle and knee. Each segment's transverse inertia about its center of mass is $m(r_g L)^2$ with Winter's radii of gyration (0.302, 0.323, 0.503, 0.322, 0.468), and the long-axis inertia is that of a cylinder of a guessed radius. The tote is a solid 0.40 × 0.30 × 0.20 m box. The feet are part of the world, so they carry no mass.

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

The feet are fixed at the ankle, so the floor must supply the ankle torque through a shift of the center of pressure (CoP). The foot is massless, so the floor's force $(F_x, F_z)$, acting at the CoP with no moment there, has to make the same moment about the ankle as the torque the foot puts into the shank. With the ankle a height $h$ above the floor (0.08 m in the model):

$$\tau_\text{ankle} = -h F_x - x_\text{CoP} F_z \quad\Rightarrow\quad x_\text{CoP} = -\frac{\tau_\text{ankle} + h F_x}{F_z}, \qquad (F_x, F_z) = M(\ddot x_c,\ g + \ddot z_c)$$

The floor force is not just the weight. It's the whole body's mass times its center of mass's acceleration plus $g$, which MuJoCo gives as the force the world puts into the shank (`cfrc_int` after `mj_rnePostConstraint`). In the 1.77 s stoop the shuttle uses, it swings between 0.89 and 1.12 times the weight. For a single point mass this reduces to the familiar $x_c - z_c\ddot x_c/(g + \ddot z_c)$.

The lift is **balanced** only if $x_\text{CoP}$ stays between heel and toe at every instant.

**Checks:** a still posture puts the CoP exactly under the center of mass. During a lift, the CoP matches the zero-moment point worked out from nothing but how each body moves (the whole body's Newton–Euler equations about a point on the floor), $x_\text{ZMP} = [\sum m_i (x_i(g + \ddot z_i) - z_i\ddot x_i) - \sum I_i\ddot\theta_i] / \sum m_i(g + \ddot z_i)$, with each body's acceleration taken by finite differences of the planned motion, to 0.2 mm (`test_center_of_pressure_matches_the_multibody_zmp_from_kinematics_alone`). My first version used $x_\text{CoP} = -\tau_\text{ankle}/(Mg)$, which is off by up to 20 mm in that stoop and fails that test. With it, the fastest balanced stoop and squat came out at 1.55 and 4.21 s. With the full formula they are 1.72 and 4.55 s.

The fastest balanced lift is found by bisection on the duration $T$.

## 4. Motor and gearbox

**Reflecting torque through the gearbox.** The joint torque divides by the ratio and efficiency. The rotor also has to be accelerated, and it spins $N$ times faster than the joint:

$$\tau_m = \frac{\tau_j}{N\eta} + J_m N \ddot q_j$$

**Copper loss** is $P_{cu} = I^2 R = \tau_m^2 / K_m^2$, with the motor constant $K_m = K_t/\sqrt R$. $K_m$ is the single best figure of merit for "torque per watt of heat".

**Iron loss.** maxon treats a brushless motor's speed-dependent stator losses as an extra friction torque $M_R = K_t I_0$, with $I_0$ the no-load current (catalog 2026/27, pp. 90 to 91), so they cost $P_{fe} = K_t I_0\,|\omega_m|$. That's small in a slow lift but not at the rated speed (below).

**Torque per amp at high torque (sensitivity only).** The model keeps $K_t$ constant up to the current cap. To see how much that matters, `Drive(kt_drop=d)` lets the torque per amp fall linearly from $K_t$ at twice the rated torque to $(1-d)K_t$ at the listed stall torque. There's no published curve behind that shape.

**Feasibility** has two limits:
- peak current: $|\tau_m| \le K_t I_\text{max}$
- voltage: with $V = K_e\omega + IR$ and $K_e = K_t$ in SI units, the reachable speed is $\omega_m \le \omega_0 - \tau_m R/K_t^2$

The shuttle's minimum ratios have to pass both, for the lift and for every point of a stride, as well as the heat limit. Each minimum is found by scanning 160 ratios from 4 to 300:1 and bisecting the first crossing.

**Optimal ratio for a pure inertia.** If $\tau_j = J_L\ddot q$, then $\tau_m = \ddot q\,(J_L/N + J_m N)$. Setting $d/dN = 0$ gives $N^* = \sqrt{J_L/J_m}$: the reflected rotor inertia then equals the load inertia (`test_inertia_matched_ratio_minimizes_loss_for_pure_inertia`).

**Gearbox efficiency against ratio (`actuator.Gearbox`).** A planetary gearbox does at most about 10:1 per stage, so a ratio $N$ needs $k = \lceil \log N/\log 10 \rceil$ stages, and with per-stage efficiency $\eta_s$ (0.97 assumed) the gearbox runs at $\eta = \eta_s^k$. That steps down at 10:1 and 100:1. A strain-wave gearbox does 30 to 160:1 in one stage at a lower, roughly constant efficiency (0.75 assumed).

**Backdriving.** When the load drives the motor ($\tau_j \dot q_j < 0$, like lowering a tote), friction still opposes the motion. If the losses are Coulomb friction, a fraction $1 - \eta$ of the forward input power is lost. Running backwards, the same friction torque subtracts instead of adds, which gives a backwards efficiency $\eta_b = 2 - 1/\eta$. So $\tau_m = \tau_j\eta_b/N$ going down against $\tau_j/(N\eta)$ going up. Below $\eta = 0.5$, $\eta_b \le 0$ and the gearbox self-locks. A gear mesh is a wedge, and an inclined plane with Coulomb friction (slope $\lambda$, friction angle $\varphi$) has $\eta = \tan\lambda/\tan(\lambda+\varphi)$ forward and $\tan(\lambda-\varphi)/\tan\lambda$ backward; for friction coefficients of a few percent that agrees with $2 - 1/\eta$ to second order, and it self-locks at the same 50% (`test_backdriven_efficiency_agrees_with_an_inclined_plane_with_coulomb_friction`). The heat made in the gearbox is $P_j(1/\eta - 1)$ forward and $|P_j|(1 - \eta_b)$ backward, so motor shaft power always equals joint power plus gearbox heat (`test_gearbox_power_balance_both_ways`). Static holding is taken at the forward value, which is conservative.

**Lowering.** Played backwards, a trajectory has the same $q$ and $\ddot q$ at each posture and $\dot q$ flipped. The Coriolis terms are quadratic in $\dot q$, so with no damping the joint torques are identical, and only the direction of power flow changes. The shuttle's lowering phase reuses the lift's torques with $\dot q \to -\dot q$.

**Backlash.** If each stage has play $b$ at its own output, the play of stage $i$ (counting back from the output) reaches the output divided by the ratio of the $i$ stages after it, $b_\text{out} = \sum_i b/n_s^i$. With 10 arcmin per stage (assumed), two 10:1 stages give 11 arcmin. The last stage sets it, so it barely grows with $N$.

**Gravity-dominated tasks.** These have no such optimum, since the loss just keeps falling as $1/N^2$. The ratio is then capped by the speed needed for *other* tasks and by impact and backdrivability ($N^2 J_m$). That is why the README shows a trade space rather than a single answer.

## 5. Winding temperature

The winding is modeled as a single thermal mass:

$$C\,\dot T = P_{cu}\,[1 + \alpha(T - 25)] + P_{fe} - \frac{T - T_\text{amb}}{R_{th}}$$

The resistance of copper rises with temperature ($\alpha = 0.00393$/K, which maxon also uses). The iron loss $P_{fe}$ goes through the same thermal resistance, the way maxon's own simplified model lumps the losses. At steady state, with the cycle-averaged losses $P = \bar P_{cu}$ and $Q = \bar P_{fe}$:

$$T = \frac{T_\text{amb} + R_{th}P(1 - 25\alpha) + R_{th}Q}{1 - R_{th}P\alpha}$$

This goes to infinity as $R_{th}P\alpha \to 1$: **thermal runaway**, where heating grows faster than cooling.

**Two nodes.** maxon's datasheets give two thermal resistances (winding to housing, $R_{wh}$, and housing to ambient, $R_{ha}$) and two time constants. They describe a winding with heat capacity $C_w = \tau_w/R_{wh}$ sitting on a housing with $C_h = \tau_m/R_{ha}$:

$$C_w\dot T_w = P\,[1 + \alpha(T_w - 25)] + Q - \frac{T_w - T_h}{R_{wh}}, \qquad C_h\dot T_h = \frac{T_w - T_h}{R_{wh}} - \frac{T_h - T_\text{amb}}{R_{ha}}$$

In steady state the two resistances are just in series, so the closed form above holds with $R_{th} = R_{wh} + R_{ha}$. In the first seconds the winding barely feels the housing and heats at about $P/C_w$, which is much faster than the one-node model says (both checked in `tests/test_actsizer.py`).

**Within a cycle.** The shuttle analysis uses each cycle's average loss. Running the two-node model through the actual 30 s cycle (lift, loaded carry, lower, empty walk back, with each walk's losses averaged over a 1.06 s stride) shows the hip winding swinging 11 K, with its peak 5 K above the cycle mean. The mean itself matches the steady value for the average loss within a degree (`test_two_node_winding_under_a_repeating_load_averages_to_the_steady_value`), so the peak-based limit only moves from 50.5 to 51.6:1.

**Datasheet values.** The tests check them against each other with maxon's own relations (catalog 2026/27, pp. 90 to 91): $K_t = 60/(2\pi K_v)$ in SI, the no-load speed just under $K_v V$, the stall current equal to $V/R$, the speed/torque gradient equal to $R/K_t^2$, the mechanical time constant equal to $RJ/K_t^2$ (rotor inertia), and the maximum efficiency equal to $(1 - \sqrt{I_0/I_A})^2$ (no-load current). maxon's rule is also that the stall torque is $K_t I_A$, but for these kits it lists about a quarter of that, and it says its frameless kits' characteristics depart from strictly linear (catalog p. 78; "Technology, short and to the point", 2023). My reading is that the iron saturates, so the current is capped at $\tau_\text{stall}/K_t$ instead of $I_A$. That's an inference: maxon doesn't publish the curve.

**The thermal values against maxon's rating.** maxon defines the nominal current as the continuous current that takes the winding to its maximum temperature (155 °C) at 25 °C ambient (catalog p. 92), at the nominal speed. So the model at that point is a check on the thermal resistances that doesn't use anything fitted:

| kit | copper only | with iron loss | iron loss on the housing only |
|---|---|---|---|
| HT 60 M | 104.5 °C | 152.8 °C | 131.2 °C |
| HT 76 M | 99.2 °C | 161.1 °C | 131.6 °C |
| HT 90 M | 107.8 °C | 164.0 °C | 135.0 °C |

Copper alone misses by 47 to 56 K. With the iron loss through the whole thermal resistance, the misses are -2, +6 and +9 K. The test (`test_maxon_rated_current_brings_the_winding_to_its_limit`) allows 10 K, 8% of the 130 K rise, because maxon's own numbers differ by about that much: the HT 76 M's product page gives $R_{th1}$ = 1.37 K/W and the catalog 1.26, and the catalog's rated torques and currents differ slightly from the product pages too. Putting the iron loss on the housing only brackets it from below. So the thermal data hang together, and copper loss alone isn't enough to use them.

**Mounting.** maxon's catalog (pp. 256, 258 and 260) gives a range of housing-to-ambient resistance for each kit: 2.0 to 4 K/W (HT 60 M), 1.5 to 2.9 (HT 76 M) and 1.0 to 2.3 (HT 90 M). The low value, which the rated data and the product pages use, is the kit built into an aluminium housing of 73, 89 or 103 mm that is bolted to an aluminium plate (150 × 150 × 6, 190 × 190 × 10 or 225 × 225 × 10 mm). The high value is the same housing without the plate. The code carries both (`MaxonDatasheet.motor("plate")` and `motor("housing")`), changing only $R_{ha}$ and keeping the housing's heat capacity.

## 6. Series elastic actuator impact

Model it as two inertias joined by the series spring $k$:
- the rotor, reflected through the gearbox: $N^2 J_m$
- the link: $J_L$

The link hits an obstacle of stiffness $k_e$, and the motor applies no torque during the millisecond-scale event. The quantity that breaks gear teeth is the spring (gearbox) torque.

**Rigid limit** ($k \to \infty$). Both inertias decelerate together, so the contact torque peaks at $\omega_0\sqrt{k_e(J_L + N^2J_m)}$. The gearbox carries the rotor's share of that:

$$\tau_\text{gear} = \omega_0\, N^2J_m\sqrt{\frac{k_e}{J_L + N^2J_m}}$$

With the hip carrying the tote, $J_L = 6.6$ kg m² per side (the hip's entry in the model's mass matrix, found by inverse dynamics with a unit hip acceleration; 3.9 standing without the tote and 18.2 stooped over it). The 120:1 case then gives 110 N·m. The simulation with a 10⁶ N·m/rad spring still gives 125 N·m, because that spring is only 5 times stiffer than the obstacle, so the figure draws the closed form for "no spring".

**Compliant case.** A soft spring decouples the rotor for the first fraction of a period, so the gearbox sees less. A medium-stiff one can see more than no spring at all, because the rotor bounces on it; at 120:1 that peaks at 189 N·m near $5 \times 10^4$ N·m/rad.

**Cost of the spring.** It deflects $\tau/k$ under load, which costs position accuracy and lowers the frequency of the rotor–spring resonance, $\sqrt{k/N^2J_m}$.

## 7. Push recovery and the hip speed (`recovery.py`)

**Linear inverted pendulum** (Kajita et al. 2001). With the center of mass held at height $z_0$ and the center of pressure at $p$:

$$\ddot x = \omega^2 (x - p), \qquad \omega = \sqrt{g/z_0}$$

$$x(t) = p + (x_0 - p)\cosh\omega t + \frac{\dot x_0}{\omega}\sinh\omega t$$

$z_0 = 1.05$ m and the hip height (the swing leg's length, 0.93 m) come from the MuJoCo model standing straight.

**Capture point** (Pratt et al. 2006). $\xi = x + \dot x/\omega$ obeys $\dot\xi = \omega(\xi - p)$, so with $p$ fixed it runs away exponentially. If the center of pressure sits exactly on $\xi$, the robot comes to rest over it. Starting over the ankle, the ankles alone can stop a push if $\xi_0 = \dot x_0/\omega$ is still inside the foot, so $\dot x_0 \le \omega\, x_\text{toe} = 0.58$ m/s.

**One step.** Harder pushes need a step. Until touchdown at time $T$ the center of pressure stays at the toe, and the new foot goes where $\xi(T)$ lands mid-foot. With the hip above the center of mass, the swing leg reaches touchdown at angle $\theta = \arcsin\big((x_\text{foot} - x(T))/L\big)$, and it isn't reachable at all if that argument passes 1. A minimum-jerk swing peaks at 15/8 of its average speed, so the hip needs $\dot\theta_\text{max} = \tfrac{15}{8}\,\theta/T$. Short swings need fast hips and long swings need long strides, so for a given push and maximum stride the code searches $T$ for the slowest hip that works (`required_hip_speed`). The search starts at an 80 ms swing (`MIN_SWING`), which is an assumption: with a 0.5 m cap, a 1.5 m/s push needs a 68 ms swing at 12.4 rad/s, so the figure shows it as impossible.

**Checks:** the closed form against numerical integration. The center of pressure on the capture point brings the robot to rest, and 1 cm short it runs away. The required speed rises with the push and as the allowed stride shrinks. The minimum-jerk peak is 15/8 of the average (`tests/test_actsizer.py`).

## 8. One-handed and twisting lifts (`asym.py`)

The legs stay lumped, but three hinges at the hip let the trunk lean sideways (roll, about $x$), bend forward (pitch, about $y$) and twist (yaw, about its own long axis, since the yaw joint comes after the pitch). One arm, at the shoulder's sideways offset $y_s = 0.20$ m, has a shoulder and an elbow and carries the tote. The other hangs at $-y_s$ with no joints. The IK reaches a 3D grip point and keeps the center of mass over the middle of the feet fore and aft. Sideways the feet are apart, so it only pushes back once the center of mass is more than 8 cm off center.

**Checks.** With the arms hanging, their weights cancel side to side, so standing straight the hip roll joint carries only the tote's moment, $m_\text{tote}\, g\, y_s$ = 49.05 N·m. Bent 90° forward with the loaded arm plumb, the trunk's twist axis is horizontal and the same moment becomes a twisting torque. Both match MuJoCo's inverse dynamics to $10^{-6}$ (`test_one_handed_static_moments_match_hand_calculations`). Over a slow twisting lift, the joint work equals the potential energy gained (`test_twisting_lift_work_equals_potential_energy_gain`).

## 9. Walking loads (`gait.py`)

The shuttle's carry and walk back are each 13.2 s of walking. The leg joints' loads there come from human gait data: Fukuchi, Fukuchi and Duarte, "A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals", PeerJ 6:e4640 (2018), doi:10.7717/peerj.4640, data doi:10.6084/m9.figshare.5722711 (CC BY 4.0). I took the processed comfortable-speed overground files (`*walkOCknt.txt` and `*walkOCang.txt`) for all 42 people, averaged the right and left legs and then the people, and kept one mean stride (101 points) of the sagittal hip, knee and ankle moments (internal, N·m per kg of body mass), the joint angles, and one leg's vertical floor force. The sign conventions are from the dataset's own analysis script. The stride time, 1.06 s, is my estimate from the heel marker in one comfortable trial per person (heel strike where the heel's forward speed falls through 20% of the walking speed), since the processed files are time-normalized.

During a walk, each leg joint's torque per side is the standing posture's torque from MuJoCo plus the human moment's swing about its stride mean, times the mass the legs carry (98 kg with the tote, 73 kg without):

$$\tau(t) = \tau_\text{stand} + m\,[\hat M(t) - \overline{\hat M}]$$

Removing the mean keeps the tote's forward offset from being counted twice. Joint speed and acceleration are the angle curve differentiated as a periodic Fourier series, so the rotor's reflected inertia and the iron loss see real walking speeds (the knee swings at up to 6.7 rad/s). The arms get no walking load.

**Checks:** one leg's vertical floor force averages to half the body weight over a stride, within 3% (it's 4.86 N/kg against 4.905), which confirms the data are per kg and cover a whole stride. In the robot's sign convention the hip pushes the trunk back in early stance and the ankle pushes off in late stance, matching the signs MuJoCo needs to hold a forward lean. The speeds integrate back to the angles, and with forward drive the average copper loss of standing torque plus swing equals the standing loss plus the swing's mean square, since the cross term averages out (`tests/test_actsizer.py`). This is human walking scaled by mass, not a robot gait, so it's a stand-in until there's a real one.

## Limitations

- The main model is sagittal, with left and right sides lumped and the feet bolted down. The one-handed model adds hip roll and trunk twist but still lumps the legs, so it can't split the hip roll between the two hips.
- Contact forces with the tote and floor are not resolved.
- Gearbox efficiency comes from a stage count with assumed per-stage values and pure Coulomb friction, with no speed- or load-dependent losses. Backlash is reported but not fed into the dynamics.
- The motors' thermal resistance to ambient depends on the mounting, and maxon's two values bracket a real joint rather than describe it. The iron loss follows maxon's friction-torque treatment and goes through the whole thermal resistance.
- The torque constant is linear up to the stall-torque cap, with a sensitivity for torque per amp falling at high torque but no measured curve.
- The walking loads are human moments scaled by mass, for comfortable-speed walking without a load. The arms don't see the tote bouncing as the robot walks.
- Push recovery uses a point-mass pendulum at constant height, with no reaction delay and no swing-leg dynamics, so its speeds are a lower bound.
