# Synchronized tracking experiments

Run in Hydrocore after sourcing the workspace environment:

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
rosrun hofa_mpc_ros run_mpc_experiment.py \
  --scene random_pillars --goal-x 5 --goal-y -2 --goal-depth 1
```

The runner now launches the reference processor and controller with a unique
`experiment_start_topic`. Both are gated from their construction: the reference
holds its schedule at zero and the controller sends no horizontal force or yaw
torque. The recorder subscribes before simulation startup and never discards
the captured startup samples.

Before release the runner requires fresh odometry, a live allocator/PWM stream,
zero horizontal commands, horizontal speed below 0.01 m/s, vertical speed below
0.02 m/s, yaw rate below 0.02 rad/s, and depth within 0.05 m of the target for
one second. It then sends a latched goal and waits for a fresh matching route
and reference window. These checks use a wall-clock timeout (`--ready-timeout`,
default 120 s); a missing publisher or paused ROS clock cannot wait forever.

The runner publishes one future ROS start timestamp, one second ahead to allow
message delivery, and requires acknowledgements from both reference and
controller before that time. Missing/late acknowledgements cancel the start.
This one-second transport margin is not a model-loading sleep. The deprecated
`--startup-wait` argument no longer controls startup.

The controller waits for a reference stamped at or after the common start before
its first horizontal command. Timer periods and computation latency still exist;
we do not claim two ROS callbacks execute in the same CPU instant.

During these experiments the MPC also holds depth using the bounded PI-D used
for single-axis identification (near-level operation). Ordinary launches with an
empty start topic retain their former behaviour. The PID baseline uses the same
start barrier; after release it resumes its existing depth controller.

The accepted experiment route is frozen after scheduling start, and its clock is
not moved backwards by the usual schedule-lead clamp. This gives the controller
and evaluator the same fixed plan. Start a new experiment to evaluate a new route.
Ordinary, ungated reference processing retains replanning and lead-clamp behaviour.

Saved evidence:

- `experiment_start.json`: common ROS epoch and both acknowledgements.
- `prestart_odometry.csv`, `prestart_wrench.csv`: captured startup evidence.
- Scored CSVs: samples from the common epoch onward; `t_rel` uses that epoch.
- `generalized_force.csv` now includes `fz` for depth-control diagnostics.
- Bag recordings include the start and acknowledgement topics.

Both online scoring and `--rescore` read the common epoch. Legacy runs lacking
start metadata retain their old first-odometry convention; their historical
startup timing cannot be repaired by inventing an epoch.

For `--no-launch`, start both nodes in gated mode yourself and pass the SAME
topic as `--experiment-start-topic`. The runner refuses ungated reuse. Do not
use a topic already carrying a latched start from another experiment.

Validation commands (in the ROS environment):

```bash
python3 -m pytest -q src/hofa_mpc_ros/test/test_arc_profile.py \
  src/hofa_mpc_ros/test/test_experiment_start.py
python3 scripts/check_experiment_start.py results/start_sync_validation/RUN_DIR
```

Full MPC validation runs: `run_20260919_041331` and `run_20260919_042811`
under `results/start_sync_validation`. The second explicitly used
`--startup-wait 0`: prestart horizontal wrench was zero, first sample was
3.1 ms after the common start at speed 0.000050 m/s, and the goal was reached.
Recorded depth was 0.935–1.006 m, progress RMSE 0.0121 m, cross-track RMSE
0.00265 m. Offline re-evaluation reproduced the live metrics exactly.

Turn yaw RMSE was still about 1.70 degrees: this change validates startup and
timing, not the remaining turn tracking/model error. MPC gains and identified
vehicle parameters were not changed by this work.
