# Pre-slew desired residual smoothness ablation

This is a new matched training experiment motivated by alternating Fy/Nz desired
commands in the seed1 ROS probes. It does not change the saved 1M-step models.

## Controlled comparison

- A_current_reward_history: original A-group tracking/applied-residual reward.
- B_desired_smooth: same configuration plus desired-residual temporal penalty.
- BOTH use the previous desired residual as three extra observation values,
  normalized by residual limits. Input size is 215; the previous 212-value
  prefix is unchanged. Hidden layers remain [128,128]. This history exposes the
  memory used by the new reward to the actor and critic. The comparison isolates
  the penalty; A is not an exact rerun of the historical 212-input policy.
- Both start fresh SAC learners with the same seed, initialization protocol,
  DAgger checkpoint, physics, warmup, observation, limits and validation rule.
- Default: 2 groups x seeds 0,1,2; 1,000,000 steps per learner; 6 workers;
  validation and checkpoints every 5,000 steps.

With d_t = clip(action_t,-1,1) * residual_limit, the added term is

    -0.02 * dt * [((d_t_Fy-d_prev_Fy)/0.15)^2
                  + ((d_t_Nz-d_prev_Nz)/0.025)^2] / 2

The term uses desired commands BEFORE slew limiting, has no saturation clipping,
and applies in straight and turning sections. The previous desired command is
zeroed on every episode reset. Fx has zero weight. The coefficient is a fixed
initial experimental setting, not a claimed optimal value. All existing reward
terms, physical limits and slew limits are retained.

Validation still selects best.zip using the same physical guards and common
objective, which excludes this added penalty. Every seed is retained, including
models for which selection.json prefers zero residual. The new screening script
evaluates all diagnostic best.zip candidates without selecting the best seed.

## Commands

From the workspace, in the same native Stonefish/GPU environment used for the
previous training (ROS master is not required):

```bash
bash scripts/train_hofa_residual_desired_smooth.sh \
  --output results/hofa_policy/residual_rl/desired_smooth_1m_v1 \
  --steps 1000000 --seeds 0,1,2 --workers 6 --device cuda
```

Append `--dry-run` to print the protocol without creating output or training.
Each learner has its own process, physics instances, replay buffer and log file.
Use a fresh output directory; existing experiments are never overwritten.

After all six runs complete:

```bash
bash scripts/evaluate_hofa_residual_desired_smooth.sh \
  --experiment results/hofa_policy/residual_rl/desired_smooth_1m_v1
```

This runs 7 methods (DAgger once and 6 SAC checkpoints) x 9 cases on the new
71000-series development split, and writes `desired_smooth_1m_v1_screening`.
Training validation uses 11000-series cases; 81000-series final cases are reserved
and are not executed by this script. Previously examined 31000/41000/61000 cases
are not presented as new holdouts.

## Reading the output

- `suite.json` / `summary.csv`: per-worker status and training budget.
- Per-run `validation.jsonl`: physical metrics, deterministic validation episodes,
  and desired/applied command diagnostics. `episodes.jsonl` in training environments
  describes exploratory training rollouts, not deterministic policy performance.
- Screening `group_summary.json`: method averages, then sample SD across training
  seeds; shared DAgger episodes are not independent seed replications.
- `paired_seed_deltas.csv`: B minus A for each matched training seed.
- `paired_case_deltas.csv`: B minus A by seed and common route instance.
- Desired/applied step RMS and slew-limited fractions are reported separately for
  turns (`abs(reference_yaw_rate)>=0.05`) and non-turn samples. These are adjacent
  control-step differences, not rates per second. A step crossing a turn boundary
  is assigned by its current sample. Missing sections remain null.

Assess whether non-turn Fy/Nz variations fall WITHOUT sacrificing success,
cross-track/yaw/turn-yaw RMSE or endpoint behavior. Better smoothness alone is not
a tracking success. Native screening is not a substitute for later ROS checks.
Do not compare the B training return directly with A to claim an improvement.

ROS ResidualActor supports both legacy 212D and new 215D checkpoints according to
the saved training config. Old ROS comparison scripts that lock a particular old
candidate still need a deliberate new evaluation protocol for these checkpoints.

## P0: fixed-endpoint checkpoint comparison

Evaluate all six `final.zip` models on exactly the existing 71000-series cases:

```bash
bash scripts/evaluate_hofa_residual_desired_smooth.sh \
  --experiment results/hofa_policy/residual_rl/desired_smooth_1m_v1 \
  --checkpoint final \
  --best-report results/hofa_policy/residual_rl/desired_smooth_1m_v1_screening
```

Append `--dry-run` for input/provenance checks without running physics.
The default output is `desired_smooth_1m_v1_final_checkpoint_screening`.
Here “final checkpoint” means the fixed training endpoint, NOT the 81000 final
holdout. This script does not execute that holdout or initiate training.

The script checks suite/case/config identity, source and physics provenance,
checkpoint hashes, raw episode metrics and actual archive timestep counts.
The nine validated DAgger episodes are reused once; six final checkpoints require
54 fresh deterministic episodes. Shared baseline records still refer to their
original run directories. Source best outputs are read-only.

- `summary`, `group_summary`, `paired_seed_deltas`, `paired_case_deltas`: final
  checkpoints only, with the shared DAgger reference.
- `best_final_summary`: six best and six final rows with checkpoint step/hash.
- `best_final_group_summary`: separate best/final group means and seed SD.
- `best_final_deltas`: final minus best for each of the six models.
- `best_final_ab_seed_deltas`: B minus A, separately for best and final.
- `shared_dagger_summary`: one baseline summary; no duplicate independent runs.

All failures and seed pairs remain in the tables. There is no per-seed best/final
selection, no new candidate selection, and no automatic P2 training launch.
P1 mechanism checks should next distinguish fixed-input deterministic behavior,
history-conditioned recurrence, and slew mapping. Preselect common intermediate
training steps before comparing their results. Any later continuation experiment
must address optimizer/replay provenance and old reward labels explicitly.
