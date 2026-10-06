# R3_rerand

- Round 3 (2026-10-03), task `robust_goto` (src/env_robust_goto.py, copy of adapt_goto; hydra_configs/task/robust_goto.yaml) on the RCI `gpu` partition (V100; R5 and R6 on `gpufast`, the per-user limit is 6 GPUs per partition QoS), 2000 iterations.
- Baseline is SRT_general_M2_enc_gamma999: encoder (8-d latent, critic gets the params), SRT, gamma 0.999, fixed lr 3e-4, B 32768 / 8 mini-batches, actor and critic [256, 256, 128], obs history 4, tilt termination off (180 deg), at_target 0.2, rewards target 10 / smooth -0.005 / angular -0.008 / crash -10.
- Changed for every round-3 run vs M2:
  - drone `general_wide`: arm 0.06-0.4 m, mass 0.25-5 kg, thrust-to-weight log-uniform 2-20 independent of the size (hover command 0.22-0.71, max_rpm follows), inertia x log-uniform 0.7-2.5 on top of the l^5 law, motor tau up 0.01-0.07 s / down x1.0-2.2. Covers RoboFly (T/W 17.6, hover 0.24, inertia 2.2x the law), a300 (T/W 6.9, tau 0.049) and x500 (T/W 5.2).
  - action delay 10-40 ms (was 10-26).
  - start and target plane at z = 4 m (was 1 m).
- Why: M2 transfers to the heavy x500 / m690 but crashes on RoboFly (immediately) and the a300 (after minutes); RoboFly's hover command 0.24 was below everything the old range sampled (>= 0.30).
- Gazebo: rl_goto `adapt_encoder: true`, `hover_center_throttle: false`, `obs_history` as below, and the drone: block of the flown drone.

## Changed in this run
- Mid-episode drone change: 50% of the episodes get a completely new drone (size, mass, inertia, arms, kf, T/W, motor tau, efficiencies, delay) at a uniformly random step, and the encoder its new parameters right away (env.rerandomize.fraction 0.5).

## Command
```bash
NOTES=runs/notes/R3_rerand.md sbatch --partition=gpu --time=8:00:00 --job-name=R3_rerand runs/rci_train.sh task=robust_goto cm=SRT m=2000 B=32768 task.env.rerandomize.fraction=0.5
```
