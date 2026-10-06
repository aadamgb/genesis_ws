# R7_all

- Round 3 (2026-10-03), task `robust_goto` (src/env_robust_goto.py, copy of adapt_goto; hydra_configs/task/robust_goto.yaml) on the RCI `gpu` partition (V100; R5 and R6 on `gpufast`, the per-user limit is 6 GPUs per partition QoS), 2000 iterations.
- Baseline is SRT_general_M2_enc_gamma999: encoder (8-d latent, critic gets the params), SRT, gamma 0.999, fixed lr 3e-4, B 32768 / 8 mini-batches, actor and critic [256, 256, 128], obs history 4, tilt termination off (180 deg), at_target 0.2, rewards target 10 / smooth -0.005 / angular -0.008 / crash -10.
- Changed for every round-3 run vs M2:
  - drone `general_wide`: arm 0.06-0.4 m, mass 0.25-5 kg, thrust-to-weight log-uniform 2-20 independent of the size (hover command 0.22-0.71, max_rpm follows), inertia x log-uniform 0.7-2.5 on top of the l^5 law, motor tau up 0.01-0.07 s / down x1.0-2.2. Covers RoboFly (T/W 17.6, hover 0.24, inertia 2.2x the law), a300 (T/W 6.9, tau 0.049) and x500 (T/W 5.2).
  - action delay 10-40 ms (was 10-26).
  - start and target plane at z = 4 m (was 1 m).
- Why: M2 transfers to the heavy x500 / m690 but crashes on RoboFly (immediately) and the a300 (after minutes); RoboFly's hover command 0.24 was below everything the old range sampled (>= 0.30).

## Changed in this run
- Everything promising at once, the deployment candidate: B 65536 / 16 mini-batches (R1), obs history 8 (R2, rl_goto obs_history: 8), 30% upset starts (R4), mid-episode drone change in 50% of the episodes (R3), encoder parameter noise +-15% / +-5 ms with the critic on the true params (R6). entropy_coef 0.004 as the other runs (a first R7 with 0.001 was cancelled at iteration ~1000: its action std collapsed to 0.1 as in R0_ent001). Drone `general_robust`, the tinywhoop range of R05_whoop (the new robust_goto default).

## Command
```bash
NOTES=runs/notes/R7_all.md sbatch --partition=gpu --time=14:00:00 --job-name=R7_all runs/rci_train.sh task=robust_goto cm=SRT m=2000 B=65536 task.rsl_rl.algorithm.num_mini_batches=16 task.obs.history=8 task.env.upset.fraction=0.3 task.env.rerandomize.fraction=0.5 +task.obs.params_noise.physical=0.15 +task.obs.params_noise.delay=0.005 task.rsl_rl.obs_groups.critic=[policy,params_true]
```
