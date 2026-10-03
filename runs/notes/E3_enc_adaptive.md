# E3_enc_adaptive

- Task `adapt_goto_encoder` (general drone, design-informed randomization, drone params encoded into an 8-d tanh latent), controller `SRT` (a in [0, 1] = motor command, rl_goto `hover_center_throttle: false`).
- Shared by E1-E4: `obs.history 4`, `at_target_threshold 0.2`, 1500 iterations, RCI V100 (gpufast).
- Randomization changed on 2026-10-02 (general.yaml / utils/design_informed_dr.py) so it covers the a300 and x500:
  - motor tau independent of the size factor c: tau_up log-uniform 0.01-0.07 s, tau_down = tau_up x U[1.0, 2.2] (before: fixed at the x500 0.0125 / 0.025 +-10%; a300 is 0.0486 / 0.0545).
  - max_rpm min/max +15% (34500 / 9200) with +-20% noise: hover command 5-95% 0.33-0.55 (a300 0.38, x500 0.44).
- Gazebo: rl_goto `adapt_encoder: true` with the `drone:` block of the flown drone; export with `test.py export=true`.

## Changed in this run
- schedule adaptive (desired_kl 0.01, lr starts at 3e-4), else E1. Earlier general runs pinned the lr at the 1e-2 cap with adaptive; this checks it with 2x the envs and the new randomization.

## Command
```bash
NOTES=runs/notes/E3_enc_adaptive.md sbatch --job-name=E3_enc_adaptive runs/rci_train.sh task=adapt_goto_encoder cm=SRT m=1500 +task.obs.history=4 task.env.at_target_threshold=0.2 B=16384 task.rsl_rl.algorithm.schedule=adaptive
```
