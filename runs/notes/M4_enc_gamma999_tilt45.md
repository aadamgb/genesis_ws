# M4_enc_gamma999_tilt45

- Task `adapt_goto_encoder` (rl_goto `adapt_encoder: true` with the flown drone's drone: block), PPO fixed lr 3e-4, gamma 0.99 unless stated.
- Round 2 (2026-10-02 evening), RCI gpufast V100, ~3 h: 3500 iterations, B 32768 envs with 8 mini-batches (E4), actor and critic hidden [256, 256, 128] (D_net256), controller `SRT`, general drone with the round-1 randomization (motor tau independent of c, max_rpm +15% / +-20%), `at_target_threshold 0.2`.
- Tilt termination off: termination_if_roll / pitch 180 deg (was 60), so the policy keeps flying past 60 deg and learns to recover; episodes still end on the ground (0.1 m) and outside the 3 m box.
- Round 1 best: no encoder D_net256 (x500 71.5 goals/min 0 crashes/min, a300 75.7 / 0.16); encoder E2_enc_gamma999 (safest, 0.006 / 0.003 crashes/min, 50-55 goals/min) and E4_enc_B32k (fastest, 75 / 86 goals/min, 0.06 / 0.38 crashes/min). eval_transfer.py, nominal drones, 18 ms delay.

## Changed in this run
- gamma 0.999 and initial tilt up to 45 deg (was 15) to practice recoveries, else M1.

## Command
```bash
NOTES=runs/notes/M4_enc_gamma999_tilt45.md sbatch --job-name=M4_enc_gamma999_tilt45 runs/rci_train.sh task=adapt_goto_encoder cm=SRT m=3500 task.env.at_target_threshold=0.2 B=32768 task.rsl_rl.algorithm.num_mini_batches=8 task.env.termination_if_roll_greater_than=180 task.env.termination_if_pitch_greater_than=180 task.rsl_rl.actor.hidden_dims=[256,256,128] task.rsl_rl.critic.hidden_dims=[256,256,128] +task.obs.history=4 task.rsl_rl.algorithm.gamma=0.999 task.env.init_tilt=45.0
```
