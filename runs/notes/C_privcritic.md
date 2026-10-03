# C_privcritic

- Task `adapt_goto` without encoder (general drone, design-informed randomization), controller `SRT` (a in [0, 1] = motor command, rl_goto `hover_center_throttle: false`), local RTX 4070 laptop, 2 runs in parallel.
- Shared by A-F: `at_target_threshold 0.2`, 1000 iterations, B 8192, PPO as adapt_goto (fixed lr 3e-4 unless stated, gamma 0.99, lam 0.95, 5 epochs, 4 mini-batches, entropy 0.004).
- Randomization changed on 2026-10-02 (general.yaml / utils/design_informed_dr.py) so it covers the a300 and x500:
  - motor tau independent of the size factor c: tau_up log-uniform 0.01-0.07 s, tau_down = tau_up x U[1.0, 2.2] (before: fixed at the x500 0.0125 / 0.025 +-10%; a300 is 0.0486 / 0.0545).
  - max_rpm min/max +15% (34500 / 9200) with +-20% noise: hover command 5-95% 0.33-0.55 (a300 0.38, x500 0.44).

## Changed in this run
- Asymmetric actor-critic: the critic also gets the privileged group (noise-free state, motor speeds, randomized drone parameters: obs.privileged true, obs_groups.critic [policy, critic]); the actor is unchanged (obs.history 4), so it deploys like A.

## Command
```bash
python train.py task=adapt_goto cm=SRT m=1000 task.env.at_target_threshold=0.2 +task.obs.history=4 +task.obs.privileged=true 'task.rsl_rl.obs_groups.critic=[policy,critic]' e=C_privcritic
```
