# PPO baseline for the Dreamer vs PPO comparison

Branch `dreamer-cmp`, task `evader_cmp` (evader with one nominal 0.8 kg drone, fixed 18 ms delay, exact hover
estimate; see hydra_configs/task/evader_cmp.yaml), B 32768, 1000 iterations, checkpoints every 50 iterations,
`dreamer` conda env, RCI gpu partition (24 h limit). PPO settings, observation history 4, privileged critic and
params encoder unchanged from evader.yaml. Seeds 0, 1, 2; compared against r2dreamer runs (dreamer_ws,
configs/env/genesis_evader_proprio.yaml) with the same seeds.
