"""Fly trained goto / adapt_goto policies on the real drones (hydra_configs/drone/<name>.yaml) in the goto env
and print the Gazebo-style metrics, to compare policies before flying them in Gazebo.

    python eval_transfer.py logs/goto/SRT_x500_stable:500 logs/adapt_goto/<run>:1000 --drones x500 a300

Each policy keeps its own controller, action scaling and observation settings; the drone, the task (targets,
threshold, terminations) and the randomization are the same for all. --mode nominal flies the nominal drone
with the nominal 18 ms delay, --mode dr samples each drone's own ranges and the 10-26 ms delay.
"""
import argparse
import copy
import math
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from utils.domain_rand import load_drone_cfg

TASK = {  # crashes: ground or out of the box, tilts past 60 deg count only if the drone does not recover
    "at_target_threshold": 0.2,
    "termination_if_roll_greater_than": 180,
    "termination_if_pitch_greater_than": 180,
    "termination_if_x_greater_than": 3.0,
    "termination_if_y_greater_than": 3.0,
    "termination_if_z_greater_than": 3.0,
    "visualize_target": False,
    "visualize_camera": False,
}
COMMAND = {"num_commands": 3, "pos_x_range": [-1.0, 1.0], "pos_y_range": [-1.0, 1.0], "pos_z_range": [1.0, 1.0]}


def make_env(run_dir, drone, mode, num_envs, time_s):
    from src.env_goto import GotoEnv

    task, env_cfg, obs_cfg, reward_cfg, _, train_cfg = pickle.load(open(f"{run_dir}/cfgs.pkl", "rb"))
    env_cfg = copy.deepcopy(env_cfg)
    trained_drone = env_cfg.get("drone")
    env_cfg.update(TASK, episode_length_s=time_s, drone=load_drone_cfg(drone))
    delay = {"nominal": 0.018, "range": [0.01, 0.026]}
    env_cfg["domain_rand"] = {"enabled": mode == "dr", "physical_resample": "startup", "action_delay": delay}
    if mode == "dr":
        env_cfg["domain_rand"]["hover_estimate_noise"] = 0.05
    reward_cfg = {"reward_scales": {"crash": -10.0}}
    env = GotoEnv(num_envs, env_cfg, obs_cfg, reward_cfg, COMMAND)
    if obs_cfg.get("params"):
        add_params(env, trained_drone)
    return env, train_cfg


def add_params(env, design_cfg):
    """Encoder policies: the "params" group of the flown drone, normalized as env_adapt_goto.py _update_params
    with the middle design of the general drone the policy was trained on (as rl_goto.cpp loadDroneParams)."""
    from tensordict import TensorDict
    from utils.design_informed_dr import DesignInformedDR

    ref = DesignInformedDR(design_cfg, {}, 1)
    dr, n = env.domain_rand, env.num_envs
    log_ratio = lambda name, x: torch.log(x / ref.nominal(name))
    max_rpm = torch.full((n, 1), env.drone_cfg["max_rpm"], device=gs.device)
    params = lambda: torch.cat(
        [
            log_ratio("mass", dr.mass[:, None]),
            log_ratio("inertia", dr.inertia),
            log_ratio("arm", dr.arm[:, None].expand(n, 4)),
            log_ratio("kf", dr.kf),
            log_ratio("max_rpm", max_rpm),
            log_ratio("motor_tau", dr.motor_tau),
            (dr.motor_eff - 1.0) * 10.0,
            dr.action_delay[:, None] / 0.02,
        ],
        dim=-1,
    )
    get_obs = env.get_observations
    env.get_observations = lambda: TensorDict({**get_obs(), "params": params()}, batch_size=[n])


@torch.no_grad()
def evaluate(run, drone, mode, num_envs, time_s):
    run_dir, ckpt = run.rsplit(":", 1)
    env, train_cfg = make_env(run_dir, drone, mode, num_envs, time_s)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), None, device=gs.device)
    runner.load(f"{run_dir}/model_{ckpt}.pt")
    policy = runner.get_inference_policy(device=gs.device)

    max_rpm = env.drone_cfg["max_rpm"]
    steps = math.ceil(time_s / env.dt)
    goals = crashes = 0
    tilt_sum = tilt_max = rate_sq = du_sq = 0.0
    obs = env.reset()
    last_u = env.motor_rpm / max_rpm
    for _ in range(steps - 1):  # the last step would time every env out
        cmd = env.commands.clone()
        obs, _, dones, extras = env.step(policy(obs))
        done = dones.bool()
        crash = done & (extras["time_outs"] == 0)
        goals += ((env.commands != cmd).any(dim=1) & ~done).sum().item()
        crashes += crash.sum().item()
        q = env.base_quat
        tilt = torch.rad2deg(2.0 * torch.asin(torch.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2).clamp(max=1.0)))
        tilt_sum += tilt.sum().item()
        tilt_max = max(tilt_max, torch.quantile(tilt, 0.99).item())
        rate_sq += env.base_ang_vel.square().sum(dim=1).sum().item()
        u = env.motor_rpm / max_rpm
        du_sq += ((u - last_u)[~done]).square().mean(dim=1).sum().item()
        last_u = u
    n = num_envs * (steps - 1)
    minutes = n * env.dt / 60.0
    return {
        "goals/min": goals / minutes,
        "crashes/min": crashes / minutes,
        "tilt mean [deg]": tilt_sum / n,
        "tilt p99 [deg]": tilt_max,
        "rates rms [rad/s]": math.sqrt(rate_sq / n),
        "du rms [-]": math.sqrt(du_sq / n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="log_dir:checkpoint")
    p.add_argument("--drones", nargs="+", default=["x500", "a300"])
    p.add_argument("--mode", choices=["nominal", "dr"], default="nominal")
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--time", type=float, default=30.0, help="[s] per env")
    a = p.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0, performance_mode=True)
    rows = []
    for run in a.runs:
        for drone in a.drones:
            m = evaluate(run, drone, a.mode, a.envs, a.time)
            rows.append((run, drone, m))
            print(f"{run} {drone}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)

    keys = list(rows[0][2])
    w = max(len(r[0]) for r in rows)
    print(f"\n{'run':{w}s} {'drone':6s} " + " ".join(f"{k:>18s}" for k in keys))
    for run, drone, m in rows:
        print(f"{run:{w}s} {drone:6s} " + " ".join(f"{m[k]:18.3f}" for k in keys))


if __name__ == "__main__":
    main()
