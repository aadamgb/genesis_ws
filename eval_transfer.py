"""Fly trained goto / adapt_goto policies on the real drones (hydra_configs/drone/<name>.yaml) in the goto env
and print the Gazebo-style metrics, to compare policies before flying them in Gazebo.

    python eval_transfer.py logs/goto/SRT_x500_stable:500 logs/adapt_goto/<run>:1000 --drones x500 a300

Each policy keeps its own controller, action scaling and observation settings; the drone, the task (targets,
threshold, terminations) and the randomization are the same for all. --mode nominal flies the nominal drone
with the nominal 18 ms delay, --mode dr samples each drone's own ranges and the 10-26 ms delay, --mode upset
starts every nominal drone at 4 m on the target at a uniformly random attitude with body rates up to 5 rad/s and
measures the recovery over --time seconds (3 s is enough).
"""
import argparse
import copy
import math
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat
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
    command = COMMAND
    if mode == "upset":
        env_cfg["base_init_pos"] = [0.0, 0.0, 4.0]
        command = {**COMMAND, "pos_x_range": [0.0, 0.0], "pos_y_range": [0.0, 0.0], "pos_z_range": [4.0, 4.0]}
    delay = {"nominal": 0.018, "range": [0.01, 0.026]}
    env_cfg["domain_rand"] = {"enabled": mode == "dr", "physical_resample": "startup", "action_delay": delay}
    if mode == "dr":
        env_cfg["domain_rand"]["hover_estimate_noise"] = 0.05
    reward_cfg = {"reward_scales": {"crash": -10.0}}
    env = GotoEnv(num_envs, env_cfg, obs_cfg, reward_cfg, command)
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
    def get_observations():
        p = params()  # the robust_goto critic may take the true params as params_true
        return TensorDict({**get_obs(), "params": p, "params_true": p}, batch_size=[n])

    env.get_observations = get_observations


def load(run, drone, mode, num_envs, time_s):
    run_dir, ckpt = run.rsplit(":", 1)
    env, train_cfg = make_env(run_dir, drone, mode, num_envs, time_s)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), None, device=gs.device)
    runner.load(f"{run_dir}/model_{ckpt}.pt")
    return env, runner.get_inference_policy(device=gs.device)


def tilt_deg(q):
    return torch.rad2deg(2.0 * torch.asin(torch.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2).clamp(max=1.0)))


@torch.no_grad()
def recovery(run, drone, num_envs, time_s):
    """Every env starts at 4 m on its target at a uniformly random attitude with body rates up to 5 rad/s.
    Recovered: no crash and back within 1 m of the target at the end; upright: first time below 30 deg tilt."""
    env, policy = load(run, drone, "upset", num_envs, time_s)
    obs = env.reset()
    n = num_envs
    quat = torch.randn((n, 4), device=gs.device)
    quat = quat / quat.norm(dim=1, keepdim=True)
    env.drone.set_quat(quat, zero_velocity=True)
    ang = 10.0 * torch.rand((n, 3), device=gs.device) - 5.0
    env.drone.set_dofs_velocity(torch.cat([torch.zeros_like(ang), ang], dim=1), dofs_idx_local=list(range(6)))
    env.base_quat[:] = quat
    env.base_ang_vel[:] = transform_by_quat(env.drone.get_ang(), inv_quat(quat))
    env.obs_hist_empty[:] = True  # the history starts at the upset, not at the upright reset
    env._update_observation()
    obs = env.get_observations()

    steps = math.ceil(time_s / env.dt)
    alive = torch.ones(n, dtype=torch.bool, device=gs.device)
    upright_at = torch.full((n,), float("nan"), device=gs.device)
    z_min = env.base_pos[:, 2].clone()
    for i in range(steps - 1):
        obs, _, dones, extras = env.step(policy(obs))
        alive &= ~dones.bool()
        z_min = torch.where(alive, torch.minimum(z_min, env.base_pos[:, 2]), z_min)
        first = alive & torch.isnan(upright_at) & (tilt_deg(env.base_quat) < 30.0)
        upright_at[first] = (i + 1) * env.dt
    ok = alive & (env.rel_pos.norm(dim=1) < 1.0)
    return {
        "recovered [%]": 100.0 * ok.float().mean().item(),
        "crashed [%]": 100.0 * (~alive).float().mean().item(),
        "upright after [s]": upright_at[ok].mean().item(),
        "altitude lost [m]": (4.0 - z_min[ok]).clamp(min=0.0).mean().item(),
    }


@torch.no_grad()
def evaluate(run, drone, mode, num_envs, time_s):
    if mode == "upset":
        return recovery(run, drone, num_envs, time_s)
    env, policy = load(run, drone, mode, num_envs, time_s)

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
        tilt = tilt_deg(env.base_quat)
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
    p.add_argument("--drones", nargs="+", default=["x500", "a300", "robofly"])
    p.add_argument("--mode", choices=["nominal", "dr", "upset"], default="nominal")
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
