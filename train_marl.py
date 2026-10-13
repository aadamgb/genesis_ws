"""Multi-agent training (task=pursuit_evasion): one PPO per agent on a shared rollout, each with its own actor
observations, rewards and critic (the critics see the joint state). Saves {agent: PPO state} as model_{it}.pt.

    python train_marl.py task=pursuit_evasion e=<name>
"""
import os
import copy
import time
import pickle
import shutil
from types import SimpleNamespace
from collections import defaultdict

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms import PPO

import genesis as gs


@hydra.main(version_base=None, config_path="hydra_configs", config_name="train")
def main(cfg: DictConfig):
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg.seed, performance_mode=True)

    task = cfg.task
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = (
        OmegaConf.to_container(task[k], resolve=True) for k in ("env", "obs", "reward", "command", "rsl_rl")
    )
    log_dir = cfg.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([task.name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    from src.env_pursuit_evasion import PursuitEvasionEnv, AGENTS
    env = PursuitEvasionEnv(cfg.B, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=cfg.v)

    # one PPO per agent; construct_algorithm reads num_envs / num_actions from the env
    obs = env.get_observations()
    view = SimpleNamespace(num_envs=env.num_envs, num_actions=env.num_actions)
    algs = {}
    for agent in AGENTS:
        agent_cfg = {**copy.deepcopy(train_cfg), "obs_groups": copy.deepcopy(train_cfg["obs_groups"][agent]), "multi_gpu": None}
        algs[agent] = PPO.construct_algorithm(obs, view, agent_cfg, gs.device)
        algs[agent].train_mode()

    writer = SummaryWriter(log_dir)
    save = lambda it: torch.save({**{a: alg.save() for a, alg in algs.items()}, "iter": it}, f"{log_dir}/model_{it}.pt")
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    for it in range(cfg.m):
        start = time.time()
        episode, outcomes, n_resets = defaultdict(list), defaultdict(float), 0
        with torch.inference_mode():
            for _ in range(train_cfg["num_steps_per_env"]):
                actions = {a: alg.act(obs) for a, alg in algs.items()}
                obs, rewards, dones, extras = env.step(actions["pursuer"], actions["evader"])
                for a, alg in algs.items():
                    alg.process_env_step(obs, rewards[a], dones, extras)
                for k, v in extras["outcomes"].items():
                    outcomes[k] += v
                if "episode" in extras and extras["episode"]:
                    for k, v in extras["episode"].items():
                        episode[k].append(v)
                    extras["episode"] = {}
            for alg in algs.values():
                alg.compute_returns(obs)
        collect = time.time() - start
        losses = {a: alg.update() for a, alg in algs.items()}

        # log: episode reward terms [per s], outcome shares of the finished games, losses, action std
        n_games = max(sum(outcomes.values()), 1.0)
        scalars = {f"episode/{k}": sum(v) / len(v) for k, v in episode.items()}
        scalars.update({f"outcome/{k}": v / n_games for k, v in outcomes.items()})
        for a, alg in algs.items():
            scalars.update({f"{a}/{k}": v for k, v in losses[a].items()})
            scalars[f"{a}/action_std"] = alg.get_policy().output_std.mean().item()
        scalars["perf/iteration_time"] = time.time() - start
        scalars["perf/collection_time"] = collect
        for k, v in scalars.items():
            writer.add_scalar(k, v, it)
        print(f"it {it}/{cfg.m}  {scalars['perf/iteration_time']:.2f} s  games {int(n_games)}  "
              + "  ".join(f"{k} {outcomes[k] / n_games:.2f}" for k in outcomes))
        if it % train_cfg["save_interval"] == 0:
            save(it)
    save(cfg.m - 1)
    writer.close()


if __name__ == "__main__":
    main()
