import os
import pickle
import shutil
import hydra
from omegaconf import OmegaConf, DictConfig
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


@hydra.main(version_base=None, config_path="hydra_configs", config_name="train")
def main(cfg: DictConfig):
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg.seed, performance_mode=True)

    task = cfg.task
    env_cfg     = OmegaConf.to_container(task.env, resolve=True)
    obs_cfg     = OmegaConf.to_container(task.obs, resolve=True)
    reward_cfg  = OmegaConf.to_container(task.reward, resolve=True)
    command_cfg = OmegaConf.to_container(task.command, resolve=True)
    train_cfg   = OmegaConf.to_container(task.rsl_rl, resolve=True)

    log_dir = cfg.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([task.name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    if task.name == "evader":
        from src.env_evader import EvaderEnv
        env = EvaderEnv(num_envs=cfg.B, env_cfg=env_cfg, obs_cfg=obs_cfg,
                        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=cfg.v)
    elif task.name == "pursuer":
        from src.env_pursuer import PursuerEnv
        env = PursuerEnv(num_envs=cfg.B, env_cfg=env_cfg, obs_cfg=obs_cfg,
                         reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=cfg.v)
    else:
        raise ValueError(f"unknown env: {task.name}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=cfg.m, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

