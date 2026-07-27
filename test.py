import os
import pickle
import torch
import hydra
from omegaconf import DictConfig
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

@hydra.main(version_base=None, config_path="hydra_configs", config_name="test")
def main(cfg: DictConfig):
    gs.init()

    log_dir = cfg.log_dir
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        loaded = pickle.load(f)

    # print(loaded)

    if len(loaded) == 6:
        task_name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = loaded
    else:
        # TODO: DELETE this later when all pkls are updated!
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = loaded
        print("Carefulll task name not saved in pck, task is set to hover by defaulet!!")
        task_name = cfg.task.name

    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = cfg.record
    env_cfg["episode_length_s"] = cfg.t

    if task_name == "hover":
        from src.env_hover import HoverEnv
        env = HoverEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True) 
           
    elif task_name == "racing":
        from src.env_racing import RaceEnv
        env = RaceEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True)
                      
    elif task_name == "sprind":
        from src.env_sprind import SprindEnv
        env = SprindEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True)
    else:
        raise ValueError(f"unknown env: {task_name}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{cfg.c}.pt"))
    if cfg.export:
        runner.export_policy_to_jit(log_dir, f"model_{cfg.c}_scripted.pt")
        return

    policy = runner.get_inference_policy(device=gs.device)
    obs_dict = env.reset()

    # ====================================================
    # ------------------ Sim Loop ------------------------
    # ====================================================
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if cfg.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()