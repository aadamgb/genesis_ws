"""Visualize a pursuit_evasion run (train_marl.py) with realistic models, no training: the pursuers are two x500
(utils/models/urdf/x500_rope.urdf) carrying the rod with the net mesh (utils/models/net.obj), the evader an a300
(utils/models/urdf/a300.urdf). The drones fly with the x500 / a300 nominal parameters (hydra_configs/drone/*.yaml on
des-inf-dom-rand) instead of randomized designs, the rod is ROD_LENGTH long.

Only for visualization: delete it together with x500_rope.urdf, a300.urdf and urdf/mesh/ when no longer needed.

    python visualize_pursuit_evasion.py --run logs/pursuit_evasion/SRT_general_check --c 99 --t 60
"""
import os
import pickle
import argparse
import torch

import genesis as gs

X500 = dict(urdf="utils/models/urdf/x500_rope.urdf", mass=2.0043, inertia=[0.03353, 0.03353, 0.06583], arm=0.2563,
            kf=4.605815e-7, max_rpm=7448.451, motor_tau=[0.0125, 0.025])
A300 = dict(urdf="utils/models/urdf/a300.urdf", mass=1.21, inertia=[0.007, 0.007, 0.01], arm=0.1507,
            kf=4.5e-8, max_rpm=21400.65, motor_tau=[0.0486, 0.0545])
ROD_LENGTH = 2.0  # [m] fixed: the net mesh is scaled to it at build


def pin(dr, drone):
    """The drone parameters of every env set to the nominals of drone (X500 / A300)."""
    for name in ("mass", "inertia", "arm", "kf", "max_rpm", "motor_tau"):
        getattr(dr, name)[:] = torch.tensor(drone[name], device=gs.device, dtype=gs.tc_float)
    dr._write(torch.arange(dr.num_envs, device=gs.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="logs/pursuit_evasion/SRT_general_check")
    parser.add_argument("--c", default="99", help="checkpoint iteration")
    parser.add_argument("--t", type=float, default=60.0, help="[s]")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    from src.env_pursuer import PursuerEnv
    from src.env_pursuit_evasion import PursuitEvasionEnv, AGENTS, load_cfgs
    from src.models import load_policy

    class RealisticEnv(PursuitEvasionEnv):
        def _add_entities(self):
            self.ev.add_to(self.scene, A300["urdf"], (0.0, 0.0, 1.0))
            depth = self.env_cfg["net_depth"]  # net.obj: a 2 x 2 m square in x, z
            self.net = self.scene.add_entity(
                gs.morphs.Mesh(file="utils/models/net.obj", scale=(ROD_LENGTH / 2.0, 1.0, depth / 2.0),
                               fixed=False, collision=False),
                surface=gs.surfaces.Default(color=(0.1, 0.7, 0.2, 1.0)),  # green
            )

        def _draw(self):
            """The rod (PursuerEnv) and the net mesh hanging below it, along the rod."""
            PursuerEnv._draw(self)
            a, _ = self._net_frame()
            yaw = torch.atan2(a[:, 1], a[:, 0])
            zeros = torch.zeros_like(yaw)
            self.net.set_pos(self.rod_pos - torch.tensor((0.0, 0.0, self.env_cfg["net_depth"] / 2.0), device=gs.device))
            self.net.set_quat(torch.stack([torch.cos(yaw / 2.0), zeros, zeros, torch.sin(yaw / 2.0)], dim=1))

    with open(os.path.join(args.run, "cfgs.pkl"), "rb") as f:
        _, env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)
    p_env_cfg = load_cfgs(env_cfg["pursuer_low_level"])[1]
    env_cfg["drone"] = {**p_env_cfg["drone"], "rope_urdf": X500["urdf"]}
    env_cfg["rod"] = {**p_env_cfg["rod"], "length": [ROD_LENGTH, ROD_LENGTH]}
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = False
    env_cfg["episode_length_s"] = args.t
    env = RealisticEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)

    for d in env.drones:
        pin(d.domain_rand, X500)
    pin(env.ev.domain_rand, A300)
    env._limit_twr(env.env_cfg["min_system_twr"])
    env.min_separation = env.env_cfg["termination_if_drones_closer_than"] * X500["arm"]
    obs = env.reset()  # hover rpm, motor lags and encoder parameters of the pinned drones

    model = os.path.join(args.run, f"model_{args.c}.pt")
    policies = {a: load_policy(model, obs, env.num_actions, agent=a) for a in AGENTS}
    with torch.no_grad():
        for _ in range(int(args.t / (env.dt * env.decimation))):
            obs, _, dones, extras = env.step(policies["pursuer"](obs), policies["evader"](obs))
            if dones.any():
                print(*(k for k, v in extras["outcomes"].items() if v))


if __name__ == "__main__":
    main()
