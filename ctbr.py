import genesis as gs
import torch
from src.controller import build_controller

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1, performance_mode=False)
dt = 0.01
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=5),
    show_viewer=True,
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=0.5,
        show_link_frame=True,
    )
)

scene.add_entity(gs.morphs.Plane())

drone = scene.add_entity(
    gs.morphs.Drone(
        file="misc/urdf/a300.urdf",
        propellers_spin=(-1, -1, 1, 1),
        pos=(0.0, 0.0, 1.0),
    )
)

env_cfg = {
        "num_actions": 4,
        "hover_rpm": 8120.65,
        "action_scale": 0.8,
        "ctbr_mixer": [             
            [ 1,   1,   1,   1],
            [-1,   1,   1,  -1],
            [-1,   1,  -1,   1],
            [-1,  -1,   1,   1],
        ],
        }

controller = build_controller("CTBR", drone, 1, dt=dt, cfg=env_cfg)
scene.build(n_envs=1)


action = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=gs.device)
for i in range(1000):
    drone.set_propellers_rpm(controller.update(action))
    scene.step()