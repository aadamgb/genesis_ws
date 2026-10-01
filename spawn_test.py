"""Spawn the a300 and the x500 and hold them at their hover rpm. After --rand_time seconds
their parameters are randomized with the ranges of hydra_configs/drone/<name>.yaml (utils/domain_rand.py)
and the new hover rpm is commanded.

    python spawn_test.py
    python spawn_test.py -s 1.05      # thrust 5% above hover: both climb
"""
import argparse
import math

import torch

import genesis as gs

from utils.domain_rand import DomainRand, load_drone_cfg

DRONES = {
    "a300": (0.0, -1.0, 1.0),
    "x500": (0.0, 0.0, 1.0),
}

PROPS_IDX = [1, 2, 3, 4]  # prop0..3_link


def drone_table(drone, dr, drone_cfg):
    mass, kf, max_rpm = drone.get_mass().item(), dr.kf.item(), drone_cfg["max_rpm"]
    return {
        "mass [kg]": mass,
        # rotational block of the free-joint mass matrix: body-frame inertia
        "J [Jxx, Jyy, Jzz] [kg m^2]": drone.get_mass_mat().reshape(6, 6)[3:, 3:].diagonal().tolist(),
        "arm length [m]": drone.get_links_COM(PROPS_IDX).reshape(4, 3)[:, :2].norm(dim=1).mean().item(),
        "kf [N/rpm^2]": kf,
        "km [Nm/rpm^2]": drone.KM * kf / dr.nominal("kf").item(),
        "thrust / weight [-]": 4 * kf * max_rpm**2 / (mass * 9.81),
        "hover rpm [rpm]": math.sqrt(9.81 * mass / 4.0 / kf),
        "max rpm [rpm]": max_rpm,
        "max thrust per rotor [N]": kf * max_rpm**2,
        "motor tau up [s]": dr.motor_tau[0, 0].item(),
        "motor tau down [s]": dr.motor_tau[0, 1].item(),
        "motor efficiency [-]": dr.motor_eff[0].tolist(),
    }


def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return f"{v:.4e}" if abs(v) < 1e-3 else f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"


def print_table(title, tables):
    rows = list(next(iter(tables.values())))
    cols = {n: [fmt(t[r]) for r in rows] for n, t in tables.items()}
    w0 = max(len(r) for r in rows)
    wc = {n: max(len(n), *(len(v) for v in c)) for n, c in cols.items()}
    line = "+" + "-" * (w0 + 2) + "+" + "+".join("-" * (w + 2) for w in wc.values()) + "+"
    print(f"\n{title}\n{line}")
    print(f"| {'property':<{w0}} | " + " | ".join(f"{n:>{wc[n]}}" for n in cols) + " |")
    print(line.replace("-", "="))
    for i, label in enumerate(rows):
        print(f"| {label:<{w0}} | " + " | ".join(f"{cols[n][i]:>{wc[n]}}" for n in cols) + " |")
    print(line)


def hover_command(title, drones, thrust_scale):
    """Print the parameters and return the rpm command of each drone for thrust_scale x its hover thrust."""
    tables = {n: drone_table(d, dr, cfg) for n, (d, dr, cfg) in drones.items()}
    print_table(title, tables)
    rpm = {n: t["hover rpm [rpm]"] * math.sqrt(thrust_scale) for n, t in tables.items()}
    print(f"commanding {thrust_scale:.3f} x hover thrust: " + ", ".join(f"{n} {r:.1f} rpm" for n, r in rpm.items()))
    # kf and the motor efficiencies act through the rpm
    return {n: rpm[n] * drones[n][1].rpm_scale for n in drones}


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=float, default=30.0, help="duration [s]")
parser.add_argument("-r", "--rand_time", type=float, default=10.0, help="time of the randomization [s]")
parser.add_argument("-s", "--thrust_scale", type=float, default=1.0, help="thrust relative to hover")
args = parser.parse_args()

dt = 0.008
gs.init(backend=gs.gpu, logging_level="warning", precision="64")

drones = {}
for name in DRONES:
    cfg = load_drone_cfg(name)
    drones[name] = (None, DomainRand(cfg, {"enabled": True, "physical_resample": "reset"}, num_envs=1), cfg)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=2),
    viewer_options=gs.options.ViewerOptions(camera_pos=(3.0, 0.0, 2.0), camera_lookat=(0.0, 0.0, 1.0), camera_fov=40),
    vis_options=gs.options.VisOptions(show_world_frame=True, world_frame_size=0.5),
    rigid_options=gs.options.RigidOptions(batch_links_info=any(dr.batch_links_info for _, dr, _ in drones.values())),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())
for name, (_, dr, cfg) in drones.items():
    # align=False: Genesis refuses to write the inertial properties of an aligned body at runtime
    morph = gs.morphs.Drone(file=cfg["urdf"], pos=DRONES[name], propellers_spin=tuple(cfg["propellers_spin"]), align=False)
    drones[name] = (scene.add_entity(morph), dr, cfg)
scene.build(n_envs=1)

for drone, dr, _ in drones.values():
    dr.build(drone)
    dr.set_nominal()
rpm = hover_command("nominal", drones, args.thrust_scale)

envs_idx = torch.arange(1, device=gs.device)
for step in range(int(args.time / dt)):
    if step == int(args.rand_time / dt):
        for _, dr, _ in drones.values():
            dr.reset_idx(envs_idx)
        rpm = hover_command(f"randomized at t = {step * dt:.1f} s", drones, args.thrust_scale)
    for name, (drone, _, _) in drones.items():
        drone.set_propellers_rpm(rpm[name])
    scene.step()
