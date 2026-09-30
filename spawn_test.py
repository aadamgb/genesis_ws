"""Spawn the a300 and the x500 and hold them at their hover rpm. After --rand_time seconds mass, inertia,
arm length, kf, km and the motor efficiencies are randomized and the new hover rpm is commanded.

    python spawn_test.py
    python spawn_test.py -s 1.05      # thrust 5% above hover: both climb
"""
import argparse
import math

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

from src.controller import drone_params

DRONES = {
    "a300": ("utils/models/urdf/a300.urdf", (0.0, -0.6, 1.0)),
    "x500": ("utils/models/urdf/x500.urdf", (0.0, 0.6, 1.0)),
}

# multipliers of the nominal, inertia sampled per axis and motor_eff per motor
RAND_SCALE = {
    "mass": (0.8, 1.2),
    "inertia": (0.8, 1.2),
    "arm": (0.95, 1.05),
    "kf": (0.85, 1.15),
    "km": (0.85, 1.15),
    "motor_eff": (0.95, 1.0),
}
SIZE = {"inertia": 3, "motor_eff": 4}

PROPS_IDX = [1, 2, 3, 4]  # prop0..3_link


def randomize(drone, rng):
    """Write the randomized parameters into Genesis and return the motor efficiencies."""
    s = {k: rng.uniform(lo, hi, size=SIZE.get(k)) for k, (lo, hi) in RAND_SCALE.items()}

    drone.set_links_mass([drone.get_links_mass([0])[0].item() * s["mass"]], [0])

    # Genesis stores the inertia in the link's principal-axes frame, R rotates it to the body frame
    R = torch.tensor(gu.quat_to_R(np.asarray(drone.base_link.desc.inertial_quat)), device=gs.device, dtype=gs.tc_float)
    J = (R @ drone.get_links_inertia([0])[0] @ R.T).diagonal() * torch.tensor(s["inertia"], device=gs.device, dtype=gs.tc_float)
    drone.set_links_inertia((R.T @ torch.diag(J) @ R)[None], [0])

    com = drone.get_links_COM(PROPS_IDX).clone()
    com[:, :2] *= s["arm"]
    drone.set_links_COM(com, PROPS_IDX)

    # read by set_propellers_rpm at every call
    drone._desc.kf *= s["kf"]
    drone._desc.km *= s["km"]

    return s["motor_eff"]


def drone_table(drone, urdf, motor_eff):
    p = drone_params(urdf)
    mass, kf, km, max_rpm = drone.get_mass().item(), drone.KF, drone.KM, p["max_rpm"]
    return {
        "mass [kg]": mass,
        # rotational block of the free-joint mass matrix: body-frame inertia
        "J [Jxx, Jyy, Jzz] [kg m^2]": drone.get_mass_mat()[3:, 3:].diagonal().tolist(),
        "arm length [m]": drone.get_links_COM(PROPS_IDX)[:, :2].norm(dim=1).mean().item(),
        "kf [N/rpm^2]": kf,
        "km [Nm/rpm^2]": km,
        "thrust / weight [-]": 4 * kf * max_rpm**2 / (mass * 9.81),
        "hover rpm [rpm]": math.sqrt(9.81 * mass / 4.0 / kf),
        "max rpm [rpm]": max_rpm,
        "max thrust per rotor [N]": kf * max_rpm**2,
        "motor tau up [s]": p["tau_up"],
        "motor tau down [s]": p["tau_down"],
        # virtual: the rpm each motor reaches is its command times its efficiency
        "motor efficiency [-]": list(motor_eff),
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


def hover_command(title, drones, motor_eff, thrust_scale):
    """Print the parameters and return the rpm of each drone for thrust_scale x its hover thrust."""
    tables = {n: drone_table(d, DRONES[n][0], motor_eff[n]) for n, d in drones.items()}
    print_table(title, tables)
    rpm = {n: t["hover rpm [rpm]"] * math.sqrt(thrust_scale) for n, t in tables.items()}
    print(f"commanding {thrust_scale:.3f} x hover thrust: " + ", ".join(f"{n} {r:.1f} rpm" for n, r in rpm.items()))
    return rpm


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=float, default=30.0, help="duration [s]")
parser.add_argument("-r", "--rand_time", type=float, default=10.0, help="time of the randomization [s]")
parser.add_argument("-s", "--thrust_scale", type=float, default=1.0, help="thrust relative to hover")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

dt = 0.008
gs.init(backend=gs.gpu, logging_level="warning", precision="64")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=2),
    viewer_options=gs.options.ViewerOptions(camera_pos=(3.0, 0.0, 2.0), camera_lookat=(0.0, 0.0, 1.0), camera_fov=40),
    vis_options=gs.options.VisOptions(show_world_frame=True, world_frame_size=0.5),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())
# align=False: Genesis refuses to write the inertial properties of an aligned body at runtime
drones = {
    name: scene.add_entity(gs.morphs.Drone(file=urdf, pos=pos, propellers_spin=(-1, -1, 1, 1), align=False))
    for name, (urdf, pos) in DRONES.items()
}
scene.build()

motor_eff = {name: np.ones(4) for name in drones}
rpm = hover_command("nominal", drones, motor_eff, args.thrust_scale)
rng = np.random.default_rng(args.seed)
for step in range(int(args.time / dt)):
    if step == int(args.rand_time / dt):
        motor_eff = {name: randomize(drone, rng) for name, drone in drones.items()}
        rpm = hover_command(f"randomized at t = {step * dt:.1f} s", drones, motor_eff, args.thrust_scale)
    for name, drone in drones.items():
        drone.set_propellers_rpm(rpm[name] * motor_eff[name])
    scene.step()
