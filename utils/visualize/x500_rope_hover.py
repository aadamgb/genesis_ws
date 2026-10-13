"""Visualize two pairs of x500 pursuers (utils/models/urdf/x500_rope.urdf) carrying a rod welded to their rope tips,
hovering in place, no policy: the first pair with the current rope (5 x 1.5 cm, as general_rope.urdf), the second
with the rope scaled by --long. The long rope is written next to x500_rope.urdf (relative mesh paths) and deleted
after the build. Each drone hovers with a geometric position / attitude controller and the x500 nominal kf, km;
with --m 1 (gentle) or 2 (aggressive) the two drones of a pair sway in opposite directions across the rod (MOVE) to
show the rope swinging; 2 saturates the rotors and the long rope pair can whip the rod and fall (failure cases).

    python utils/visualize/x500_rope_hover.py --long 4 --m 1 --t 60
"""
import os
import re
import math
import argparse
import tempfile

import torch

import genesis as gs
from genesis.utils.geom import quat_to_R

URDF = "utils/models/urdf/x500_rope.urdf"
X500 = dict(inertia=[0.03353, 0.03353, 0.06583], max_rpm=7448.451,  # [kg m^2], [rpm]
            props=[[0.1812, -0.1812], [-0.1812, 0.1812], [0.1812, 0.1812], [-0.1812, -0.1812]])  # [m] prop0-3 x, y
SPIN = (-1, -1, 1, 1)  # hydra_configs/drone/general_pursuer.yaml
TIP_LINK = "segment_5_cylinder"
TIP_INSET = 0.05  # [m] rope tips welded this far from the rod ends
Z = 2.5           # [m] hover height of the drones
PAIR_Y = 1.5      # [m] the pairs at -y (current rope) and +y (long rope)
DT = 0.008        # [s] as the pursuer task
SUBSTEPS = 4
MOVE = {1: (0.2, 4.0), 2: (0.5, 1.5)}  # --m: [m] y amplitude of each drone (opposite within a pair), [s] period
POS_WN, ATT_WN, ZETA = 2.0, 15.0, 1.0  # [rad/s], [rad/s], [-] natural frequencies and damping of the controller


def long_rope_urdf(scale):
    """x500_rope.urdf with every rope segment scaled by scale, in a temporary file next to it."""
    with open(URDF) as f:
        text = f.read()
    i = text.index('<link name="center_of_mass_link">')
    rope = re.sub(r'(xyz="0 0 |cylinder length=")(-?[0-9.]+)"', lambda m: f'{m[1]}{float(m[2]) * scale:g}"', text[i:])
    fd, path = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(URDF))
    with os.fdopen(fd, "w") as f:
        f.write(text[:i] + rope)
    return path


class Hover:
    """Geometric controller (Lee et al. 2010) to a reference position and velocity with zero yaw, rotor rpm out."""

    def __init__(self, drone, payload):
        self.drone = drone
        self.home = drone.get_pos().clone()
        self.mass = drone.get_mass() + payload
        self.J = torch.tensor(X500["inertia"], device=gs.device)
        # mixer: total thrust and body torques from the 4 rotor thrusts
        r = torch.tensor(X500["props"], device=gs.device)
        spin = torch.tensor(SPIN, device=gs.device, dtype=gs.tc_float)
        A = torch.stack([torch.ones(4, device=gs.device), r[:, 1], -r[:, 0], spin * drone.KM / drone.KF])
        self.A_inv = torch.linalg.inv(A)
        self.g = torch.tensor((0.0, 0.0, 9.81), device=gs.device)

    def rpm(self, ref, ref_vel, ref_acc):
        d = self.drone
        R = quat_to_R(d.get_quat())
        w = R.T @ d.get_ang()  # body rates
        acc = POS_WN**2 * (ref - d.get_pos()) + 2.0 * ZETA * POS_WN * (ref_vel - d.get_vel()) + ref_acc + self.g
        b3 = acc / acc.norm()
        b2 = torch.linalg.cross(b3, torch.tensor((1.0, 0.0, 0.0), device=gs.device))
        b2 = b2 / b2.norm()
        Rd = torch.stack([torch.linalg.cross(b2, b3), b2, b3], dim=1)
        E = Rd.T @ R - R.T @ Rd
        e_R = 0.5 * torch.stack([E[2, 1], E[0, 2], E[1, 0]])
        torque = self.J * (-ATT_WN**2 * e_R - 2.0 * ZETA * ATT_WN * w)
        thrust = self.mass * (acc @ R[:, 2])
        f = (self.A_inv @ torch.cat([thrust[None], torque])).clamp(min=0.0)
        return torch.sqrt(f / d.KF).clamp(max=X500["max_rpm"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", type=float, default=4.0, help="[-] rope scale of the second pair")
    parser.add_argument("--rod_length", type=float, default=2.0, help="[m]")
    parser.add_argument("--rod_mass", type=float, default=0.5, help="[kg]")
    parser.add_argument("--rod_radius", type=float, default=0.02, help="[m]")
    parser.add_argument("--m", type=int, default=0, choices=(0, 1, 2),
                        help="0 hover, 1 / 2: the drones of a pair sway in opposite directions, gently / aggressively")
    parser.add_argument("--t", type=float, default=60.0, help="[s]")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        viewer_options=gs.options.ViewerOptions(camera_pos=(4.0, -3.5, 3.2), camera_lookat=(0.0, 0.0, 2.2),
                                                camera_fov=45),
        rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton, enable_collision=True,
                                              enable_joint_limit=True),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    long_urdf = long_rope_urdf(args.long)
    half = args.rod_length / 2.0 - TIP_INSET
    pairs = []
    try:
        for y, urdf in ((-PAIR_Y, URDF), (PAIR_Y, long_urdf)):
            drones = [scene.add_entity(gs.morphs.Drone(file=urdf, pos=(x, y, Z), propellers_spin=SPIN))
                      for x in (-half, half)]
            # placed at the rope tips after the build
            rod = scene.add_entity(
                gs.morphs.Cylinder(radius=args.rod_radius, height=args.rod_length, pos=(0.0, y, Z),
                                   euler=(0.0, 90.0, 0.0), collision=False),
                surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1, 1.0)),
            )
            pairs.append((drones, rod))
        scene.build()
    finally:
        os.remove(long_urdf)

    controllers = []
    for (drones, rod), name in zip(pairs, ("current", f"x{args.long:g}")):
        tip = drones[0].get_link(TIP_LINK).get_pos()
        rod.set_pos(torch.tensor((0.0, rod.get_pos()[1], tip[2]), device=gs.device))
        rod.set_mass(args.rod_mass)
        for d in drones:
            scene.sim.rigid_solver.add_weld_constraint(d.get_link(TIP_LINK).idx, rod.base_link.idx)
            controllers.append(Hover(d, args.rod_mass / 2.0))
        length = (tip - drones[0].get_pos()).norm().item()  # the rope hangs straight at the build
        print(f"{name} rope: {length:.3f} m from base_link to the rod attachment")

    # --m 1 / 2: y offset A sin(w t) of each drone, +/- within a pair
    amp, period = MOVE.get(args.m, (0.0, 1.0))
    w = 2.0 * math.pi / period
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=gs.device) * amp
    ey = torch.tensor((0.0, 1.0, 0.0), device=gs.device)
    for step in range(int(args.t / DT)):
        t = step * DT
        for c, s in zip(controllers, sign):
            c.drone.set_propellers_rpm(c.rpm(c.home + s * math.sin(w * t) * ey, s * w * math.cos(w * t) * ey,
                                             -s * w**2 * math.sin(w * t) * ey))
        scene.step()


if __name__ == "__main__":
    main()
