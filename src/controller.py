import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
)

from utils.domain_rand import load_drone_cfg

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity

def drone_params(urdf: str, drone_cfg: dict | None = None) -> dict:
    """Nominal mass, kf, km, max_rpm, hover_rpm and thrust2weight of a drone, from its config (given, or
    hydra_configs/drone/<urdf name>.yaml). Urdfs without a config (bros300, bambi) give mass, kf, km and
    thrust2weight in their <properties>."""
    name = os.path.splitext(os.path.basename(urdf))[0]
    if drone_cfg is None and os.path.exists(f"hydra_configs/drone/{name}.yaml"):
        drone_cfg = load_drone_cfg(name)
    if drone_cfg is not None:
        p = {k: drone_cfg[k]["nominal"] for k in ("mass", "kf", "km")}
        p["max_rpm"] = drone_cfg["max_rpm"]
    else:
        root = ET.parse(urdf).getroot()
        p = {k: float(v) for k, v in root.find("properties").attrib.items()}
        p.setdefault("mass", sum(float(m.get("value")) for m in root.iter("mass")))

    p["hover_rpm"] = np.sqrt(9.81 * p["mass"] / 4.0 / p["kf"])
    if "max_rpm" in p:
        p["thrust2weight"] = (p["max_rpm"] / p["hover_rpm"]) ** 2
    else:
        p["max_rpm"] = p["hover_rpm"] * np.sqrt(p["thrust2weight"])
    return p


class BaseController(ABC):

    def __init__(self, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict):
        self.drone = drone
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = cfg

        params = drone_params(drone.morph.file, cfg.get("drone"))
        self.KF = params["kf"]
        self.KM = params["km"]
        self.mass = params["mass"]
        self.hover_rpm = params["hover_rpm"]
        self.max_rpm = params["max_rpm"]
        self.min_rpm    = 3200.0 # TODO: Hard coded for now.... 
        self.hover_cmd  = (self.hover_rpm - self.min_rpm) / (self.max_rpm - self.min_rpm)

    @abstractmethod
    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (num_envs, num_actions) -> rpms: (num_envs, 4)"""

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Override to for env reset"""

class SRT(BaseController):

    def __init__(self, drone, num_envs, dt, cfg):
        super().__init__(drone, num_envs, dt, cfg)

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.max_rpm


class SRTHover(BaseController):

    def __init__(self, drone, num_envs, dt, cfg):
        super().__init__(drone, num_envs, dt, cfg)
        self.action_scale = cfg.get("action_scale", 1.0)
        self.hover_u = self.hover_rpm / self.max_rpm  # sqrt(1 / thrust2weight)

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(self.hover_u * (1.0 + self.action_scale * actions), 0.0, 1.0)
        return u * self.max_rpm



CONTROLLERS = {
    "SRT": SRT,
    "SRTHover": SRTHover,
}


def build_controller(name: str, drone: "DroneEntity", num_envs: int, dt: float, cfg: dict) -> BaseController:
    try:
        controller_cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"Unknown controller_type '{name}'. Available: {list(CONTROLLERS)}") from None
    return controller_cls(drone, num_envs, dt, cfg)
