"""Domain randomization: per-env physical and actuator parameters.

Config (env_cfg["domain_rand"]), every entry optional:

    enabled: true             # false uses the nominals everywhere
    physical_resample: startup  # when mass, inertia and arm are sampled: startup (once per env) or reset
    mass:         {nominal, range | scale}   [kg] base link
    inertia:      {nominal, range | scale}   [kg m^2] ixx, iyy, izz in the body frame
    arm:          {nominal, range | scale}   [m] rotor distance from the COM, scales the rotor xy positions
    thrust_scale: {nominal, range | scale}   multiplies thrust and yaw moment
    motor_tau:    {up, down, range | scale}  [s] first-order rpm lag, one sample shared by up and down
    action_delay: {nominal, range | scale}   [s]

range: [lo, hi] samples uniformly in physical units, scale: [lo, hi] samples a multiplier of the nominal
(lo and hi may be lists for vector parameters). Without either the nominal is used. The nominals of mass,
inertia and arm default to the urdf.

Mass, inertia and arm are written into Genesis, per env when randomized, which needs the scene built with
RigidOptions(batch_links_info=DomainRand.batch_links_info) and the drone loaded with align=False. Every
write costs ~0.5 ms whatever the number of envs, and with thousands of envs some reset at almost every
step, so physical_resample: reset adds ~1.5-2.5 ms per step (8192 envs, RTX 4070 laptop); startup is free.
The other parameters are resampled at every reset into preallocated (num_envs, ...) buffers.
"""
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

PHYSICAL = ("mass", "inertia", "arm")


def from_legacy_env_cfg(env_cfg):
    """domain_rand config for env configs saved before it existed (flat keys in env_cfg)."""
    if "domain_rand" in env_cfg:
        return env_cfg["domain_rand"]
    return {
        "enabled": True,
        "thrust_scale": {"nominal": 1.0, "range": env_cfg.get("thrust_scale_range", [1.0, 1.0])},
        "motor_tau": {
            "up": env_cfg.get("motor_tau_up", 0.0),
            "down": env_cfg.get("motor_tau_down", 0.0),
            "scale": env_cfg.get("motor_tau_scale_range", [1.0, 1.0]),
        },
        "action_delay": {"nominal": 0.0, "range": env_cfg.get("action_delay_range", [0.0, 0.0])},
    }


class DomainRand:
    def __init__(self, cfg, num_envs):
        self.cfg = cfg or {}
        self.enabled = self.cfg.get("enabled", False)
        self.physical_resample = self.cfg.get("physical_resample", "startup")
        if self.physical_resample not in ("startup", "reset"):
            raise ValueError(f"physical_resample must be startup or reset, got {self.physical_resample}")
        self.num_envs = num_envs

        def buf(*shape):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=gs.tc_float)

        # sampled parameters, (num_envs, ...)
        self.mass = buf()
        self.inertia = buf(3)
        self.arm = buf()
        self.thrust_scale = buf(1)
        self.rpm_scale = buf(1)  # sqrt(thrust_scale), F ~ rpm^2
        self.motor_tau = buf(2)  # [up, down]
        self.action_delay = buf()

        self.drone = None
        self._bounds = {}  # name -> (lo, hi, shared) for randomized parameters

    def _spec(self, name):
        return self.cfg.get(name) or {}

    def _randomized(self, name):
        spec = self._spec(name)
        return self.enabled and ("range" in spec or "scale" in spec)

    @property
    def batch_links_info(self):
        """Whether the scene needs per-env link info (mass, inertia, arm randomized)."""
        return any(self._randomized(name) for name in PHYSICAL)

    @property
    def max_action_delay(self):
        name = "action_delay"
        if name in self._bounds:
            return float(self._bounds[name][1].max())
        return float(self._spec(name).get("nominal", 0.0))

    def build(self, drone):
        """After scene.build: read the urdf nominals from the drone, sample every env and write the physical parameters."""
        self.drone = drone
        base = drone.base_link
        self._base_idx = [base.idx_local]
        self._props_idx = [link.idx_local for link in drone.links if link.idx in set(drone.propellers_idx.tolist())]

        # the inertia is stored in the link's inertial frame (principal axes), R maps it to the body frame
        R = torch.tensor(gu.quat_to_R(np.asarray(base.desc.inertial_quat)), device=gs.device, dtype=gs.tc_float)
        self._R_inertial = R
        I_urdf = (R @ drone.get_links_inertia(self._base_idx)[..., 0, :, :].reshape(-1, 3, 3)[0] @ R.T).diagonal()
        self._props_com = drone.get_links_COM(self._props_idx).reshape(-1, len(self._props_idx), 3)[0]
        urdf = {
            "mass": drone.get_links_mass(self._base_idx).reshape(-1)[0].item(),
            "inertia": I_urdf.tolist(),
            "arm": self._props_com[:, :2].norm(dim=1).mean().item(),
        }
        self._arm_urdf = urdf["arm"]

        spec = self._spec("motor_tau")
        nominals = {
            **{name: self._spec(name).get("nominal", urdf[name]) for name in PHYSICAL},
            "thrust_scale": self._spec("thrust_scale").get("nominal", 1.0),
            "motor_tau": [spec.get("up", 0.0), spec.get("down", 0.0)],
            "action_delay": self._spec("action_delay").get("nominal", 0.0),
        }
        self._nominal = {k: torch.tensor(v, device=gs.device, dtype=gs.tc_float) for k, v in nominals.items()}

        for name, nominal in self._nominal.items():
            if not self._randomized(name):
                continue
            s = self._spec(name)
            lo, hi = (torch.tensor(v, device=gs.device, dtype=gs.tc_float) for v in s.get("range", s.get("scale")))
            if "range" not in s:
                lo, hi = lo * nominal, hi * nominal
            self._bounds[name] = (lo, hi, name == "motor_tau")  # one tau scale for spin up and down

        # physical parameters written per env at reset, or once here if only the nominal moved away from the urdf
        self._write_physical = [name for name in PHYSICAL if self._randomized(name)]
        fixed = [
            name for name in PHYSICAL
            if name not in self._write_physical and not np.allclose(nominals[name], urdf[name], rtol=1e-4)
        ]
        all_envs = torch.arange(self.num_envs, device=gs.device)
        self._sample(all_envs, self._nominal)
        if fixed:
            self._write(fixed, None)
        if self._write_physical:
            self._write(self._write_physical, all_envs)
        # parameters resampled at every reset
        self._resampled = [n for n in self._nominal if n not in PHYSICAL or self.physical_resample == "reset"]
        if self.physical_resample == "startup":
            self._write_physical = []

    def _sample(self, envs_idx, names):
        n = len(envs_idx)
        for name in names:
            nominal = self._nominal[name]
            out = getattr(self, name)
            if name not in self._bounds:
                out[envs_idx] = nominal
                continue
            lo, hi, shared = self._bounds[name]
            shape = (n, 1) if shared else (n, *out.shape[1:])
            out[envs_idx] = lo + (hi - lo) * torch.rand(shape, device=gs.device, dtype=gs.tc_float)
        self.rpm_scale[envs_idx] = torch.sqrt(self.thrust_scale[envs_idx])

    def _write(self, names, envs_idx):
        """Write the sampled physical parameters of envs_idx into Genesis (None: all envs, shared value)."""
        d = self.drone
        idx = slice(None) if envs_idx is None else envs_idx
        pick = (lambda x: x[0]) if envs_idx is None else (lambda x: x)
        if "mass" in names:
            d.set_links_mass(pick(self.mass[idx].unsqueeze(-1)), self._base_idx, envs_idx)
        if "inertia" in names:
            R = self._R_inertial
            I = R.T @ torch.diag_embed(self.inertia[idx]) @ R
            d.set_links_inertia(pick(I.unsqueeze(-3)), self._base_idx, envs_idx)
        if "arm" in names:
            com = self._props_com.expand(self.arm[idx].shape[0], -1, -1).clone()
            com[..., :2] *= (self.arm[idx] / self._arm_urdf)[:, None, None]
            d.set_links_COM(pick(com), self._props_idx, envs_idx)

    def reset_idx(self, envs_idx):
        """Resample the parameters of envs_idx and apply the physical ones (physical_resample: reset)."""
        self._sample(envs_idx, self._resampled)
        if self._write_physical:
            self._write(self._write_physical, envs_idx)
