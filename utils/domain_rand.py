"""Domain randomization: per-env drone parameters and action delay.

The drone parameters and their ranges come from the drone config (hydra_configs/drone/<name>.yaml):

    mass:      {nominal, range | scale}   [kg] base link
    inertia:   {nominal, range | scale}   [kg m^2] Jxx, Jyy, Jzz in the body frame
    arm:       {nominal, range | scale}   [m] rotor distance from the COM, scales the rotor xy positions
    kf:        {nominal, range | scale}   [N/rpm^2] thrust coefficient, km follows it
    km:        {nominal}                  [Nm/rpm^2]
    motor_tau: {up, down, range | scale}  [s] first-order rpm lag, one sample shared by up and down
    motor_eff: {nominal, range | scale}   [-] per motor, multiplies its rpm command

and the task ones from env_cfg["domain_rand"]:

    enabled: true                 # false uses the nominals everywhere
    physical_resample: startup    # when mass, inertia and arm are sampled: startup (once per env) or reset
    action_delay: {nominal, range | scale}   [s]
    hover_estimate_noise: 0.05    # hover_u per env from the sampled mass and kf, times 1 +- this (uniform,
                                  # every reset), like PX4's hover thrust estimate; absent: nominal hover

range: [lo, hi] samples uniformly in physical units, scale: [lo, hi] samples a multiplier of the nominal
(lo and hi may be lists for vector parameters). Without either the nominal is used.

The nominals of mass, inertia, arm, kf and km are written into Genesis at build, over the urdf values.
Mass, inertia and arm are written per env when randomized, which needs the scene built with
RigidOptions(batch_links_info=DomainRand.batch_links_info) and the drone loaded with align=False. Every
write costs ~0.5 ms whatever the number of envs, and with thousands of envs some reset at almost every
step, so physical_resample: reset adds ~1.5-2.5 ms per step (8192 envs, RTX 4070 laptop); startup is free.
kf and motor_eff act through the rpm (rpm_scale), so the other parameters are plain tensors resampled at
every reset into preallocated (num_envs, ...) buffers.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

import genesis as gs
import genesis.utils.geom as gu

PHYSICAL = ("mass", "inertia", "arm")
DRONE = ("mass", "inertia", "arm", "kf", "motor_tau", "motor_eff")


def load_drone_cfg(name):
    return OmegaConf.to_container(OmegaConf.load(f"hydra_configs/drone/{name}.yaml"), resolve=True)


def from_legacy_env_cfg(env_cfg):
    """(drone_cfg, domain_rand_cfg) for env configs saved before the drone configs existed: the a300 with the
    ranges they were trained with, and no motor_eff."""
    if "drone" in env_cfg:
        return env_cfg["drone"], env_cfg["domain_rand"]
    drone = load_drone_cfg("a300")
    del drone["motor_eff"]
    for name in PHYSICAL + ("kf", "motor_tau"):
        drone[name].pop("range", None), drone[name].pop("scale", None)
    old = env_cfg.get("domain_rand")
    if old is not None:  # domain_rand with the drone parameters in it
        for name in PHYSICAL + ("motor_tau",):
            drone[name].update(old.get(name, {}))
        thrust = old.get("thrust_scale", {})
        dr = {k: old[k] for k in ("enabled", "physical_resample", "action_delay") if k in old}
    else:  # flat keys in env_cfg
        thrust = {"range": env_cfg.get("thrust_scale_range", [1.0, 1.0])}
        drone["motor_tau"] = {
            "up": env_cfg.get("motor_tau_up", 0.0),
            "down": env_cfg.get("motor_tau_down", 0.0),
            "scale": env_cfg.get("motor_tau_scale_range", [1.0, 1.0]),
        }
        dr = {"enabled": True, "action_delay": {"nominal": 0.0, "range": env_cfg.get("action_delay_range", [0.0, 0.0])}}
    if "range" in thrust or "scale" in thrust:  # thrust scale = kf / kf nominal
        drone["kf"]["scale"] = thrust.get("range", thrust.get("scale"))
    return drone, dr


class DomainRand:
    def __init__(self, drone_cfg, cfg, num_envs):
        self.drone_cfg = drone_cfg
        self.cfg = cfg or {}
        self.enabled = self.cfg.get("enabled", False)
        self.physical_resample = self.cfg.get("physical_resample", "startup")
        if self.physical_resample not in ("startup", "reset"):
            raise ValueError(f"physical_resample must be startup or reset, got {self.physical_resample}")
        if "range" in self._spec("km") or "scale" in self._spec("km"):
            raise ValueError("km cannot be randomized on its own, it follows kf")
        self.num_envs = num_envs

        def buf(*shape):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=gs.tc_float)

        # sampled parameters, (num_envs, ...)
        self.mass = buf()
        self.inertia = buf(3)
        self.arm = buf()
        self.kf = buf(1)
        self.motor_tau = buf(2)  # [up, down]
        self.motor_eff = buf(4)
        self.action_delay = buf()
        self.rpm_scale = buf(4)  # sqrt(kf / kf nominal) * motor_eff, F ~ rpm^2
        self.hover_rpm = buf(1)  # true hover rpm command (sampled mass and kf, motor_eff left out)
        self.hover_u = buf(1)    # estimated hover command, hover_rpm / max_rpm with the estimate error
        self.hover_noise = self.cfg.get("hover_estimate_noise")

        self.drone = None
        self._bounds = {}  # name -> (lo, hi, shared) for randomized parameters

    def _spec(self, name):
        src = self.drone_cfg if name in DRONE + ("km",) else self.cfg
        return src.get(name) or {}

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
        if self._randomized(name):
            s = self._spec(name)
            hi = s["range"][1] if "range" in s else s["scale"][1] * s.get("nominal", 0.0)
            return float(np.max(hi))
        return float(self._spec(name).get("nominal", 0.0))

    def nominal(self, name):
        return self._nominal[name]

    def build(self, drone):
        """After scene.build: write the nominals into Genesis, sample every env and write the physical ones."""
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

        # kf and km are the drone's (shared) coefficients, read by set_propellers_rpm at every call
        drone._desc.kf = self._spec("kf").get("nominal", drone.KF)
        drone._desc.km = self._spec("km").get("nominal", drone.KM)

        tau = self._spec("motor_tau")
        nominals = {
            **{name: self._spec(name).get("nominal", urdf[name]) for name in PHYSICAL},
            "kf": drone.KF,
            "motor_tau": [tau.get("up", 0.0), tau.get("down", 0.0)],
            "motor_eff": self._spec("motor_eff").get("nominal", 1.0),
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

        # physical parameters written per env, or once here if only the nominal differs from the urdf
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
        self.rpm_scale[envs_idx] = torch.sqrt(self.kf[envs_idx] / self._nominal["kf"]) * self.motor_eff[envs_idx]
        self._update_hover(envs_idx)

    def _update_hover(self, envs_idx):
        self.hover_rpm[envs_idx] = torch.sqrt(9.81 * self.mass[envs_idx, None] / 4.0 / self.kf[envs_idx])
        err = 1.0 + (self.hover_noise or 0.0) * (2.0 * torch.rand_like(self.hover_rpm[envs_idx]) - 1.0)
        self.hover_u[envs_idx] = self.hover_rpm[envs_idx] / self.drone_cfg["max_rpm"] * err

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

    def set_nominal(self, envs_idx=None):
        """Put envs_idx (default: all) on the nominal parameters, the physical ones written per env."""
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)
        for name, nominal in self._nominal.items():
            getattr(self, name)[envs_idx] = nominal
        self.rpm_scale[envs_idx] = self.motor_eff[envs_idx]
        self._update_hover(envs_idx)
        written = [name for name in PHYSICAL if self._randomized(name)]
        if written:
            self._write(written, envs_idx)

    def reset_idx(self, envs_idx):
        """Resample the parameters of envs_idx and apply the physical ones (physical_resample: reset)."""
        self._sample(envs_idx, self._resampled)
        if self._write_physical:
            self._write(self._write_physical, envs_idx)
