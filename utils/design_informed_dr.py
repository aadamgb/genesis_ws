"""Design-informed domain randomization, Zhang et al. 2024, "A Learning-based Quadcopter Controller with
Extreme Adaptation" (arXiv 2409.12949), Sec. II.E.

A size factor c ~ U(c) sizes each drone between the min (c = 0, smallest) and max (c = 1, largest) design of
drone_cfg["design_informed"]: the arm length is linear in c, the mass scales as l^3, the inertia as l^5, kf
log-uniformly and the rest linearly. Every parameter then gets +- noise (or its own noise: entry), each arm
length +- arm_noise. The motor time constants do not follow the size (the small a300 has the slowest motors):
tau_up is log-uniform in motor_tau.up and tau_down = tau_up x U(motor_tau.down_ratio), each then +- motor_tau.noise
(default 0).

Optional, decoupled from the size: twr: [lo, hi] samples the thrust-to-weight ratio log-uniformly
and sets max_rpm from it (no max_rpm entry then), inertia_factor: [lo, hi] multiplies the inertia by a
log-uniform factor (drones heavier at the rim than the l^5 law, e.g. RoboFly x2.2).

The drone is sampled once per env at build (physical_resample: startup) or at every reset (reset, ~2 ms per
step at 8192 envs for the Genesis writes). Motor efficiencies, action delay and hover estimate are sampled at
every reset. kf and the motor efficiencies act through the rpm (rpm_scale), km follows kf.

Several drones can share a design (pursuer: both drones of an env from the same c and size-independent draws, each
with its own noise): set the same draw(num_envs) as their shared before build.
"""
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu


def lerp(spec, s):
    lo, hi = (torch.tensor(spec[k], device=gs.device, dtype=gs.tc_float) for k in ("min", "max"))
    return lo + s.unsqueeze(-1) * (hi - lo) if lo.dim() else lo + s * (hi - lo)


class DesignInformedDR:
    def __init__(self, drone_cfg, cfg, num_envs, shared=None):
        self.design = drone_cfg["design_informed"]
        self.shared = shared  # {"c": (num_envs,), "u": (num_envs, 4)} from draw(), or None: own draws
        self.cfg = cfg
        self.num_envs = num_envs
        self.enabled = cfg.get("enabled", True)
        self.resample_drone = cfg.get("physical_resample", "startup") == "reset"
        self.hover_noise = cfg.get("hover_estimate_noise", 0.0)

        def buf(*shape):
            return torch.zeros((num_envs, *shape), device=gs.device, dtype=gs.tc_float)

        self.c = buf()
        self.arm = buf(4)
        self.mass = buf()
        self.inertia = buf(3)
        self.kf = buf(1)
        self.max_rpm = buf(1)
        self.motor_tau = buf(2)
        self.motor_eff = buf(4)
        self.action_delay = buf()
        self.rpm_scale = buf(4)
        self.hover_rpm = buf(1)
        self.hover_u = buf(1)

        kf = self.design["kf"]
        self.kf_ref = float(np.sqrt(kf["min"] * kf["max"]))
        half = lambda *shape: torch.full(shape, 0.5, device=gs.device, dtype=gs.tc_float)
        self._nominal = self.sample_design(half(1), half(1, 4), noise=False)

    batch_links_info = True  # mass, inertia and arm are written per env

    @property
    def max_action_delay(self):
        delay = self.cfg.get("action_delay", {})
        return float(delay["range"][1]) if self.enabled and "range" in delay else float(delay.get("nominal", 0.0))

    def nominal(self, name):
        """Parameters of the middle design, c = 0.5, without noise."""
        return self._nominal[name][0]

    def draw(self, n):
        """Size factors c (n,) and size-independent draws u (n, 4) in [0, 1]: inertia factor, twr, tau_up, tau ratio."""
        lo, hi = self.design["c"]
        rand = lambda *shape: torch.rand((n, *shape), device=gs.device, dtype=gs.tc_float)
        return {"c": lo + (hi - lo) * rand(), "u": rand(4)}

    def sample_design(self, c, u, noise=True):
        """Design of size c with the size-independent draws u (n, 4), +- the noise on every parameter (noise)."""
        d = self.design
        n = len(c)

        def noisy(x, name):
            a = d[name].get("noise", d["noise"]) if noise else 0.0
            return x * (1.0 + a * (2.0 * torch.rand_like(x) - 1.0))

        def uniform(lo, hi, i, log=False):
            return lo * (hi / lo) ** u[:, i] if log else lo + (hi - lo) * u[:, i]

        a_tau = d["motor_tau"].get("noise", 0.0) if noise else 0.0
        noisy_tau = lambda x: x * (1.0 + a_tau * (2.0 * torch.rand_like(x) - 1.0))
        tau_up = noisy_tau(uniform(*d["motor_tau"]["up"], 2, log=True))
        tau_down = noisy_tau(tau_up * uniform(*d["motor_tau"]["down_ratio"], 3))
        arm_noise = d["arm_noise"] if noise else 0.0

        l_min, l_max = d["arm"]["min"], d["arm"]["max"]
        l = lerp(d["arm"], c)
        c_m = (l**3 - l_min**3) / (l_max**3 - l_min**3)
        c_J = (l**5 - l_min**5) / (l_max**5 - l_min**5)
        kf_min, kf_max = d["kf"]["min"], d["kf"]["max"]
        mass = noisy(lerp(d["mass"], c_m), "mass")
        inertia = noisy(lerp(d["inertia"], c_J), "inertia")
        if "inertia_factor" in d:
            inertia = inertia * uniform(*d["inertia_factor"], 0, log=True).unsqueeze(-1)
        kf = noisy((kf_min * (kf_max / kf_min) ** c).unsqueeze(-1), "kf")
        if "twr" in d:
            max_rpm = torch.sqrt(9.81 * mass.unsqueeze(-1) / 4.0 / kf * uniform(*d["twr"], 1, log=True).unsqueeze(-1))
        else:
            max_rpm = noisy(lerp(d["max_rpm"], c).unsqueeze(-1), "max_rpm")
        return {
            "arm": l.unsqueeze(-1) * (1.0 + arm_noise * (2.0 * torch.rand((n, 4), device=gs.device) - 1.0)),
            "mass": mass,
            "inertia": inertia,
            "kf": kf,
            "max_rpm": max_rpm,
            "motor_tau": torch.stack([tau_up, tau_down], dim=-1),
        }

    def build(self, drone):
        self.drone = drone
        self._base_idx = [drone.base_link.idx_local]
        self._props_idx = [link.idx_local for link in drone.links if link.idx in set(drone.propellers_idx.tolist())]
        # the inertia is stored in the link's principal-axes frame, R rotates it to the body frame
        self._R = torch.tensor(gu.quat_to_R(np.asarray(drone.base_link.desc.inertial_quat)), device=gs.device, dtype=gs.tc_float)
        com = drone.get_links_COM(self._props_idx).reshape(-1, 4, 3)[0]
        self._props_dir = com[:, :2] / com[:, :2].norm(dim=1, keepdim=True)
        self._props_z = com[:, 2]

        # shared by all envs, read by set_propellers_rpm at every call
        drone._desc.kf = self.kf_ref
        drone._desc.km = self.kf_ref * self.design["km_kf"]

        all_envs = torch.arange(self.num_envs, device=gs.device)
        self._sample_drone(all_envs)
        self._sample_episode(all_envs)

    def _sample_drone(self, envs_idx):
        if self.enabled:
            draws = self.draw(len(envs_idx)) if self.shared is None else {k: v[envs_idx] for k, v in self.shared.items()}
            self.c[envs_idx] = draws["c"]
            params = self.sample_design(draws["c"], draws["u"])
        else:
            self.c[envs_idx] = 0.5
            params = {k: v.expand(len(envs_idx), *v.shape[1:]) for k, v in self._nominal.items()}
        for name, value in params.items():
            getattr(self, name)[envs_idx] = value
        self._write(envs_idx)

    def _sample_episode(self, envs_idx):
        n = len(envs_idx)
        rand = lambda *shape: torch.rand((n, *shape), device=gs.device, dtype=gs.tc_float)
        if self.enabled:
            lo, hi = self.design["motor_eff"]
            self.motor_eff[envs_idx] = lo + (hi - lo) * rand(4)
        else:
            self.motor_eff[envs_idx] = 1.0
        delay = self.cfg.get("action_delay", {})
        if self.enabled and "range" in delay:
            lo, hi = delay["range"]
            self.action_delay[envs_idx] = lo + (hi - lo) * rand()
        else:
            self.action_delay[envs_idx] = delay.get("nominal", 0.0)

        self.rpm_scale[envs_idx] = torch.sqrt(self.kf[envs_idx] / self.kf_ref) * self.motor_eff[envs_idx]
        self.hover_rpm[envs_idx] = torch.sqrt(9.81 * self.mass[envs_idx, None] / 4.0 / self.kf[envs_idx])
        estimate = 1.0 + self.hover_noise * (2.0 * rand(1) - 1.0)
        self.hover_u[envs_idx] = self.hover_rpm[envs_idx] / self.max_rpm[envs_idx] * estimate

    def _write(self, envs_idx):
        d, R = self.drone, self._R
        d.set_links_mass(self.mass[envs_idx, None], self._base_idx, envs_idx)
        d.set_links_inertia((R.T @ torch.diag_embed(self.inertia[envs_idx]) @ R).unsqueeze(1), self._base_idx, envs_idx)
        com = torch.cat([self._props_dir * self.arm[envs_idx, :, None], self._props_z.expand(len(envs_idx), 4)[..., None]], dim=-1)
        d.set_links_COM(com, self._props_idx, envs_idx)

    def reset_idx(self, envs_idx):
        if self.resample_drone:
            self._sample_drone(envs_idx)
        self._sample_episode(envs_idx)

    def resample(self, envs_idx):
        """A new drone for envs_idx now, mid-episode (evader rerandomize)."""
        self._sample_drone(envs_idx)
        self._sample_episode(envs_idx)

    def nominal_cfg(self):
        """Middle design as a drone config for build_controller (drone_params), the per-env values set after build."""
        kf = self.nominal("kf").item()
        return {
            "mass": {"nominal": self.nominal("mass").item()},
            "kf": {"nominal": kf},
            "km": {"nominal": kf * self.design["km_kf"]},
            "max_rpm": self.nominal("max_rpm").item(),
        }
