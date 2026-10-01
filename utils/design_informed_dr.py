"""Design-informed domain randomization, Zhang et al. 2024, "A Learning-based Quadcopter Controller with
Extreme Adaptation" (arXiv 2409.12949), Sec. II.E.

A size factor c ~ U(c) sizes each drone between the min (c = 0, smallest) and max (c = 1, largest) design of
drone_cfg["design_informed"]: the arm length is linear in c, the mass scales as l^3, the inertia as l^5, kf
log-uniformly and the rest linearly. Every parameter then gets +- noise, each arm length +- arm_noise.

The drone is sampled once per env at build (physical_resample: startup) or at every reset (reset, ~2 ms per
step at 8192 envs for the Genesis writes). Motor efficiencies, action delay and hover estimate are sampled at
every reset. kf and the motor efficiencies act through the rpm (rpm_scale), km follows kf.
"""
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu


def lerp(spec, s):
    lo, hi = (torch.tensor(spec[k], device=gs.device, dtype=gs.tc_float) for k in ("min", "max"))
    return lo + s.unsqueeze(-1) * (hi - lo) if lo.dim() else lo + s * (hi - lo)


class DesignInformedDR:
    def __init__(self, drone_cfg, cfg, num_envs):
        self.design = drone_cfg["design_informed"]
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
        self._nominal = self.sample_design(torch.full((1,), 0.5, device=gs.device, dtype=gs.tc_float), 0.0, 0.0)

    batch_links_info = True  # mass, inertia and arm are written per env

    @property
    def max_action_delay(self):
        delay = self.cfg.get("action_delay", {})
        return float(delay["range"][1]) if self.enabled and "range" in delay else float(delay.get("nominal", 0.0))

    def nominal(self, name):
        """Parameters of the middle design, c = 0.5, without noise."""
        return self._nominal[name][0]

    def sample_design(self, c, noise, arm_noise):
        d = self.design
        n = len(c)

        def noisy(x, noise=noise):
            return x * (1.0 + noise * (2.0 * torch.rand_like(x) - 1.0))

        l_min, l_max = d["arm"]["min"], d["arm"]["max"]
        l = lerp(d["arm"], c)
        c_m = (l**3 - l_min**3) / (l_max**3 - l_min**3)
        c_J = (l**5 - l_min**5) / (l_max**5 - l_min**5)
        kf_min, kf_max = d["kf"]["min"], d["kf"]["max"]
        return {
            "arm": noisy(l.unsqueeze(-1).expand(n, 4).clone(), arm_noise),
            "mass": noisy(lerp(d["mass"], c_m)),
            "inertia": noisy(lerp(d["inertia"], c_J)),
            "kf": noisy((kf_min * (kf_max / kf_min) ** c).unsqueeze(-1)),
            "max_rpm": noisy(lerp(d["max_rpm"], c).unsqueeze(-1)),
            "motor_tau": noisy(lerp(d["motor_tau"], c)),
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
            lo, hi = self.design["c"]
            self.c[envs_idx] = lo + (hi - lo) * torch.rand(len(envs_idx), device=gs.device, dtype=gs.tc_float)
            params = self.sample_design(self.c[envs_idx], self.design["noise"], self.design["arm_noise"])
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
