import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization


class EncoderMLPModel(MLPModel):
    """MLPModel whose encoder_obs_group (the drone parameters) is compressed by an MLP encoder into a latent vector,
    concatenated with the other observation groups and passed to the MLP. The encoder is trained end-to-end with
    the rest of the model by the RL loss."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        encoder_obs_group: str = "params",
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        latent_dim: int = 8,
        latent_activation: str | None = None,
        activation: str = "elu",
        obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        self.encoder_obs_group = encoder_obs_group
        self.encoder_input_dim = obs[encoder_obs_group].shape[-1]
        self.latent_dim = latent_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, activation=activation, obs_normalization=obs_normalization, **kwargs)
        self.mlp_obs_groups = [g for g in self.obs_groups if g != encoder_obs_group]
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim - self.encoder_input_dim)
        self.encoder = MLP(self.encoder_input_dim, latent_dim, encoder_hidden_dims, activation, latent_activation)
        self.latent_activation = latent_activation

    def encode(self, obs: TensorDict) -> torch.Tensor:
        return self.encoder(obs[self.encoder_obs_group])

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        x = self.obs_normalizer(torch.cat([obs[g] for g in self.mlp_obs_groups], dim=-1))
        return torch.cat([x, self.encode(obs)], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(torch.cat([obs[g] for g in self.mlp_obs_groups], dim=-1))

    def _get_latent_dim(self) -> int:
        return self.obs_dim - self.encoder_input_dim + self.latent_dim

    def as_jit(self, params_ref: torch.Tensor | None = None) -> nn.Module:
        """Exportable model taking [policy observation, params] concatenated. params_ref (the env's
        params_reference(), the values the params are normalized with) is stored in the file for deployment."""
        return _TorchEncoderMLPModel(self, params_ref)


class _TorchEncoderMLPModel(nn.Module):
    """TorchScript export of EncoderMLPModel. Parameter names start with encoder. or mlp. so a deployment can
    evaluate the two MLPs separately; latent_tanh says whether the encoder output goes through tanh."""

    def __init__(self, model: EncoderMLPModel, params_ref: torch.Tensor | None) -> None:
        super().__init__()
        if model.obs_normalization:
            raise NotImplementedError("export with observation normalization")
        if model.latent_activation not in (None, "tanh"):
            raise NotImplementedError(f"latent activation {model.latent_activation}")
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()
        self.policy_dim: int = model.obs_dim - model.encoder_input_dim
        self.latent_tanh: bool = model.latent_activation == "tanh"
        self.register_buffer("params_ref", torch.zeros(0) if params_ref is None else params_ref.detach().cpu().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x[..., self.policy_dim :])
        return self.deterministic_output(self.mlp(torch.cat([x[..., : self.policy_dim], z], dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass
