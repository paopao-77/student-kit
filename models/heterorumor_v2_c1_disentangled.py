import math
from typing import Any

import torch
from torch import nn

from models.heterorumor_v1 import HeteroRumorV1


class HeteroRumorV2C1Disentangled(HeteroRumorV1):
    """Separates content factors from propagation-dynamics factors."""

    def __init__(
        self,
        text_dim: int,
        node_dim: int,
        global_dim: int,
        temporal_dim: int,
        user_dim: int,
        hidden_dim: int = 64,
        graph_layers: int = 2,
        dropout: float = 0.3,
        latent_dim: int = 4,
        content_latent_dim: int = 4,
    ) -> None:
        super().__init__(
            text_dim=text_dim,
            node_dim=node_dim,
            global_dim=global_dim,
            temporal_dim=temporal_dim,
            user_dim=user_dim,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )
        self.latent_dim = latent_dim
        self.content_latent_dim = content_latent_dim

        self.content_factor_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.content_mu = nn.Linear(hidden_dim, content_latent_dim)
        self.content_logvar = nn.Linear(hidden_dim, content_latent_dim)
        self.content_decoder = nn.Sequential(
            nn.Linear(content_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        dynamics_input_dim = hidden_dim * 3 + 3
        self.dynamics_factor_encoder = nn.Sequential(
            nn.Linear(dynamics_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dynamics_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.dynamics_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.dynamics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
        )

        prediction_dim = latent_dim + content_latent_dim
        self.disentangled_prediction_head = nn.Sequential(
            nn.Linear(prediction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.disentangled_growth_probability_head = nn.Linear(prediction_dim, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encode_batch(batch)
        text_representation, graph_representation, temporal_representation, user_representation = (
            encoded["representations"]
        )
        mask = batch["modality_mask"]

        content_target = text_representation * mask[:, 0:1]
        content_hidden = self.content_factor_encoder(
            torch.cat([content_target, mask[:, 0:1]], dim=1)
        )
        content_mu = self.content_mu(content_hidden)
        content_logvar = self.content_logvar(content_hidden).clamp(min=-10.0, max=8.0)
        content_latent = self.reparameterize(content_mu, content_logvar)
        content_reconstruction = self.content_decoder(content_latent)

        dynamics_representations = torch.stack(
            [graph_representation, temporal_representation, user_representation], dim=1
        )
        dynamics_mask = mask[:, 1:]
        masked_dynamics = dynamics_representations * dynamics_mask.unsqueeze(-1)
        dynamics_target = masked_dynamics.flatten(start_dim=1)
        dynamics_hidden = self.dynamics_factor_encoder(
            torch.cat([dynamics_target, dynamics_mask], dim=1)
        )
        dynamics_mu = self.dynamics_mu(dynamics_hidden)
        dynamics_logvar = self.dynamics_logvar(dynamics_hidden).clamp(min=-10.0, max=8.0)
        dynamics_latent = self.reparameterize(dynamics_mu, dynamics_logvar)
        dynamics_reconstruction = self.dynamics_decoder(dynamics_latent)

        joint_latent = torch.cat([dynamics_latent, content_latent], dim=1)
        raw_log_growth = self.disentangled_prediction_head(joint_latent).squeeze(1)
        log_growth_magnitude = torch.nn.functional.softplus(raw_log_growth)
        growth_logit = self.disentangled_growth_probability_head(joint_latent).squeeze(1)
        growth_probability = torch.sigmoid(growth_logit)
        predicted_growth = growth_probability * torch.expm1(
            log_growth_magnitude.clamp(max=math.log(1e6))
        )
        predicted_final_size = batch["observed_sizes"] + predicted_growth

        return {
            "log_growth": torch.log1p(predicted_growth),
            "log_growth_magnitude": log_growth_magnitude,
            "growth_logit": growth_logit,
            "growth_probability": growth_probability,
            "predicted_final_size": predicted_final_size,
            "fusion_weights": encoded["fusion_weights"],
            "latent": dynamics_latent,
            "latent_mu": dynamics_mu,
            "latent_logvar": dynamics_logvar,
            "content_latent_mu": content_mu,
            "content_latent_logvar": content_logvar,
            "vae_mu": torch.cat([dynamics_mu, content_mu], dim=1),
            "vae_logvar": torch.cat([dynamics_logvar, content_logvar], dim=1),
            "reconstruction": torch.cat(
                [dynamics_reconstruction, content_reconstruction], dim=1
            ),
            "reconstruction_target": torch.cat(
                [dynamics_target.detach(), content_target.detach()], dim=1
            ),
        }
