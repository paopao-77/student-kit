import math
from typing import Any

import torch
from torch import nn

from models.heterorumor_v1 import HeteroRumorV1


class HeteroRumorV2C1(HeteroRumorV1):
    """V2/C1 model with a variational propagation-momentum factor bottleneck."""

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
        latent_dim: int = 16,
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
        factor_input_dim = hidden_dim * 4 + 4
        self.factor_encoder = nn.Sequential(
            nn.Linear(factor_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.factor_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.factor_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.factor_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
        )
        self.latent_prediction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.latent_growth_probability_head = nn.Linear(latent_dim, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encode_batch(batch)
        mask = batch["modality_mask"]
        stacked = torch.stack(encoded["representations"], dim=1)
        masked = stacked * mask.unsqueeze(-1)
        reconstruction_target = masked.flatten(start_dim=1).detach()
        factor_input = torch.cat([masked.flatten(start_dim=1), mask], dim=1)
        factor_hidden = self.factor_encoder(factor_input)
        mu = self.factor_mu(factor_hidden)
        logvar = self.factor_logvar(factor_hidden).clamp(min=-10.0, max=8.0)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.factor_decoder(latent)

        raw_log_growth = self.latent_prediction_head(latent).squeeze(1)
        log_growth_magnitude = torch.nn.functional.softplus(raw_log_growth)
        growth_logit = self.latent_growth_probability_head(latent).squeeze(1)
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
            "latent": latent,
            "latent_mu": mu,
            "latent_logvar": logvar,
            "reconstruction": reconstruction,
            "reconstruction_target": reconstruction_target,
        }
