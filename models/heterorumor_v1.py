import math
from typing import Any

import torch
from torch import nn


class DirectionalGraphConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(x)
        if src.numel() > 0:
            aggregate.index_add_(0, dst, x[src])
            degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            degree.index_add_(
                0,
                dst,
                torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device),
            )
            aggregate = aggregate / degree.clamp_min(1.0)
        hidden = self.self_linear(x) + self.message_linear(aggregate)
        return torch.relu(self.norm(self.dropout(hidden)))


def graph_pool(x: torch.Tensor, graph_id: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = []
    for graph_index in range(num_graphs):
        graph_x = x[graph_id == graph_index]
        if graph_x.shape[0] == 0:
            zeros = torch.zeros(x.shape[1] * 2, dtype=x.dtype, device=x.device)
            pooled.append(zeros)
        else:
            pooled.append(torch.cat([graph_x.mean(dim=0), graph_x.max(dim=0).values], dim=0))
    return torch.stack(pooled, dim=0)


class DirectionalGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.top_down_layers = nn.ModuleList(
            DirectionalGraphConv(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.bottom_up_layers = nn.ModuleList(
            DirectionalGraphConv(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4 + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        graph_id: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(node_features)
        if edge_index.numel() == 0:
            top_down_src = edge_index.new_empty((0,))
            top_down_dst = edge_index.new_empty((0,))
        else:
            top_down_src, top_down_dst = edge_index[0], edge_index[1]

        top_down = hidden
        bottom_up = hidden
        for top_down_layer, bottom_up_layer in zip(
            self.top_down_layers, self.bottom_up_layers
        ):
            top_down = top_down_layer(top_down, top_down_src, top_down_dst)
            bottom_up = bottom_up_layer(bottom_up, top_down_dst, top_down_src)

        num_graphs = global_features.shape[0]
        graph_representation = torch.cat(
            [
                graph_pool(top_down, graph_id, num_graphs),
                graph_pool(bottom_up, graph_id, num_graphs),
                global_features,
            ],
            dim=1,
        )
        return self.output(graph_representation)


class MaskAwareFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float) -> None:
        super().__init__()
        self.modality_bias = nn.Parameter(torch.zeros(num_modalities))
        self.gates = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_modalities))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        representations: list[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(representations, dim=1)
        logits = torch.cat(
            [gate(representation) for gate, representation in zip(self.gates, representations)],
            dim=1,
        )
        logits = logits + self.modality_bias.unsqueeze(0)
        safe_mask = modality_mask.clone()
        no_modality = safe_mask.sum(dim=1) == 0
        if no_modality.any():
            safe_mask[no_modality, 0] = 1.0
        logits = logits.masked_fill(safe_mask <= 0, -1e4)
        weights = torch.softmax(logits, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(torch.cat([fused, modality_mask], dim=1)), weights


class HeteroRumorV1(nn.Module):
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
    ) -> None:
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.graph_encoder = DirectionalGraphEncoder(
            node_dim=node_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            dropout=dropout,
        )
        self.temporal_encoder = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = MaskAwareFusion(hidden_dim, num_modalities=4, dropout=dropout)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.growth_probability_head = nn.Linear(hidden_dim, 1)

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        text_representation = self.text_encoder(batch["text_features"])
        graph_representation = self.graph_encoder(
            node_features=batch["node_features"],
            edge_index=batch["edge_index"],
            graph_id=batch["graph_id"],
            global_features=batch["global_features"],
        )
        temporal_output, _state = self.temporal_encoder(batch["temporal_features"])
        lengths = batch["temporal_mask"].sum(dim=1).long().clamp_min(1)
        last_indices = lengths - 1
        temporal_representation = temporal_output[
            torch.arange(temporal_output.shape[0], device=temporal_output.device),
            last_indices,
        ]
        temporal_representation = torch.relu(self.temporal_norm(temporal_representation))
        user_representation = self.user_encoder(batch["user_features"])

        fused, fusion_weights = self.fusion(
            [
                text_representation,
                graph_representation,
                temporal_representation,
                user_representation,
            ],
            batch["modality_mask"],
        )
        return {
            "representations": [
                text_representation,
                graph_representation,
                temporal_representation,
                user_representation,
            ],
            "fused_representation": fused,
            "fusion_weights": fusion_weights,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encode_batch(batch)
        fused = encoded["fused_representation"]
        raw_log_growth = self.prediction_head(fused).squeeze(1)
        log_growth_magnitude = torch.nn.functional.softplus(raw_log_growth)
        growth_logit = self.growth_probability_head(fused).squeeze(1)
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
        }
