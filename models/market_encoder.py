"""Codificador MLP para el bloque de mercado y estado de cartera."""

from typing import List

import torch
import torch.nn as nn


class MarketEncoder(nn.Module):
    """Resume las features de mercado, pesos previos y exposiciones factoriales."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        activation_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    activation_cls(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, market_state: torch.Tensor) -> torch.Tensor:
        return self.net(market_state)


def build_market_encoder(input_dim: int, config: dict) -> nn.Module:
    encoder_cfg = config["model"]["market_encoder"]
    return MarketEncoder(
        input_dim=input_dim,
        hidden_dims=encoder_cfg["hidden_dims"],
        output_dim=encoder_cfg["output_dim"],
        dropout=encoder_cfg.get("dropout", 0.1),
    )
