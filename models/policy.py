"""Extractor de features para la politica PPO del modelo final."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.context_encoder import ContextEncoder
from models.market_encoder import build_market_encoder


def _build_gate_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    layers.extend([nn.Linear(prev_dim, output_dim), nn.Sigmoid()])
    gate = nn.Sequential(*layers)
    for module in gate.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    last_linear = [module for module in gate.modules() if isinstance(module, nn.Linear)][-1]
    nn.init.constant_(last_linear.bias, 0.0)
    return gate


class FullGatedExtractor(BaseFeaturesExtractor):
    """
    Extractor del modelo final.

    Combina una rama de mercado con una rama de contexto y modula la
    contribucion del contexto mediante dos compuertas:
    - una compuerta escalar de regimen
    - una compuerta vectorial sobre la representacion contextual
    """

    def __init__(
        self,
        observation_space: gym.Space,
        market_dim: int,
        context_dim: int,
        factor_dim: int,
        n_assets: int,
        market_feature_names: list[str],
        context_feature_names: list[str],
        factor_names: list[str],
        config: dict,
    ):
        feature_dim = int(config["model"]["market_encoder"]["output_dim"])
        super().__init__(observation_space, features_dim=feature_dim)

        self.market_dim = market_dim
        self.context_dim = context_dim
        self.factor_dim = factor_dim
        self.n_assets = n_assets
        self.market_feature_names = market_feature_names or []
        self.context_feature_names = context_feature_names or []
        self.factor_names = factor_names or []
        self.latest_gate: Optional[torch.Tensor] = None

        market_input_dim = market_dim + n_assets + factor_dim
        self.market_encoder = build_market_encoder(input_dim=market_input_dim, config=config)

        context_cfg = config["model"]["context_encoder"]
        self.context_encoder = ContextEncoder(
            input_dim=context_dim,
            hidden_dims=context_cfg["hidden_dims"],
            output_dim=context_cfg["output_dim"],
            dropout=context_cfg.get("dropout", 0.1),
        )
        self.context_proj = nn.Linear(context_cfg["output_dim"], self.features_dim)
        nn.init.orthogonal_(self.context_proj.weight, gain=0.01)
        nn.init.zeros_(self.context_proj.bias)

        gate_cfg = config["model"]["gating"]
        self.market_state_idx = [
            idx
            for idx, name in enumerate(self.market_feature_names)
            if str(name).startswith("__market_") or str(name).startswith("__benchmark_")
        ]
        self.regime_idx = [idx for idx, name in enumerate(self.context_feature_names) if str(name).startswith("regime_")]

        regime_gate_input_dim = self.features_dim * 2 + len(self.market_state_idx) + len(self.regime_idx) + factor_dim + 1
        asset_gate_input_dim = self.features_dim * 2 + len(self.market_state_idx) + factor_dim
        self.regime_gate = _build_gate_mlp(regime_gate_input_dim, 1, gate_cfg["hidden_dims"])
        self.asset_gate = _build_gate_mlp(
            asset_gate_input_dim,
            self.features_dim,
            gate_cfg.get("asset_hidden_dims", gate_cfg["hidden_dims"]),
        )

    def _split_obs(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        market = observations[..., : self.market_dim]
        offset = self.market_dim
        weights = observations[..., offset : offset + self.n_assets]
        offset += self.n_assets
        factor_exposures = (
            observations[..., offset : offset + self.factor_dim]
            if self.factor_dim > 0
            else observations.new_zeros((observations.shape[0], 0))
        )
        offset += self.factor_dim
        context = observations[..., offset : offset + self.context_dim]
        return market, weights, factor_exposures, context

    def _market_backbone(self, market: torch.Tensor, weights: torch.Tensor, factor_exposures: torch.Tensor) -> torch.Tensor:
        return self.market_encoder(torch.cat([market, weights, factor_exposures], dim=-1))

    def _vol_proxy(self, market: torch.Tensor) -> torch.Tensor:
        feature_names = [str(name) for name in self.market_feature_names]
        if "__market_realized_vol_20d" in feature_names:
            idx = feature_names.index("__market_realized_vol_20d")
            return market[..., idx : idx + 1]
        return market.std(dim=-1, keepdim=True)

    def _slice_features(self, market: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        market_state = (
            market[..., self.market_state_idx] if self.market_state_idx else market.new_zeros((market.shape[0], 0))
        )
        regime_state = (
            context[..., self.regime_idx] if self.regime_idx else market.new_zeros((market.shape[0], 0))
        )
        return market_state, regime_state

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market, weights, factor_exposures, context = self._split_obs(observations)
        market_embedding = self._market_backbone(market, weights, factor_exposures)
        context_embedding = self.context_proj(self.context_encoder(context))
        market_state, regime_state = self._slice_features(market, context)
        vol_proxy = self._vol_proxy(market)

        regime_gate_input = torch.cat(
            [market_embedding, context_embedding, market_state, regime_state, factor_exposures, vol_proxy],
            dim=-1,
        )
        asset_gate_input = torch.cat([market_embedding, context_embedding, market_state, factor_exposures], dim=-1)

        regime_gate = self.regime_gate(regime_gate_input)
        asset_gate = self.asset_gate(asset_gate_input)
        fused = market_embedding + regime_gate * asset_gate * context_embedding
        self.latest_gate = torch.cat([regime_gate, asset_gate.mean(dim=-1, keepdim=True)], dim=-1).detach()
        return fused


def build_policy_kwargs(*, env: Any, config: dict) -> Dict[str, Any]:
    return {
        "features_extractor_class": FullGatedExtractor,
        "features_extractor_kwargs": {
            "market_dim": int(env.market_dim),
            "context_dim": int(env.context_dim),
            "factor_dim": int(getattr(env, "factor_dim", 0)),
            "n_assets": int(env.n_assets),
            "market_feature_names": list(getattr(env, "market_feature_names", [])),
            "context_feature_names": list(getattr(env, "context_feature_names", [])),
            "factor_names": list(getattr(env, "factor_names", [])),
            "config": config,
        },
        "net_arch": {
            "pi": config["model"]["policy"]["net_arch"],
            "vf": config["model"]["policy"]["net_arch"],
        },
        "activation_fn": nn.Tanh,
    }
