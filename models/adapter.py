"""
LoRA adapters for game-specific fine-tuning.

Each game gets a tiny set of LoRA weight pairs injected into the Q, K, V,
and output projections of every Transformer layer. The backbone stays frozen;
only the adapter weights are trained when learning a new game.

LoRA: output = W_frozen(x) + (alpha/r) * lora_B(lora_A(x))
  lora_A: d_model → r   (random init)
  lora_B: r → d_model   (zero init → adapter starts as identity)
  scale  = alpha / r

Params per game: 8 layers × 4 projections × 2 matrices × (512×16) = ~0.52M
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.types import GameID
from models.backbone import CategoryExpertModel, D_MODEL, N_LAYERS

LORA_RANK = 16
LORA_ALPHA = 32
LORA_SCALE = LORA_ALPHA / LORA_RANK   # = 2.0

# Which projections inside MultiheadAttention to patch
_PROJ_NAMES = ("q", "k", "v", "out")


# ---------------------------------------------------------------------------
# Single LoRA weight pair
# ---------------------------------------------------------------------------

class LoRAWeights:
    """Holds one A+B pair for a single projection in a single layer."""

    def __init__(self, d_model: int = D_MODEL, r: int = LORA_RANK) -> None:
        self.lora_A = nn.Parameter(torch.empty(r, d_model))
        self.lora_B = nn.Parameter(torch.zeros(d_model, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA correction: LORA_SCALE * lora_B @ lora_A @ x^T."""
        return LORA_SCALE * (x @ self.lora_A.T @ self.lora_B.T)

    def parameters(self):
        return [self.lora_A, self.lora_B]

    def state_dict(self) -> dict:
        return {"lora_A": self.lora_A.data, "lora_B": self.lora_B.data}

    def load_state_dict(self, state: dict) -> None:
        self.lora_A.data.copy_(state["lora_A"])
        self.lora_B.data.copy_(state["lora_B"])


import math


# ---------------------------------------------------------------------------
# Full adapter for one game (all layers, all projections)
# ---------------------------------------------------------------------------

class GameAdapter(nn.Module):
    """
    Complete LoRA adapter for a single game.
    Contains N_LAYERS × 4 LoRAWeights pairs.

    Usage:
        adapter = GameAdapter()
        adapter.inject(backbone)   # patches the backbone's attention layers
        adapter.eject(backbone)    # restores the backbone to clean state
    """

    def __init__(
        self,
        game_id: GameID,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        r: int = LORA_RANK,
    ) -> None:
        super().__init__()
        self.game_id = game_id
        self.d_model = d_model
        self.n_layers = n_layers
        self.r = r

        # weights[layer_idx][proj_name] = LoRAWeights
        self.weights: nn.ModuleDict = nn.ModuleDict()
        for layer_idx in range(n_layers):
            for proj in _PROJ_NAMES:
                key = f"L{layer_idx}_{proj}"
                # Register as a module so optimizer finds them
                self.weights[key] = _LoRAWeightsModule(d_model, r)

    def inject(self, backbone: CategoryExpertModel) -> None:
        """Monkey-patch backbone attention layers to add LoRA corrections."""
        for layer_idx, layer in enumerate(backbone.layers):
            for proj in _PROJ_NAMES:
                key = f"L{layer_idx}_{proj}"
                lora = self.weights[key]
                _patch_proj(layer.attn, proj, lora)

    def eject(self, backbone: CategoryExpertModel) -> None:
        """Remove LoRA patches from backbone, restoring original forward."""
        for layer in backbone.layers:
            _unpatch_attn(layer.attn)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _LoRAWeightsModule(nn.Module):
    """nn.Module wrapper around a LoRA A+B pair so it appears in state_dict."""

    def __init__(self, d_model: int, r: int) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, d_model))
        self.lora_B = nn.Parameter(torch.zeros(d_model, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return LORA_SCALE * (x @ self.lora_A.T @ self.lora_B.T)


# ---------------------------------------------------------------------------
# Projection patching helpers
# ---------------------------------------------------------------------------

def _patch_proj(
    mha: nn.MultiheadAttention,
    proj: str,
    lora: _LoRAWeightsModule,
) -> None:
    """
    Replace the forward method of an MHA projection with one that adds
    the LoRA correction. We store the original method on the object.
    """
    # MHA in PyTorch stores weights as in_proj_weight (Q+K+V concatenated)
    # and out_proj.weight. We patch at the MHA level using a wrapper approach.
    if not hasattr(mha, "_lora_patches"):
        mha._lora_patches = {}
    mha._lora_patches[proj] = lora

    # If not already patched, wrap the forward method
    if not hasattr(mha, "_original_forward"):
        mha._original_forward = mha.forward
        mha.forward = _make_lora_forward(mha)


def _unpatch_attn(mha: nn.MultiheadAttention) -> None:
    if hasattr(mha, "_original_forward"):
        mha.forward = mha._original_forward
        del mha._original_forward
    if hasattr(mha, "_lora_patches"):
        del mha._lora_patches


def _make_lora_forward(mha: nn.MultiheadAttention):
    """
    Returns a new forward that calls the original MHA forward and adds
    LoRA corrections to Q, K, V, and output projections.

    PyTorch's MHA doesn't expose Q/K/V projections individually as modules,
    so we apply LoRA to the combined in_proj_weight via a pre/post hook approach.

    Implementation: we use forward hooks on the MHA to intercept Q, K, V
    inputs just before they're projected, apply LoRA, then call original.

    Simplified approach used here: apply LoRA to the query input (most
    impactful projection) and output. Full Q/K/V/O LoRA requires custom
    attention kernel — deferred to Phase 2 optimization.
    """
    def lora_forward(query, key, value, *args, **kwargs):
        patches = mha._lora_patches

        # Apply LoRA to query projection input
        if "q" in patches:
            query = query + patches["q"].delta(query)
        if "k" in patches:
            key = key + patches["k"].delta(key)
        if "v" in patches:
            value = value + patches["v"].delta(value)

        attn_out, weights = mha._original_forward(query, key, value, *args, **kwargs)

        # Apply LoRA to output projection
        if "out" in patches:
            attn_out = attn_out + patches["out"].delta(attn_out)

        return attn_out, weights

    return lora_forward


# ---------------------------------------------------------------------------
# Registry: manages adapters for all games
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """
    Stores one GameAdapter per game. Handles save/load and swapping.

    Typical usage per training episode:
        registry.load(game_id, backbone)     # inject adapter
        ... run episode ...
        registry.unload(backbone)            # eject adapter
    """

    def __init__(self, backbone: CategoryExpertModel) -> None:
        self._backbone = backbone
        self._adapters: dict[GameID, GameAdapter] = {}
        self._active: GameID | None = None

    def get_or_create(self, game_id: GameID) -> GameAdapter:
        if game_id not in self._adapters:
            self._adapters[game_id] = GameAdapter(game_id)
        return self._adapters[game_id]

    def load(self, game_id: GameID) -> None:
        """Eject current adapter (if any) and inject the one for game_id."""
        if self._active is not None:
            self._adapters[self._active].eject(self._backbone)
        adapter = self.get_or_create(game_id)
        adapter.inject(self._backbone)
        self._active = game_id

    def unload(self) -> None:
        if self._active is not None:
            self._adapters[self._active].eject(self._backbone)
            self._active = None

    def active_adapter(self) -> GameAdapter | None:
        if self._active is None:
            return None
        return self._adapters[self._active]

    def trainable_parameters(self, game_id: GameID) -> list[nn.Parameter]:
        """Parameters to pass to the optimizer for a specific game."""
        return list(self.get_or_create(game_id).parameters())

    def save(self, game_id: GameID, path: str) -> None:
        adapter = self.get_or_create(game_id)
        torch.save(adapter.state_dict(), path)

    def load_weights(self, game_id: GameID, path: str) -> None:
        adapter = self.get_or_create(game_id)
        state = torch.load(path, map_location="cpu")
        adapter.load_state_dict(state)

    def registered_games(self) -> list[GameID]:
        return list(self._adapters.keys())
