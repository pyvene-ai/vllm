"""vllm.reft.layer – CUDA-graph-compatible ReFT decoder layer factories.

Overview
--------
For GRPO training we need vLLM to apply the ReFT adapter delta during
inference (prefill) while staying on-policy with the HF training model.

This module provides two things:

  1. Graph-safe cache helpers (_install_R_cache, _install_pinv_cache) –
     identical logic to the ones in the repo's local vllm_config/, but now
     living inside the fork so callers don't need to import project internals.

  2. Layer factory functions (make_reft_qwen2_layer, make_reft_llama_layer) –
     return nn.Module *classes* (not instances) that subclass the standard
     vLLM decoder layer and run the adapter delta in their forward pass.

The factories are called by Qwen2ForCausalLM / LlamaForCausalLM when a
``_reft_spec`` is present (see vllm.reft.__init__ for the thread-local API).

CUDA-graph safety
-----------------
The forward logic uses only tensor ops; there is no Python branching on live
tensor values.  The adapter is stored as a plain Python attribute (bypassing
nn.Module) so vLLM's weight loader does not raise on unknown parameters.

Graph-safe caches
-----------------
  _R_cache      – precomputed Householder product (avoids 48 decompositions
                  per prefill for 24-layer models)
  _w2_pinv_cache – precomputed pseudo-inverse (avoids SVD inside graph)

Both are registered as nn.Buffers on the adapter instance so they stay on
the correct device after .to() calls.
"""

import copy
import logging
import math
import re
import types
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph-safe cache helpers (mirrors vllm_config/vllm_reft_layer.py)
# ---------------------------------------------------------------------------

def _install_R_cache(adapter: nn.Module) -> None:
    """Cache the Householder rotation matrix and patch _compute_delta.

    Avoids recomputing the full Householder product on every forward pass.
    Cache is stored in the same dtype as learned_source.weight (usually bf16)
    so that learned_source(h) - h @ R_cache == 0 exactly at initialisation,
    preventing vLLM log-probs from drifting away from the HF model at step 0.
    """
    R_f32 = adapter.rotate_layer.weight.detach().float()
    w_dtype = adapter.learned_source.weight.dtype
    adapter.register_buffer("_R_cache", R_f32.to(w_dtype))

    def _cached_compute_delta(self, hidden_states):
        source = self.learned_source(hidden_states) - hidden_states @ self._R_cache
        mixer = getattr(self, "mixer", None)
        if mixer is not None and not getattr(mixer, "needs_kv", False):
            mix_out, _ = mixer(source, None)
            source = source + mix_out
        delta = source @ self._R_cache.T
        if getattr(self, "scaled", True):
            delta = delta / math.sqrt(self.low_rank_dim)
        return delta

    adapter._compute_delta = types.MethodType(_cached_compute_delta, adapter)


def _install_pinv_cache(adapter: nn.Module) -> None:
    """Cache pinv(w2) and patch _compute_delta to avoid SVD inside graph.

    torch.linalg.pinv uses cuSolver which allocates memory dynamically and
    is not CUDA-graph-safe.  This replaces the runtime call with a buffer
    lookup and a refresh call after each optimizer step.
    """
    w2 = adapter.w2.detach().float()
    adapter.register_buffer("_w2_pinv_cache", torch.linalg.pinv(w2))

    def _cached_compute_delta(self, hidden_states):
        h = hidden_states.float()
        if hasattr(self, "_R_cache"):
            source = self.learned_source(h) - h @ self._R_cache
        else:
            source = self._compute_raw_source(h)
        delta = source @ self._w2_pinv_cache
        if getattr(self, "scaled", True):
            delta = delta / math.sqrt(self.low_rank_dim)
        return delta.to(hidden_states.dtype)

    adapter._compute_delta = types.MethodType(_cached_compute_delta, adapter)


def _refresh_adapter_caches(adapter: nn.Module) -> None:
    """Refresh _R_cache and _w2_pinv_cache after loading new adapter weights.

    Called by the ReFT decoder layer's load_reft_state() method after each
    optimizer step to keep the cached tensors in sync with the trained weights.
    """
    if hasattr(adapter, "_R_cache") and hasattr(adapter, "rotate_layer"):
        R = adapter.rotate_layer.weight.detach().float()
        cache_dtype = adapter._R_cache.dtype
        adapter._R_cache.data.copy_(R.to(adapter._R_cache.device).to(cache_dtype))

    if hasattr(adapter, "_w2_pinv_cache"):
        w2 = adapter.w2.detach().float()
        adapter._w2_pinv_cache.data.copy_(
            torch.linalg.pinv(w2).to(adapter._w2_pinv_cache.device)
        )


# ---------------------------------------------------------------------------
# Layer index extraction from vLLM prefix string
# ---------------------------------------------------------------------------

def _extract_layer_idx(prefix: str) -> Optional[int]:
    """Extract the integer layer index from a vLLM prefix like 'model.layers.7'."""
    if not prefix:
        return None
    m = re.search(r"\.layers\.(\d+)", prefix)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Position masking (CUDA-graph-safe)
# ---------------------------------------------------------------------------

def _apply_position_mask(
    delta: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
) -> torch.Tensor:
    """Apply the ReFT position mask to *delta*.

    All masking uses tensor ops only – no Python branching on live tensor
    values – so the full masking path is captured inside CUDA graphs.

    Args:
        delta:        (num_tokens, hidden_size) – unmasked ReFT delta.
        positions:    (num_tokens,) – vLLM position indices.
        position:     "prefill" | "first" | "last".
        dtype:        Hidden-state dtype used for mask scalars.
        num_tokens:   Number of tokens in this batch (tensor shape dim 0).
        attn_metadata: Optional attention metadata from ForwardContext.
    """
    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)

    if position == "prefill":
        if num_prefill_tokens is not None:
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx < num_prefill_tokens).to(dtype)
            return delta * prefill_mask.unsqueeze(-1)
        else:
            # Fallback: all tokens at position 0 → we are in a prefill pass.
            gate = (positions[0:1] == 0).to(dtype)
            return delta * gate.expand(num_tokens).unsqueeze(-1)

    elif position == "first":
        mask = (positions == 0).to(dtype)
        if num_prefill_tokens is not None:
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx < num_prefill_tokens).to(dtype)
            mask = mask * prefill_mask
        else:
            in_prefill = (positions[0:1] == 0).to(dtype)
            mask = mask * in_prefill
        return delta * mask.unsqueeze(-1)

    elif position == "last":
        if num_prefill_tokens is not None:
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx < num_prefill_tokens).to(dtype)
            next_prefill = torch.zeros_like(prefill_mask)
            next_prefill[:-1] = prefill_mask[1:]
            next_is_new_seq = torch.zeros_like(prefill_mask)
            next_is_new_seq[:-1] = (positions[1:] == 0).to(dtype)
            is_last = prefill_mask * ((1.0 - next_prefill) + next_is_new_seq)
            is_last = torch.clamp(is_last, max=1.0)
            is_last[-1] = prefill_mask[-1]
            return delta * is_last.unsqueeze(-1)
        else:
            # Fallback tensor-only "last" mask.
            is_last = torch.zeros_like(positions, dtype=dtype)
            is_last[-1] = 1.0
            next_is_zero = (torch.roll(positions, -1) == 0).to(dtype)
            is_last = is_last + next_is_zero * (1.0 - is_last)
            in_prefill = (positions[0:1] == 0).to(dtype)
            return delta * (is_last * in_prefill).unsqueeze(-1)

    # "all" or unknown position – apply unconditionally (guarded at call-site).
    return delta


# ---------------------------------------------------------------------------
# Qwen2 ReFT-aware decoder layer factory
# ---------------------------------------------------------------------------

def make_reft_qwen2_layer(reft_spec: dict) -> type:
    """Return a Qwen2DecoderLayer subclass that applies ReFT adapters.

    The returned class is drop-in for the *decoder_layer_type* argument of
    ``Qwen2Model``.  Each instance with a layer index in
    ``reft_spec["layer_indices"]`` gets its own deep-copied adapter; all other
    instances pass through unchanged.

    Args:
        reft_spec: dict with keys
            layer_indices  list[int]  – which layers get adapters
            position       str        – "prefill" | "first" | "last"
            sample_adapter nn.Module  – architecture template (weights synced
                                        later via load_reft_state)
    """
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
    from vllm.forward_context import get_forward_context

    layer_indices_set = frozenset(reft_spec["layer_indices"])
    position = reft_spec["position"]
    sample_adapter: nn.Module = reft_spec["sample_adapter"]

    class ReFTQwen2DecoderLayer(Qwen2DecoderLayer):
        """Qwen2DecoderLayer with optional ReFT adapter delta."""

        def __init__(self, config, cache_config=None, quant_config=None,
                     prefix=""):
            super().__init__(config=config, cache_config=cache_config,
                             quant_config=quant_config, prefix=prefix)

            layer_idx = _extract_layer_idx(prefix)
            if layer_idx is not None and layer_idx in layer_indices_set:
                try:
                    dev = next(self.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
                adapter_copy = copy.deepcopy(sample_adapter).to(dev)

                # Install CUDA-graph-safe caches.
                if (hasattr(adapter_copy, "rotate_layer")
                        and not hasattr(adapter_copy, "_R_cache")):
                    _install_R_cache(adapter_copy)
                if (hasattr(adapter_copy, "w2")
                        and not hasattr(adapter_copy, "_w2_pinv_cache")):
                    _install_pinv_cache(adapter_copy)

                # Hidden from nn.Module: vLLM's weight loader would raise for
                # adapter params that don't exist in the checkpoint.
                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Qwen2 layer %d "
                    "(has_R_cache=%s, has_pinv_cache=%s)",
                    layer_idx,
                    hasattr(adapter_copy, "_R_cache"),
                    hasattr(adapter_copy, "_w2_pinv_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)

            # Store position string as a normal attribute (not a parameter).
            object.__setattr__(self, "_reft_position", position)

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual,
        ):
            hidden_states, residual = super().forward(
                positions, hidden_states, residual)

            # Use normal attribute lookup here so TorchDynamo can trace the
            # forward path; explicit object.__getattribute__ is unsupported.
            adapter = self._reft_adapter
            if adapter is None:
                return hidden_states, residual

            h_full = hidden_states + residual
            delta = adapter._compute_delta(
                h_full.unsqueeze(0)).squeeze(0)  # (n, H)

            ctx = get_forward_context()
            attn_metadata = getattr(ctx, "attn_metadata", None)
            reft_position = self._reft_position
            delta = _apply_position_mask(
                delta, positions, reft_position,
                hidden_states.dtype, positions.shape[0], attn_metadata,
            )

            hidden_states = hidden_states + delta
            return hidden_states, residual

        def load_reft_state(self, state_dict: dict) -> None:
            """Update adapter weights and refresh graph-safe caches."""
            adapter = self._reft_adapter
            if adapter is None:
                return
            missing, unexpected = adapter.load_state_dict(
                state_dict, strict=False)
            allowed_missing = {"_R_cache", "_w2_pinv_cache"}
            unexpected_missing = [k for k in missing
                                   if k not in allowed_missing]
            if unexpected_missing:
                logger.warning(
                    "[ReFT-vLLM] Missing state keys in ReFT adapter sync: %s",
                    unexpected_missing)
            if unexpected:
                logger.warning(
                    "[ReFT-vLLM] Unexpected state keys in ReFT adapter sync: %s",
                    unexpected)
            _refresh_adapter_caches(adapter)

    ReFTQwen2DecoderLayer.__name__ = "ReFTQwen2DecoderLayer"
    ReFTQwen2DecoderLayer.__qualname__ = "ReFTQwen2DecoderLayer"
    return ReFTQwen2DecoderLayer


# ---------------------------------------------------------------------------
# Llama ReFT-aware decoder layer factory
# ---------------------------------------------------------------------------

def make_reft_llama_layer(reft_spec: dict) -> type:
    """Return a LlamaDecoderLayer subclass that applies ReFT adapters.

    Analogous to ``make_reft_qwen2_layer`` but for the Llama/Mistral/Mixtral
    architecture family, which passes the full ``vllm_config`` to the layer.
    """
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    from vllm.forward_context import get_forward_context

    layer_indices_set = frozenset(reft_spec["layer_indices"])
    position = reft_spec["position"]
    sample_adapter: nn.Module = reft_spec["sample_adapter"]

    class ReFTLlamaDecoderLayer(LlamaDecoderLayer):
        """LlamaDecoderLayer with optional ReFT adapter delta."""

        def __init__(self, vllm_config, prefix="", config=None):
            super().__init__(vllm_config=vllm_config, prefix=prefix,
                             config=config)

            layer_idx = _extract_layer_idx(prefix)
            if layer_idx is not None and layer_idx in layer_indices_set:
                try:
                    dev = next(self.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
                adapter_copy = copy.deepcopy(sample_adapter).to(dev)

                if (hasattr(adapter_copy, "rotate_layer")
                        and not hasattr(adapter_copy, "_R_cache")):
                    _install_R_cache(adapter_copy)
                if (hasattr(adapter_copy, "w2")
                        and not hasattr(adapter_copy, "_w2_pinv_cache")):
                    _install_pinv_cache(adapter_copy)

                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Llama layer %d "
                    "(has_R_cache=%s, has_pinv_cache=%s)",
                    layer_idx,
                    hasattr(adapter_copy, "_R_cache"),
                    hasattr(adapter_copy, "_w2_pinv_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)

            object.__setattr__(self, "_reft_position", position)

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual,
        ):
            hidden_states, residual = super().forward(
                positions, hidden_states, residual)

            # Use normal attribute lookup here so TorchDynamo can trace the
            # forward path; explicit object.__getattribute__ is unsupported.
            adapter = self._reft_adapter
            if adapter is None:
                return hidden_states, residual

            h_full = hidden_states + residual
            delta = adapter._compute_delta(
                h_full.unsqueeze(0)).squeeze(0)

            ctx = get_forward_context()
            attn_metadata = getattr(ctx, "attn_metadata", None)
            reft_position = self._reft_position
            delta = _apply_position_mask(
                delta, positions, reft_position,
                hidden_states.dtype, positions.shape[0], attn_metadata,
            )

            hidden_states = hidden_states + delta
            return hidden_states, residual

        def load_reft_state(self, state_dict: dict) -> None:
            """Update adapter weights and refresh graph-safe caches."""
            adapter = self._reft_adapter
            if adapter is None:
                return
            missing, unexpected = adapter.load_state_dict(
                state_dict, strict=False)
            allowed_missing = {"_R_cache", "_w2_pinv_cache"}
            unexpected_missing = [k for k in missing
                                   if k not in allowed_missing]
            if unexpected_missing:
                logger.warning(
                    "[ReFT-vLLM] Missing state keys in ReFT adapter sync: %s",
                    unexpected_missing)
            if unexpected:
                logger.warning(
                    "[ReFT-vLLM] Unexpected state keys in ReFT adapter sync: %s",
                    unexpected)
            _refresh_adapter_caches(adapter)

    ReFTLlamaDecoderLayer.__name__ = "ReFTLlamaDecoderLayer"
    ReFTLlamaDecoderLayer.__qualname__ = "ReFTLlamaDecoderLayer"
    return ReFTLlamaDecoderLayer
