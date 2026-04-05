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
import os
import re
import types
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
_reft_mask_debug_counts: dict[tuple[str, int, str, str], int] = {}


def _reft_mask_debug_enabled() -> bool:
    return os.environ.get("VLLM_REFT_DEBUG_MASK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _reft_mask_debug_limit() -> int:
    return int(os.environ.get("VLLM_REFT_DEBUG_LIMIT", "12"))

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

def _compute_position_mask(
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
) -> Optional[torch.Tensor]:
    """Return the per-token ReFT mask, or ``None`` for unconditional apply.

    All masking uses tensor ops only – no Python branching on live tensor
    values – so the full masking path is captured inside CUDA graphs.

    Args:
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
            return (token_idx < num_prefill_tokens).to(dtype)
        else:
            # Fallback: all tokens at position 0 → we are in a prefill pass.
            gate = (positions[0:1] == 0).to(dtype)
            return gate.expand(num_tokens)

    if position == "first":
        mask = (positions == 0).to(dtype)
        if num_prefill_tokens is not None:
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx < num_prefill_tokens).to(dtype)
            mask = mask * prefill_mask
        else:
            in_prefill = (positions[0:1] == 0).to(dtype)
            mask = mask * in_prefill
        return mask

    if position == "last":
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
            return is_last
        else:
            # Fallback tensor-only "last" mask.
            is_last = torch.zeros_like(positions, dtype=dtype)
            is_last[-1] = 1.0
            next_is_zero = (torch.roll(positions, -1) == 0).to(dtype)
            is_last = is_last + next_is_zero * (1.0 - is_last)
            in_prefill = (positions[0:1] == 0).to(dtype)
            return is_last * in_prefill

    # "all" or unknown position – apply unconditionally (guarded at call-site).
    return None


def _apply_position_mask(
    delta: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the ReFT position mask to *delta* and return ``(delta, mask)``."""
    mask = _compute_position_mask(positions, position, dtype, num_tokens, attn_metadata)
    if mask is None:
        return delta, None
    return delta * mask.unsqueeze(-1), mask


def _maybe_log_mask_debug(
    *,
    arch: str,
    layer_idx: int,
    position: str,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor],
    attn_metadata,
) -> None:
    """Log how many tokens receive a nonzero ReFT mask in eager mode."""
    if not _reft_mask_debug_enabled():
        return
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return

    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return

    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    if num_prefill_tokens is None:
        first_is_prefill = bool((positions[:1] == 0).all().item())
        prefill_tokens = num_tokens if first_is_prefill else 0
    else:
        prefill_tokens = int(num_prefill_tokens)
    decode_tokens = max(0, num_tokens - prefill_tokens)

    if mask is None:
        masked_tokens = num_tokens
    else:
        masked_tokens = int((mask.detach().float() > 0).sum().item())

    if prefill_tokens > 0 and decode_tokens > 0:
        phase = "mixed"
    elif prefill_tokens > 0:
        phase = "prefill"
    else:
        phase = "decode"

    key = (arch, layer_idx, position, phase)
    count = _reft_mask_debug_counts.get(key, 0)
    if count >= _reft_mask_debug_limit():
        return
    _reft_mask_debug_counts[key] = count + 1

    positions_head = positions[: min(8, num_tokens)].detach().cpu().tolist()
    logger.info(
        "[ReFT-vLLM mask debug] arch=%s layer=%d position=%s phase=%s "
        "masked_tokens=%d total_tokens=%d prefill_tokens=%d decode_tokens=%d "
        "positions_head=%s",
        arch,
        layer_idx,
        position,
        phase,
        masked_tokens,
        num_tokens,
        prefill_tokens,
        decode_tokens,
        positions_head,
    )


def _init_reft_debug_buffers(module: nn.Module, device: torch.device) -> None:
    """Initialize non-persistent buffers used for compiled-path mask stats."""
    if hasattr(module, "_reft_debug_total_tokens"):
        return

    def _zero() -> torch.Tensor:
        return torch.zeros((), dtype=torch.int64, device=device)

    def _zero_f() -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=device)

    module.register_buffer("_reft_debug_total_tokens", _zero(), persistent=False)
    module.register_buffer("_reft_debug_masked_tokens", _zero(), persistent=False)
    module.register_buffer("_reft_debug_prefill_tokens", _zero(), persistent=False)
    module.register_buffer("_reft_debug_decode_tokens", _zero(), persistent=False)
    module.register_buffer("_reft_debug_prefill_calls", _zero(), persistent=False)
    module.register_buffer("_reft_debug_decode_calls", _zero(), persistent=False)
    module.register_buffer("_reft_debug_mixed_calls", _zero(), persistent=False)
    module.register_buffer("_reft_debug_delta_l2_sum", _zero_f(), persistent=False)
    module.register_buffer("_reft_debug_delta_abs_sum", _zero_f(), persistent=False)
    module.register_buffer("_reft_debug_delta_abs_max", _zero_f(), persistent=False)
    module.register_buffer("_reft_debug_hidden_l2_sum", _zero_f(), persistent=False)


def _record_mask_debug_stats(
    module: nn.Module,
    *,
    delta: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: Optional[torch.Tensor],
    positions: torch.Tensor,
    attn_metadata,
) -> None:
    """Accumulate compiled-path-safe ReFT mask stats into layer buffers."""
    debug_enabled = getattr(module, "_reft_debug_enabled", False) or _reft_mask_debug_enabled()
    if not debug_enabled or not hasattr(module, "_reft_debug_total_tokens"):
        return

    num_tokens_int = positions.shape[0]
    if num_tokens_int == 0:
        return
    stats_device = module._reft_debug_total_tokens.device
    num_tokens = torch.tensor(num_tokens_int, dtype=torch.int64, device=stats_device)

    if mask is None:
        masked_tokens = num_tokens
    else:
        masked_tokens = (mask.to(device=stats_device) > 0).to(torch.int64).sum()

    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    if num_prefill_tokens is not None:
        prefill_tokens = torch.tensor(int(num_prefill_tokens), dtype=torch.int64, device=stats_device)
        prefill_tokens = torch.minimum(prefill_tokens, num_tokens)
    else:
        first_is_prefill = (positions[:1].to(device=stats_device) == 0).all()
        prefill_tokens = torch.where(first_is_prefill, num_tokens, torch.zeros_like(num_tokens))

    decode_tokens = num_tokens - prefill_tokens
    one = torch.ones((), dtype=torch.int64, device=stats_device)
    zero = torch.zeros((), dtype=torch.int64, device=stats_device)

    module._reft_debug_total_tokens.add_(num_tokens)
    module._reft_debug_masked_tokens.add_(masked_tokens)
    module._reft_debug_prefill_tokens.add_(prefill_tokens)
    module._reft_debug_decode_tokens.add_(decode_tokens)
    module._reft_debug_prefill_calls.add_(
        torch.where((prefill_tokens > 0) & (decode_tokens == 0), one, zero)
    )
    module._reft_debug_decode_calls.add_(
        torch.where((prefill_tokens == 0) & (decode_tokens > 0), one, zero)
    )
    module._reft_debug_mixed_calls.add_(
        torch.where((prefill_tokens > 0) & (decode_tokens > 0), one, zero)
    )

    delta_f = delta.detach().to(device=stats_device, dtype=torch.float32)
    hidden_f = hidden_states.detach().to(device=stats_device, dtype=torch.float32)
    module._reft_debug_delta_l2_sum.add_(delta_f.norm(dim=-1).sum())
    module._reft_debug_delta_abs_sum.add_(delta_f.abs().sum())
    module._reft_debug_delta_abs_max.copy_(
        torch.maximum(module._reft_debug_delta_abs_max, delta_f.abs().amax())
    )
    module._reft_debug_hidden_l2_sum.add_(hidden_f.norm(dim=-1).sum())


def _collect_layer_reft_debug_stats(layer: nn.Module) -> Optional[dict]:
    """Return serializable ReFT debug stats for one layer."""
    total_tokens = getattr(layer, "_reft_debug_total_tokens", None)
    if total_tokens is None:
        return None
    total = int(total_tokens.detach().cpu().item())

    return {
        "layer_idx": int(getattr(layer, "_reft_layer_idx", -1)),
        "has_adapter": getattr(layer, "_reft_adapter", None) is not None,
        "debug_enabled": bool(getattr(layer, "_reft_debug_enabled", False)),
        "has_debug_buffers": True,
        "position": getattr(layer, "_reft_position", None),
        "masked_tokens": int(layer._reft_debug_masked_tokens.detach().cpu().item()),
        "total_tokens": total,
        "prefill_tokens": int(layer._reft_debug_prefill_tokens.detach().cpu().item()),
        "decode_tokens": int(layer._reft_debug_decode_tokens.detach().cpu().item()),
        "prefill_calls": int(layer._reft_debug_prefill_calls.detach().cpu().item()),
        "decode_calls": int(layer._reft_debug_decode_calls.detach().cpu().item()),
        "mixed_calls": int(layer._reft_debug_mixed_calls.detach().cpu().item()),
        "delta_l2_sum": float(layer._reft_debug_delta_l2_sum.detach().cpu().item()),
        "delta_abs_sum": float(layer._reft_debug_delta_abs_sum.detach().cpu().item()),
        "delta_abs_max": float(layer._reft_debug_delta_abs_max.detach().cpu().item()),
        "hidden_l2_sum": float(layer._reft_debug_hidden_l2_sum.detach().cpu().item()),
    }


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
    debug_mask_enabled = bool(reft_spec.get("debug_mask", False))

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
                _init_reft_debug_buffers(self, dev)

                # Hidden from nn.Module: vLLM's weight loader would raise for
                # adapter params that don't exist in the checkpoint.
                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Qwen2 layer %d "
                    "(has_R_cache=%s, has_pinv_cache=%s)",
                    layer_idx,
                    hasattr(adapter_copy, "_R_cache"),
                    hasattr(adapter_copy, "_w2_pinv_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

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
            delta, mask = _apply_position_mask(
                delta, positions, reft_position,
                hidden_states.dtype, positions.shape[0], attn_metadata,
            )
            _maybe_log_mask_debug(
                arch="qwen2",
                layer_idx=self._reft_layer_idx,
                position=reft_position,
                positions=positions,
                mask=mask,
                attn_metadata=attn_metadata,
            )
            _record_mask_debug_stats(
                self,
                delta=delta,
                hidden_states=h_full,
                mask=mask,
                positions=positions,
                attn_metadata=attn_metadata,
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

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

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
    debug_mask_enabled = bool(reft_spec.get("debug_mask", False))

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
                _init_reft_debug_buffers(self, dev)

                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Llama layer %d "
                    "(has_R_cache=%s, has_pinv_cache=%s)",
                    layer_idx,
                    hasattr(adapter_copy, "_R_cache"),
                    hasattr(adapter_copy, "_w2_pinv_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

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
            delta, mask = _apply_position_mask(
                delta, positions, reft_position,
                hidden_states.dtype, positions.shape[0], attn_metadata,
            )
            _maybe_log_mask_debug(
                arch="llama",
                layer_idx=self._reft_layer_idx,
                position=reft_position,
                positions=positions,
                mask=mask,
                attn_metadata=attn_metadata,
            )
            _record_mask_debug_stats(
                self,
                delta=delta,
                hidden_states=h_full,
                mask=mask,
                positions=positions,
                attn_metadata=attn_metadata,
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

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTLlamaDecoderLayer.__name__ = "ReFTLlamaDecoderLayer"
    ReFTLlamaDecoderLayer.__qualname__ = "ReFTLlamaDecoderLayer"
    return ReFTLlamaDecoderLayer
