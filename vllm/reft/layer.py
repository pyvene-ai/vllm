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
_reft_init_debug_count = 0


def _reft_mask_debug_enabled() -> bool:
    return os.environ.get("VLLM_REFT_DEBUG_MASK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _reft_mask_debug_limit() -> int:
    return int(os.environ.get("VLLM_REFT_DEBUG_LIMIT", "12"))


def _maybe_log_reft_layer_init(
    *,
    arch: str,
    layer_idx: int,
    attached: bool,
    debug_enabled: bool,
    position: str,
    adapter: Optional[nn.Module],
) -> None:
    """Emit a small construction-time log for ReFT-aware decoder layers."""
    global _reft_init_debug_count

    if not (debug_enabled or _reft_mask_debug_enabled()):
        return

    limit = int(os.environ.get("VLLM_REFT_DEBUG_INIT_LIMIT", "64"))
    if _reft_init_debug_count >= limit:
        return
    _reft_init_debug_count += 1

    adapter_type = type(adapter).__name__ if adapter is not None else None
    logger.info(
        "[ReFT-vLLM init debug] arch=%s layer=%d attached=%s debug_enabled=%s "
        "position=%s adapter_type=%s",
        arch,
        layer_idx,
        attached,
        debug_enabled,
        position,
        adapter_type,
    )

# ---------------------------------------------------------------------------
# Weight verification helper (prints to stderr for worker visibility)
# ---------------------------------------------------------------------------

def _log_weight_verification(
    arch: str,
    layer_idx: int,
    adapter: nn.Module,
    input_state_dict: dict,
) -> None:
    """Compare adapter weights with the input state dict after load.

    Prints to stderr so the output is visible even in V1 worker processes
    where Python logging may not be configured.
    """
    import sys
    adapter_sd = adapter.state_dict()
    # Only compare keys that exist in both
    common_keys = set(adapter_sd.keys()) & set(input_state_dict.keys())
    only_adapter = set(adapter_sd.keys()) - set(input_state_dict.keys())
    only_input = set(input_state_dict.keys()) - set(adapter_sd.keys())
    diffs = []
    for key in sorted(common_keys):
        a = adapter_sd[key].detach().float().cpu()
        b = input_state_dict[key].detach().float().cpu()
        if a.shape != b.shape:
            diffs.append(f"  {key}: SHAPE MISMATCH adapter={a.shape} input={b.shape}")
        else:
            max_diff = (a - b).abs().max().item()
            if max_diff > 1e-6:
                diffs.append(f"  {key}: MAX_DIFF={max_diff:.6e} (NOT LOADED?)")

    status = "OK" if not diffs else "MISMATCH"
    msg = (
        f"[ReFT weight verify] {arch} L{layer_idx} "
        f"adapter_type={type(adapter).__name__} "
        f"status={status} "
        f"common={len(common_keys)} "
        f"only_adapter={sorted(only_adapter)} "
        f"only_input={sorted(only_input)}"
    )
    print(msg, file=sys.stderr, flush=True)
    if diffs:
        for d in diffs:
            print(f"  {d}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Graph-safe cache helpers (mirrors vllm_config/vllm_reft_layer.py)
# ---------------------------------------------------------------------------

def _install_adapter_caches(adapter: nn.Module) -> None:
    """Install inference caches on an adapter via the standard protocol.

    Calls ``adapter.install_inference_caches()`` if the method exists.
    Each adapter class is responsible for defining what needs to be cached
    (e.g. ``_R_cache`` for the rotation matrix, ``_w2_pinv_cache`` for
    pseudo-inverses, ``_w2_ridge_cache`` for regularised solves, etc.).
    vLLM never needs to know about specific adapter formulas.
    """
    if hasattr(adapter, "install_inference_caches"):
        adapter.install_inference_caches()


def _refresh_adapter_caches(adapter: nn.Module) -> None:
    """Refresh inference caches after loading new adapter weights.

    Calls ``adapter.refresh_inference_caches()`` if the method exists.
    Called by the ReFT decoder layer's ``load_reft_state()`` after each
    weight sync so cached tensors stay in sync with the trained weights.
    """
    if hasattr(adapter, "refresh_inference_caches"):
        adapter.refresh_inference_caches()


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
        # HF BaseAdapter.forward() skips decode tokens (seq_len == 1) for
        # non-"all" positions.  Match that: apply only to prefill tokens.
        #
        # vLLM V1 reorders batches so decode tokens come FIRST and prefill
        # tokens come LAST.  num_prefill_tokens counts the tokens at the
        # END of the batch, so the prefill region is
        # [num_tokens - num_prefill_tokens, num_tokens).
        if num_prefill_tokens is not None:
            # num_decode_tokens = num_tokens - num_prefill_tokens
            token_idx = torch.arange(num_tokens, device=positions.device)
            return (token_idx < num_prefill_tokens).to(dtype)
        else:
            # Fallback: all tokens at position 0 → we are in a prefill pass.
            gate = (positions[0:1] == 0).to(dtype)
            return gate.expand(num_tokens)

    if position == "first":
        mask = (positions == 0).to(dtype)
        if num_prefill_tokens is not None:
            num_decode_tokens = num_tokens - num_prefill_tokens
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx >= num_decode_tokens).to(dtype)
            mask = mask * prefill_mask
        else:
            in_prefill = (positions[0:1] == 0).to(dtype)
            mask = mask * in_prefill
        return mask

    if position == "last":
        if num_prefill_tokens is not None:
            num_decode_tokens = num_tokens - num_prefill_tokens
            token_idx = torch.arange(num_tokens, device=positions.device)
            prefill_mask = (token_idx >= num_decode_tokens).to(dtype)
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


_CAPTURE_MAX_TOKENS = 2048  # pre-allocated buffer size


def _init_reft_capture_buffers(module: nn.Module, hidden_size: int,
                               device: torch.device) -> None:
    """Pre-allocate buffers for per-token h_full/delta capture.

    Buffers are (max_tokens, hidden_size) — we always write to the full
    buffer and track the actual token count separately.  All operations
    are pure tensor ops, fully compatible with TorchDynamo/CUDA graphs.
    """
    if hasattr(module, "_reft_cap_h"):
        return
    M = _CAPTURE_MAX_TOKENS
    H = hidden_size
    # Full per-token captures
    module.register_buffer(
        "_reft_cap_h", torch.zeros(M, H, dtype=torch.float32,
                                   device=device), persistent=False)
    module.register_buffer(
        "_reft_cap_delta", torch.zeros(M, H, dtype=torch.float32,
                                       device=device), persistent=False)
    # Per-token norms (1-D, one per token position)
    module.register_buffer(
        "_reft_cap_h_norms", torch.zeros(M, dtype=torch.float32,
                                         device=device), persistent=False)
    module.register_buffer(
        "_reft_cap_delta_norms", torch.zeros(M, dtype=torch.float32,
                                             device=device), persistent=False)
    # Scalar aggregates
    module.register_buffer(
        "_reft_cap_num_tokens", torch.zeros((), dtype=torch.int64,
                                            device=device), persistent=False)
    module.register_buffer(
        "_reft_cap_delta_abs_max", torch.zeros((), dtype=torch.float32,
                                               device=device), persistent=False)
    module.register_buffer(
        "_reft_cap_h_mean_norm", torch.zeros((), dtype=torch.float32,
                                             device=device), persistent=False)
    module.register_buffer(
        "_reft_cap_delta_mean_norm", torch.zeros((), dtype=torch.float32,
                                                 device=device), persistent=False)


def _capture_last_token(
    module: nn.Module,
    *,
    h_full: torch.Tensor,
    delta: torch.Tensor,
    mask: Optional[torch.Tensor],  # noqa: ARG001 — kept for call-site compat
) -> None:
    """Capture all tokens' h_full/delta into pre-allocated buffers.

    Pure tensor ops only — no data-dependent indexing, no Python scalars
    from tensor values. Fully compatible with TorchDynamo and CUDA graphs.

    The buffers are fixed-size (CAPTURE_MAX_TOKENS x hidden_size).
    We write min(num_tokens, max) into the buffer and zero the rest.
    """
    if not hasattr(module, "_reft_cap_h"):
        return

    M = _CAPTURE_MAX_TOKENS
    n = h_full.shape[0]  # Python int from .shape — Dynamo-safe
    dev = module._reft_cap_h.device
    write_n = min(n, M)

    h_f = h_full[:write_n].detach().to(device=dev, dtype=torch.float32)
    d_f = delta[:write_n].detach().to(device=dev, dtype=torch.float32)

    # Zero out then write — avoids dynamic slicing on the right side
    module._reft_cap_h.zero_()
    module._reft_cap_delta.zero_()
    module._reft_cap_h_norms.zero_()
    module._reft_cap_delta_norms.zero_()

    module._reft_cap_h[:write_n].copy_(h_f)
    module._reft_cap_delta[:write_n].copy_(d_f)
    module._reft_cap_h_norms[:write_n].copy_(h_f.norm(dim=-1))
    module._reft_cap_delta_norms[:write_n].copy_(d_f.norm(dim=-1))

    # Store token count: zero then add n as a tensor to avoid SymInt in .fill_()
    module._reft_cap_num_tokens.zero_()
    module._reft_cap_num_tokens.add_(h_f.shape[0])
    module._reft_cap_delta_abs_max.copy_(d_f.abs().amax())
    module._reft_cap_h_mean_norm.copy_(h_f.norm(dim=-1).mean())
    module._reft_cap_delta_mean_norm.copy_(d_f.norm(dim=-1).mean())


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

    stats = {
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

    # Include per-token capture data if available
    cap_n = getattr(layer, "_reft_cap_num_tokens", None)
    if cap_n is not None:
        n = int(cap_n.item())
        if n > 0:
            stats["cap_num_tokens"] = n
            # Full per-token h_full and delta tensors (truncated to actual token count)
            stats["cap_h"] = layer._reft_cap_h[:n].detach().cpu().tolist()
            stats["cap_delta"] = layer._reft_cap_delta[:n].detach().cpu().tolist()
            stats["cap_h_norms"] = layer._reft_cap_h_norms[:n].detach().cpu().tolist()
            stats["cap_delta_norms"] = layer._reft_cap_delta_norms[:n].detach().cpu().tolist()
            stats["cap_delta_abs_max"] = float(layer._reft_cap_delta_abs_max.item())
            stats["cap_h_mean_norm"] = float(layer._reft_cap_h_mean_norm.item())
            stats["cap_delta_mean_norm"] = float(layer._reft_cap_delta_mean_norm.item())

    return stats


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

                # Install CUDA-graph-safe caches via the adapter's own protocol.
                _install_adapter_caches(adapter_copy)
                _init_reft_debug_buffers(self, dev)
                hidden_size = getattr(config, "hidden_size", 896)
                _init_reft_capture_buffers(self, hidden_size, dev)

                # Hidden from nn.Module: vLLM's weight loader would raise for
                # adapter params that don't exist in the checkpoint.
                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Qwen2 layer %d "
                    "(adapter_type=%s, has_R_cache=%s)",
                    layer_idx,
                    type(adapter_copy).__name__,
                    hasattr(adapter_copy, "_R_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

            # Store position string as a normal attribute (not a parameter).
            object.__setattr__(self, "_reft_position", position)
            _maybe_log_reft_layer_init(
                arch="qwen2",
                layer_idx=getattr(self, "_reft_layer_idx", -1),
                attached=self._reft_adapter is not None,
                debug_enabled=bool(getattr(self, "_reft_debug_enabled", False)),
                position=position,
                adapter=self._reft_adapter,
            )

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
            _capture_last_token(
                self, h_full=h_full, delta=delta, mask=mask,
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
            allowed_missing = {
                "_R_cache", "_w2_pinv_cache", "_w2_ridge_cache",
            }
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
            _log_weight_verification(
                "qwen2", self._reft_layer_idx,
                adapter, state_dict,
            )

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

                _install_adapter_caches(adapter_copy)
                _init_reft_debug_buffers(self, dev)
                llama_config = getattr(vllm_config, "model_config", vllm_config)
                hf_config = getattr(llama_config, "hf_config",
                                    getattr(llama_config, "config", config))
                hidden_size = getattr(hf_config, "hidden_size", 4096)
                _init_reft_capture_buffers(self, hidden_size, dev)

                object.__setattr__(self, "_reft_adapter", adapter_copy)
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Llama layer %d "
                    "(adapter_type=%s, has_R_cache=%s)",
                    layer_idx,
                    type(adapter_copy).__name__,
                    hasattr(adapter_copy, "_R_cache"),
                )
            else:
                object.__setattr__(self, "_reft_adapter", None)
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

            object.__setattr__(self, "_reft_position", position)
            _maybe_log_reft_layer_init(
                arch="llama",
                layer_idx=getattr(self, "_reft_layer_idx", -1),
                attached=self._reft_adapter is not None,
                debug_enabled=bool(getattr(self, "_reft_debug_enabled", False)),
                position=position,
                adapter=self._reft_adapter,
            )

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
            _capture_last_token(
                self, h_full=h_full, delta=delta, mask=mask,
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
            allowed_missing = {
                "_R_cache", "_w2_pinv_cache", "_w2_ridge_cache",
            }
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
            _log_weight_verification(
                "llama", self._reft_layer_idx,
                adapter, state_dict,
            )

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTLlamaDecoderLayer.__name__ = "ReFTLlamaDecoderLayer"
    ReFTLlamaDecoderLayer.__qualname__ = "ReFTLlamaDecoderLayer"
    return ReFTLlamaDecoderLayer
