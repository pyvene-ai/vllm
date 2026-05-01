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
_reft_nan_check: Optional[bool] = None


def _nan_check_enabled() -> bool:
    """Return True if VLLM_REFT_NAN_CHECK=1 is set (cached after first call)."""
    global _reft_nan_check
    if _reft_nan_check is None:
        _reft_nan_check = os.environ.get(
            "VLLM_REFT_NAN_CHECK", "").lower() in {"1", "true", "yes"}
    return _reft_nan_check


def _check_nan(tensor: torch.Tensor, label: str, layer_idx: int) -> bool:
    """If tensor has NaN/Inf, log details and return True.

    Only called from the custom op implementation which runs eagerly
    (outside Dynamo / CUDA graphs), so .item() is safe.
    """
    if not _nan_check_enabled():
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        logger.error(
            "[ReFT NaN check] layer=%d label=%s has_nan=%s has_inf=%s "
            "num_nan=%d num_inf=%d total=%d shape=%s dtype=%s "
            "abs_max=%.6g abs_mean=%.6g",
            layer_idx, label, has_nan, has_inf,
            num_nan, num_inf, total, list(tensor.shape), tensor.dtype,
            tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].abs().max().item()
            if (total - num_nan - num_inf) > 0 else float('nan'),
            tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].abs().mean().item()
            if (total - num_nan - num_inf) > 0 else float('nan'),
        )
        return True
    return False


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

# ---------------------------------------------------------------------------
# Graph-safe cache helpers (mirrors vllm_config/vllm_reft_layer.py)
# ---------------------------------------------------------------------------

def _install_adapter_caches(adapter: nn.Module,
                            model_dtype: torch.dtype = torch.bfloat16) -> None:
    """Install inference caches on an adapter via the standard protocol.

    Calls ``adapter.install_inference_caches()`` if the method exists.
    Each adapter class is responsible for defining what needs to be cached
    (e.g. ``_R_cache`` for the rotation matrix, ``_w2_pinv_cache`` for
    pseudo-inverses, ``_w2_ridge_cache`` for regularised solves, etc.).
    vLLM never needs to know about specific adapter formulas.
    """
    if hasattr(adapter, "install_inference_caches"):
        adapter.install_inference_caches(model_dtype=model_dtype)


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

def _resolve_attn_metadata(attn_metadata):
    """Resolve attn_metadata to a single object.

    In vLLM V1, ForwardContext.attn_metadata is a dict mapping layer names
    to per-layer metadata objects.  All layers within a KV-cache group share
    the same metadata, so we just grab the first value.
    """
    if isinstance(attn_metadata, dict):
        for v in attn_metadata.values():
            return v
    return attn_metadata


def _get_prefill_info(attn_metadata) -> tuple[
    Optional[int], int, int, Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """Extract prefill/decode split info from any V1 attention metadata.

    Returns (num_prefill_tokens, num_decodes, num_prefills,
             query_start_loc, seq_lens).

    Works with both FlashInfer (has explicit num_prefill_tokens) and
    Flash Attention (only has query_start_loc, need to derive the split).
    """
    # --- Try explicit fields first (FlashInfer) ---
    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    num_decodes = getattr(attn_metadata, "num_decodes", None)
    num_prefills = getattr(attn_metadata, "num_prefills", None)
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)

    if num_prefill_tokens is not None:
        # FlashInfer or similar backend with explicit split info
        if num_decodes is None:
            num_decodes = 0
        if num_prefills is None:
            num_prefills = 0
        return (int(num_prefill_tokens), int(num_decodes), int(num_prefills),
                query_start_loc, seq_lens)

    # --- Derive from query_start_loc (Flash Attention and others) ---
    if query_start_loc is not None and query_start_loc.numel() > 1:
        # query_start_loc: (num_reqs + 1,) cumulative query token counts.
        # In V1, decode requests (query_len=1) come FIRST, then prefill
        # requests (query_len>1).
        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # (num_reqs,)
        is_prefill = (query_lens > 1)
        num_reqs = query_lens.numel()

        # Since decodes come first and prefills last, find the split point.
        # The first prefill request is where query_len > 1 first appears.
        if is_prefill.any():
            # All decode requests have query_len=1 and come first.
            first_prefill_idx = int(is_prefill.to(torch.int32).argmax().item())
            n_decodes = first_prefill_idx
            n_prefills = num_reqs - n_decodes
            n_prefill_tokens = int(
                (query_start_loc[-1] - query_start_loc[n_decodes]).item())
        else:
            # All decode, no prefill
            n_decodes = num_reqs
            n_prefills = 0
            n_prefill_tokens = 0

        return (n_prefill_tokens, n_decodes, n_prefills,
                query_start_loc, seq_lens)

    # No usable metadata
    return (None, 0, 0, None, seq_lens)


def _prefill_request_indices(
    num_prefill_tokens: int,
    num_decodes: int,
    num_prefills: int,
    query_start_loc: Optional[torch.Tensor],
    prefill_positions: torch.Tensor,
) -> torch.Tensor:
    """Map each prefill token to its request index (0-based within prefills).

    Prefers ``query_start_loc`` (from attn_metadata) for an exact answer.
    Falls back to detecting position resets when metadata is unavailable.
    """
    if num_prefill_tokens <= 1:
        return torch.zeros(num_prefill_tokens, device=prefill_positions.device,
                           dtype=torch.long)

    # --- Preferred path: use query_start_loc from metadata ---
    if query_start_loc is not None and num_prefills > 0:
        # query_start_loc covers all requests (decode + prefill), shape (N+1,).
        # Prefill requests start at index num_decodes.
        # The prefill-region offsets are relative to the start of prefill tokens.
        prefill_qsl = query_start_loc[num_decodes:] - query_start_loc[num_decodes]
        prefill_tok_idx = torch.arange(
            num_prefill_tokens, device=prefill_positions.device)
        req_idx = torch.searchsorted(
            prefill_qsl[1:], prefill_tok_idx, right=True)
        return req_idx.clamp(max=num_prefills - 1)

    # --- Fallback: detect request boundaries from position resets ---
    is_boundary = torch.zeros(num_prefill_tokens,
                              device=prefill_positions.device, dtype=torch.long)
    is_boundary[1:] = (
        prefill_positions[1:] <= prefill_positions[:-1]).long()
    return is_boundary.cumsum(0)


def _compute_position_mask(
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
) -> Optional[torch.Tensor]:
    """Return the per-token ReFT mask, or ``None`` for unconditional apply.

    Handles both vLLM V0 (single metadata object) and V1 (dict of per-layer
    metadata).  Works with any V1 attention backend (FlashInfer, Flash
    Attention, etc.) by deriving the prefill/decode split from
    ``query_start_loc`` when explicit fields are absent.

    Uses ``seq_lens`` to correctly identify the true first/last prefill
    token under chunked prefill.
    """
    attn_metadata = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, num_decodes, num_prefills,
     query_start_loc, seq_lens) = _get_prefill_info(attn_metadata)

    if position == "prefill":
        if num_prefill_tokens is not None:
            num_decode_tokens = num_tokens - num_prefill_tokens
            token_idx = torch.arange(num_tokens, device=positions.device)
            return (token_idx >= num_decode_tokens).to(dtype)
        else:
            gate = (positions[0:1] == 0).to(dtype)
            return gate.expand(num_tokens)

    if position == "first":
        if num_prefill_tokens is not None:
            num_decode_tokens = num_tokens - num_prefill_tokens
            # position == 0 only appears in the very first chunk of a
            # request, so a simple check is sufficient — no seq_lens needed.
            full_mask = torch.zeros(num_tokens, device=positions.device,
                                    dtype=dtype)
            if num_prefill_tokens > 0:
                prefill_positions = positions[num_decode_tokens:]
                full_mask[num_decode_tokens:] = (prefill_positions == 0).to(dtype)
            return full_mask
        else:
            in_prefill = (positions[0:1] == 0).to(dtype)
            return (positions == 0).to(dtype) * in_prefill

    if position == "last":
        if num_prefill_tokens is not None:
            num_decode_tokens = num_tokens - num_prefill_tokens
            full_mask = torch.zeros(num_tokens, device=positions.device,
                                    dtype=dtype)
            if num_prefill_tokens == 0:
                return full_mask

            prefill_positions = positions[num_decode_tokens:]  # (P,)

            # --- Identify the last token of each request's query span ---
            req_idx = _prefill_request_indices(
                num_prefill_tokens, num_decodes, num_prefills,
                query_start_loc, prefill_positions)

            # last-in-query-span: token where req_idx changes or final token
            is_last_in_span = torch.zeros(num_prefill_tokens,
                                          device=positions.device, dtype=dtype)
            if num_prefill_tokens > 1:
                is_last_in_span[:-1] = (
                    req_idx[1:] != req_idx[:-1]).to(dtype)
            is_last_in_span[-1] = 1.0

            # --- Filter to only true last prefill tokens ---
            if seq_lens is not None:
                # seq_lens is ordered [decode_reqs..., prefill_reqs...]
                prefill_seq_lens = seq_lens[num_decodes:]
                expected_last_pos = prefill_seq_lens[req_idx] - 1
                is_true_last = (
                    prefill_positions == expected_last_pos).to(dtype)
                full_mask[num_decode_tokens:] = (
                    is_last_in_span * is_true_last)
            else:
                # No seq_lens available — fall back to last-in-span only.
                full_mask[num_decode_tokens:] = is_last_in_span

            return full_mask
        else:
            # Fallback tensor-only "last" mask (no prefill/decode split info).
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
    return (delta * mask.unsqueeze(-1)).contiguous(), mask


# ---------------------------------------------------------------------------
# Adapter registry & custom op (Dynamo splitting op)
# ---------------------------------------------------------------------------
# The custom op wraps both adapter._compute_delta() AND position masking
# in a single cudagraph_unsafe boundary.  This prevents CUDA graph capture
# from baking in stale adapter weights — after sync_weights() updates the
# adapter parameters, the custom op always uses the current weights.

_REFT_ADAPTER_REGISTRY: dict[int, nn.Module] = {}


def _reft_apply_adapter_op(
    h_full: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    num_tokens: int,
    layer_idx: int,
) -> torch.Tensor:
    """Compute adapter delta and apply position mask (runs eagerly)."""
    # CUDA graph capture is decode-only; ReFT is a no-op for decode tokens.
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros_like(h_full)

    adapter = _REFT_ADAPTER_REGISTRY.get(layer_idx)
    if adapter is None:
        return torch.zeros_like(h_full)

    _check_nan(h_full, "h_full_input", layer_idx)

    from vllm.forward_context import get_forward_context
    attn_metadata = get_forward_context().attn_metadata
    resolved = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, num_decodes, num_prefills,
     query_start_loc, seq_lens) = _get_prefill_info(resolved)

    # Pure decode batch: skip adapter entirely.
    if (position != "all"
            and num_prefill_tokens is not None
            and num_prefill_tokens == 0):
        return torch.zeros_like(h_full)

    # Mixed batch: only run adapter on prefill tokens (at end of batch).
    if (position != "all"
            and num_prefill_tokens is not None
            and 0 < num_prefill_tokens < num_tokens):
        num_decode_tokens = num_tokens - num_prefill_tokens
        h_prefill = h_full[num_decode_tokens:]
        _check_nan(h_prefill, "h_prefill_input", layer_idx)
        delta_prefill = adapter._compute_delta(
            h_prefill.unsqueeze(0)).squeeze(0)
        _check_nan(delta_prefill, "delta_after_compute_delta(mixed)", layer_idx)

        if position == "prefill":
            delta = torch.zeros_like(h_full)
            delta[num_decode_tokens:] = delta_prefill
            return delta

        # first/last: apply mask within the prefill region.
        mask = _compute_position_mask(
            positions, position, h_full.dtype, num_tokens, attn_metadata)
        if mask is not None:
            delta_prefill = (
                delta_prefill * mask[num_decode_tokens:].unsqueeze(-1)
            ).contiguous()
        delta = torch.zeros_like(h_full)
        delta[num_decode_tokens:] = delta_prefill
        _check_nan(delta, "delta_after_mask(mixed)", layer_idx)
        return delta

    # Full batch (all prefill, or position="all").
    delta = adapter._compute_delta(h_full.unsqueeze(0)).squeeze(0)
    _check_nan(delta, "delta_after_compute_delta(full)", layer_idx)
    mask = _compute_position_mask(
        positions, position, h_full.dtype, num_tokens, attn_metadata)
    if mask is not None:
        delta = (delta * mask.unsqueeze(-1)).contiguous()
    _check_nan(delta, "delta_final(full)", layer_idx)
    return delta


def _reft_apply_adapter_fake(
    h_full: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    num_tokens: int,
    layer_idx: int,
) -> torch.Tensor:
    return torch.empty_like(h_full).contiguous()


# NOTE: The reft_apply_adapter custom op was removed.  Position masking
# now uses a pre-computed buffer updated by the model runner before each
# forward pass.  The adapter's _compute_delta runs inline in the compiled
# forward — no splitting ops, no graph breaks.
_REFT_CUSTOM_OP_AVAILABLE = False


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

    resolved_meta = _resolve_attn_metadata(attn_metadata)
    npt, _, _, _, _ = _get_prefill_info(resolved_meta)
    if npt is None:
        first_is_prefill = bool((positions[:1] == 0).all().item())
        prefill_tokens = num_tokens if first_is_prefill else 0
    else:
        prefill_tokens = int(npt)
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
_REFT_MASK_FALLBACK_SIZE = 131072  # fallback if model runner doesn't init


def _init_reft_position_mask_buffer(module: nn.Module,
                                    max_tokens: int,
                                    device: torch.device) -> None:
    """Pre-allocate the position mask buffer used by the compiled forward.

    The buffer lives at a fixed address so CUDA graph replays read updated
    values after .copy_() / .fill_() calls.  The model runner updates it
    before each forward pass.
    """
    if hasattr(module, "_reft_position_mask"):
        return
    module.register_buffer(
        "_reft_position_mask",
        torch.zeros(max_tokens, dtype=torch.float32, device=device),
        persistent=False,
    )


def update_reft_position_masks(
    reft_layers: list[nn.Module],
    positions: torch.Tensor,
    attn_metadata,
    num_tokens: int,
) -> None:
    """Compute and store ReFT position masks on each layer's buffer.

    Called from the model runner (not compiled) before each model forward.
    Uses the existing ``_compute_position_mask`` logic which handles all
    attention backends and chunked prefill correctly.

    Groups layers by position so each unique mask is computed only once
    (avoids 23 redundant GPU→CPU syncs for a 24-layer model).
    """
    if not reft_layers:
        return

    # Group layers by position string.
    by_position: dict[str, list[nn.Module]] = {}
    for layer in reft_layers:
        pos = getattr(layer, "_reft_position", "prefill")
        by_position.setdefault(pos, []).append(layer)

    for pos, layers in by_position.items():
        if pos in ("all", "all_tokens"):
            mask = None
        else:
            mask = _compute_position_mask(
                positions, pos, torch.float32, num_tokens, attn_metadata)

        for layer in layers:
            buf = getattr(layer, "_reft_position_mask", None)
            if buf is None:
                continue
            if mask is None:
                buf[:num_tokens].fill_(1.0)
            else:
                buf[:num_tokens].copy_(mask)
            # Zero the tail so stale data from a larger prior batch
            # doesn't leak if something indexes past num_tokens.
            buf[num_tokens:].zero_()


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
    write_n = min(n, M)

    # Zero out then copy with implicit dtype cast — avoids allocating
    # temporary float32 tensors which break CUDA graph capture.
    module._reft_cap_h.zero_()
    module._reft_cap_delta.zero_()
    module._reft_cap_h_norms.zero_()
    module._reft_cap_delta_norms.zero_()

    module._reft_cap_h[:write_n].copy_(h_full[:write_n].detach())
    module._reft_cap_delta[:write_n].copy_(delta[:write_n].detach())
    module._reft_cap_h_norms[:write_n].copy_(
        module._reft_cap_h[:write_n].norm(dim=-1))
    module._reft_cap_delta_norms[:write_n].copy_(
        module._reft_cap_delta[:write_n].norm(dim=-1))

    # Store token count: zero then add n as a tensor to avoid SymInt in .fill_()
    module._reft_cap_num_tokens.zero_()
    module._reft_cap_num_tokens.add_(write_n)
    module._reft_cap_delta_abs_max.copy_(
        module._reft_cap_delta[:write_n].abs().amax())
    module._reft_cap_h_mean_norm.copy_(
        module._reft_cap_h[:write_n].norm(dim=-1).mean())
    module._reft_cap_delta_mean_norm.copy_(
        module._reft_cap_delta[:write_n].norm(dim=-1).mean())


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

    resolved_meta = _resolve_attn_metadata(attn_metadata)
    npt, _, _, _, _ = _get_prefill_info(resolved_meta)
    if npt is not None:
        prefill_tokens = torch.tensor(int(npt), dtype=torch.int64, device=stats_device)
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
        "has_adapter": getattr(layer, "reft_adapter", None) is not None,
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

# ---------------------------------------------------------------------------
# Shared multi-adapter forward logic
# ---------------------------------------------------------------------------

def _prepare_adapter(source: nn.Module, dev: torch.device,
                     model_dtype: torch.dtype) -> nn.Module:
    """Deep-copy an adapter, cast linears to model dtype, install caches."""
    adapter_copy = copy.deepcopy(source).to(dev)
    for child in adapter_copy.modules():
        if isinstance(child, nn.Linear):
            child.to(dtype=model_dtype)
    _install_adapter_caches(adapter_copy, model_dtype=model_dtype)
    return adapter_copy


def _init_multi_reft_state(layer: nn.Module, dev: torch.device,
                           hidden_size: int) -> None:
    """Initialise the multi-adapter bookkeeping on a decoder layer.

    Sets up:
      - ``reft_adapters``  (nn.ModuleDict) keyed by ``str(reft_int_id)``
      - ``_reft_adapter_positions`` (dict) adapter_id -> position string
      - ``_reft_combined_masks`` (dict) adapter_id -> pre-allocated mask buffer
    """
    layer.reft_adapters = nn.ModuleDict()
    layer._reft_adapter_positions: dict[int, str] = {}
    layer._reft_combined_masks: dict[int, torch.Tensor] = {}
    _init_reft_debug_buffers(layer, dev)
    _init_reft_capture_buffers(layer, hidden_size, dev)


def _add_adapter_to_layer(layer: nn.Module, reft_int_id: int,
                          adapter: nn.Module, position: str,
                          dev: torch.device) -> None:
    """Register an adapter on a decoder layer (in-place)."""
    key = str(reft_int_id)
    layer.reft_adapters[key] = adapter
    layer._reft_adapter_positions[reft_int_id] = position
    # Pre-allocate combined mask buffer
    layer._reft_combined_masks[reft_int_id] = torch.zeros(
        _REFT_MASK_FALLBACK_SIZE, dtype=torch.float32, device=dev)
    # Backward compat: keep reft_adapter pointing to first loaded adapter
    if not hasattr(layer, "reft_adapter") or layer.reft_adapter is None:
        layer.reft_adapter = adapter


def _multi_reft_forward(
    layer_self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    *,
    super_forward,
):
    """Shared multi-adapter forward pass for Qwen2 and Llama layers.

    Only iterates adapters that are active in the current batch (set by
    ``update_multi_reft_position_masks`` before each forward).  Inactive
    adapters are skipped entirely — no ``_compute_delta``, no masking.

    Pure-decode optimisation: if ``_reft_all_masks_zero`` is set (e.g.
    decode-only batch with prefill adapters), skip everything.
    """
    hidden_states, residual = super_forward(positions, hidden_states, residual)

    if not hasattr(layer_self, "reft_adapters") or len(layer_self.reft_adapters) == 0:
        return hidden_states, residual

    # Skip when all masks are zero (pure decode with non-"all" adapters).
    if getattr(layer_self, "_reft_all_masks_zero", False):
        return hidden_states, residual

    # Only iterate adapters that have at least one token in this batch.
    # _reft_active_ids is a Python set computed by
    # update_multi_reft_position_masks (outside compilation, no GPU sync).
    active_ids = getattr(layer_self, "_reft_active_ids", None)

    h_full = hidden_states + residual
    delta = torch.zeros_like(hidden_states)

    for str_id, adapter in layer_self.reft_adapters.items():
        int_id = int(str_id)
        if active_ids is not None and int_id not in active_ids:
            continue
        adapter_delta = adapter._compute_delta(
            h_full.unsqueeze(0)).squeeze(0)
        mask_buf = layer_self._reft_combined_masks.get(int_id)
        if mask_buf is not None:
            N = adapter_delta.shape[0]
            adapter_delta = adapter_delta * mask_buf[:N].unsqueeze(-1).to(
                adapter_delta.dtype)
        delta = delta + adapter_delta

    hidden_states = delta
    residual = h_full
    return hidden_states, residual


def update_multi_reft_position_masks(
    reft_layers: list[nn.Module],
    token_reft_ids: torch.Tensor,
    positions: torch.Tensor,
    attn_metadata,
    num_tokens: int,
) -> None:
    """Pre-compute combined masks for all adapters on all ReFT layers.

    For each adapter on each layer, the combined mask is:
        (token belongs to this adapter) AND (position satisfies adapter's mode)

    Called from the model runner before each forward pass.
    """
    if not reft_layers:
        return

    # Detect pure-decode batch from metadata (no GPU sync needed).
    # If no prefill tokens and all adapters use non-"all" positions,
    # skip mask computation and adapter forward entirely.
    resolved_meta = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, _, _, _, _) = _get_prefill_info(resolved_meta)
    is_pure_decode = (num_prefill_tokens is not None
                      and num_prefill_tokens == 0)

    if is_pure_decode:
        # Check if any adapter uses position="all" (needs decode-time compute)
        any_all_position = any(
            pos in ("all", "all_tokens")
            for layer in reft_layers
            if hasattr(layer, "_reft_adapter_positions")
            for pos in layer._reft_adapter_positions.values()
        )
        if not any_all_position:
            # Pure decode + no "all" adapters → skip everything
            for layer in reft_layers:
                layer._reft_all_masks_zero = True
            return

    # Compute the set of adapter IDs actually referenced in this batch.
    # One .unique() call on a small 1-D int tensor — fast, no GPU sync
    # needed since we only use the result as a Python set for membership
    # checks in the forward pass.
    active_ids_tensor = token_reft_ids.unique()
    batch_active_ids: set[int] = set(active_ids_tensor.tolist())
    batch_active_ids.discard(0)  # 0 = no adapter

    # Cache position masks by position string to avoid redundant computation.
    pos_mask_cache: dict[str, Optional[torch.Tensor]] = {}

    for layer in reft_layers:
        if not hasattr(layer, "reft_adapters"):
            continue
        # Intersect batch-active IDs with this layer's loaded adapters.
        layer._reft_active_ids = batch_active_ids & {
            int(s) for s in layer.reft_adapters}
        layer._reft_all_masks_zero = len(layer._reft_active_ids) == 0
        if layer._reft_all_masks_zero:
            continue
        for str_id in layer.reft_adapters:
            int_id = int(str_id)
            if int_id not in layer._reft_active_ids:
                continue
            # Adapter membership mask
            adapter_mask = (token_reft_ids == int_id).float()
            # Position mask (cached by position string)
            pos = layer._reft_adapter_positions.get(int_id, "prefill")
            if pos not in pos_mask_cache:
                if pos in ("all", "all_tokens"):
                    pos_mask_cache[pos] = None
                else:
                    pos_mask_cache[pos] = _compute_position_mask(
                        positions, pos, torch.float32, num_tokens,
                        attn_metadata)
            pos_mask = pos_mask_cache[pos]
            # Combined mask
            if pos_mask is not None:
                combined = adapter_mask * pos_mask
            else:
                combined = adapter_mask
            # Write into pre-allocated buffer
            buf = layer._reft_combined_masks.get(int_id)
            if buf is None or buf.shape[0] < num_tokens:
                buf = torch.zeros(max(num_tokens, _REFT_MASK_FALLBACK_SIZE),
                                  dtype=torch.float32,
                                  device=positions.device)
                layer._reft_combined_masks[int_id] = buf
            buf[:num_tokens].copy_(combined[:num_tokens])
            buf[num_tokens:].zero_()


# ---------------------------------------------------------------------------
# Qwen2 ReFT-aware decoder layer factory
# ---------------------------------------------------------------------------

def make_reft_qwen2_layer(reft_spec: Optional[dict] = None) -> type:
    """Return a Qwen2DecoderLayer subclass that supports multi-ReFT adapters.

    If *reft_spec* is provided (single-adapter backward compat), the initial
    adapter is loaded at construction time with ``reft_int_id=1``.
    If *reft_spec* is ``None`` (``enable_reft=True`` mode), layers are created
    with empty adapter dicts ready for dynamic loading.
    """
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

    if reft_spec is not None:
        layer_indices_set = frozenset(reft_spec["layer_indices"])
        position = reft_spec["position"]
        sample_adapter: Optional[nn.Module] = reft_spec["sample_adapter"]
        per_layer_adapters: dict = reft_spec.get("adapters", {})
        debug_mask_enabled = bool(reft_spec.get("debug_mask", False))
    else:
        layer_indices_set = frozenset()
        position = "prefill"
        sample_adapter = None
        per_layer_adapters = {}
        debug_mask_enabled = False

    class ReFTQwen2DecoderLayer(Qwen2DecoderLayer):
        """Qwen2DecoderLayer with multi-ReFT adapter support."""

        def __init__(self, config, cache_config=None, quant_config=None,
                     prefix=""):
            super().__init__(config=config, cache_config=cache_config,
                             quant_config=quant_config, prefix=prefix)

            layer_idx = _extract_layer_idx(prefix)
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

            model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
            if model_dtype is None:
                model_dtype = torch.bfloat16
            hidden_size = getattr(config, "hidden_size", 896)

            object.__setattr__(self, "_reft_layer_idx",
                               layer_idx if layer_idx is not None else -1)
            object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
            self.reft_adapter = None  # backward compat attribute

            _init_multi_reft_state(self, dev, hidden_size)

            # Load initial adapter from reft_spec (backward compat, id=1)
            if (layer_idx is not None and layer_idx in layer_indices_set
                    and sample_adapter is not None):
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                adapter_copy = _prepare_adapter(source, dev, model_dtype)
                _add_adapter_to_layer(self, 1, adapter_copy, position, dev)
                _REFT_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[ReFT-vLLM] Attached initial adapter to Qwen2 layer %d "
                    "(adapter_type=%s)", layer_idx,
                    type(adapter_copy).__name__)

        def forward(self, positions, hidden_states, residual):
            return _multi_reft_forward(
                self, positions, hidden_states, residual,
                super_forward=super().forward)

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTQwen2DecoderLayer.__name__ = "ReFTQwen2DecoderLayer"
    ReFTQwen2DecoderLayer.__qualname__ = "ReFTQwen2DecoderLayer"
    return ReFTQwen2DecoderLayer


# ---------------------------------------------------------------------------
# Llama ReFT-aware decoder layer factory
# ---------------------------------------------------------------------------

def make_reft_llama_layer(reft_spec: Optional[dict] = None) -> type:
    """Return a LlamaDecoderLayer subclass that supports multi-ReFT adapters.

    Analogous to ``make_reft_qwen2_layer`` but for the Llama architecture.
    """
    from vllm.model_executor.models.llama import LlamaDecoderLayer

    if reft_spec is not None:
        layer_indices_set = frozenset(reft_spec["layer_indices"])
        position = reft_spec["position"]
        sample_adapter: Optional[nn.Module] = reft_spec["sample_adapter"]
        per_layer_adapters: dict = reft_spec.get("adapters", {})
        debug_mask_enabled = bool(reft_spec.get("debug_mask", False))
    else:
        layer_indices_set = frozenset()
        position = "prefill"
        sample_adapter = None
        per_layer_adapters = {}
        debug_mask_enabled = False

    class ReFTLlamaDecoderLayer(LlamaDecoderLayer):
        """LlamaDecoderLayer with multi-ReFT adapter support."""

        def __init__(self, vllm_config, prefix="", config=None):
            super().__init__(vllm_config=vllm_config, prefix=prefix,
                             config=config)

            layer_idx = _extract_layer_idx(prefix)
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

            llama_config = getattr(vllm_config, "model_config", vllm_config)
            hf_config = getattr(llama_config, "hf_config",
                                getattr(llama_config, "config", config))
            model_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
            if model_dtype is None:
                model_dtype = torch.bfloat16
            hidden_size = getattr(hf_config, "hidden_size", 4096)

            object.__setattr__(self, "_reft_layer_idx",
                               layer_idx if layer_idx is not None else -1)
            object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
            self.reft_adapter = None  # backward compat attribute

            _init_multi_reft_state(self, dev, hidden_size)

            # Load initial adapter from reft_spec (backward compat, id=1)
            if (layer_idx is not None and layer_idx in layer_indices_set
                    and sample_adapter is not None):
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                adapter_copy = _prepare_adapter(source, dev, model_dtype)
                _add_adapter_to_layer(self, 1, adapter_copy, position, dev)
                _REFT_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[ReFT-vLLM] Attached initial adapter to Llama layer %d "
                    "(adapter_type=%s)", layer_idx,
                    type(adapter_copy).__name__)

        def forward(self, positions, hidden_states, residual):
            return _multi_reft_forward(
                self, positions, hidden_states, residual,
                super_forward=super().forward)

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTLlamaDecoderLayer.__name__ = "ReFTLlamaDecoderLayer"
    ReFTLlamaDecoderLayer.__qualname__ = "ReFTLlamaDecoderLayer"
    return ReFTLlamaDecoderLayer


# ---------------------------------------------------------------------------
# Qwen3 MoE ReFT-aware decoder layer factory
# ---------------------------------------------------------------------------

def make_reft_qwen3_moe_layer(reft_spec: Optional[dict] = None) -> type:
    """Return a Qwen3MoeDecoderLayer subclass that applies ReFT adapters.

    MoE-agnostic: the residual-stream delta is applied AFTER the full layer
    (attention + sparse MoE block) executes, so the adapter never interacts
    with expert routing or per-expert state.

    Mirrors ``make_reft_llama_layer``; Qwen3MoeDecoderLayer.__init__ has the
    same ``(vllm_config, prefix)`` signature and the same
    ``(positions, hidden_states, residual)`` forward contract. Pass
    reft_spec=None to get a layer that registers the multi-adapter
    ModuleDict scaffolding without baking in a default adapter — the
    bench / serving path then loads adapters dynamically via the
    `load_reft_adapter` collective_rpc.
    """
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    if reft_spec is not None:
        layer_indices_set = frozenset(reft_spec["layer_indices"])
        position = reft_spec["position"]
        sample_adapter: Optional[nn.Module] = reft_spec["sample_adapter"]
        per_layer_adapters: dict = reft_spec.get("adapters", {})
        debug_mask_enabled = bool(reft_spec.get("debug_mask", False))
    else:
        layer_indices_set = frozenset()
        position = "prefill"
        sample_adapter = None
        per_layer_adapters = {}
        debug_mask_enabled = False

    class ReFTQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
        """Qwen3MoeDecoderLayer with optional ReFT adapter delta."""

        def __init__(self, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

            layer_idx = _extract_layer_idx(prefix)
            if layer_idx is not None and layer_idx in layer_indices_set:
                try:
                    dev = next(self.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                hf_config = vllm_config.model_config.hf_text_config
                model_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
                if model_dtype is None:
                    model_dtype = torch.bfloat16
                adapter_copy = copy.deepcopy(source).to(dev)
                for child in adapter_copy.modules():
                    if isinstance(child, nn.Linear):
                        child.to(dtype=model_dtype)

                _install_adapter_caches(adapter_copy, model_dtype=model_dtype)
                _init_reft_debug_buffers(self, dev)
                hidden_size = getattr(hf_config, "hidden_size", 4096)
                _init_reft_capture_buffers(self, hidden_size, dev)
                _init_reft_position_mask_buffer(
                    self, _REFT_MASK_FALLBACK_SIZE, dev)

                self.reft_adapter = adapter_copy
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                _REFT_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Qwen3MoE layer %d "
                    "(adapter_type=%s, has_R_cache=%s)",
                    layer_idx,
                    type(adapter_copy).__name__,
                    hasattr(adapter_copy, "_R_cache"),
                )
            else:
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

            object.__setattr__(self, "_reft_position", position)
            _maybe_log_reft_layer_init(
                arch="qwen3_moe",
                layer_idx=getattr(self, "_reft_layer_idx", -1),
                attached=getattr(self, "reft_adapter", None) is not None,
                debug_enabled=bool(getattr(self, "_reft_debug_enabled", False)),
                position=position,
                adapter=getattr(self, "reft_adapter", None),
            )

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual,
        ):
            hidden_states, residual = super().forward(
                positions, hidden_states, residual)

            if getattr(self, "reft_adapter", None) is None:
                return hidden_states, residual

            h_full = hidden_states + residual

            delta = self.reft_adapter._compute_delta(
                h_full.unsqueeze(0)).squeeze(0)

            mask_buf = getattr(self, "_reft_position_mask", None)
            if mask_buf is not None:
                N = delta.shape[0]
                delta = delta * mask_buf[:N].unsqueeze(-1).to(delta.dtype)

            hidden_states = delta
            residual = h_full
            return hidden_states, residual

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTQwen3MoeDecoderLayer.__name__ = "ReFTQwen3MoeDecoderLayer"
    ReFTQwen3MoeDecoderLayer.__qualname__ = "ReFTQwen3MoeDecoderLayer"
    return ReFTQwen3MoeDecoderLayer
