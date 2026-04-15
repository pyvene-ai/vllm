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


try:
    from vllm.utils import direct_register_custom_op
    try:
        _tag_cudagraph_unsafe = (torch._C.Tag.cudagraph_unsafe,)
    except AttributeError:
        _tag_cudagraph_unsafe = ()

    direct_register_custom_op(
        op_name="reft_apply_adapter",
        op_func=_reft_apply_adapter_op,
        fake_impl=_reft_apply_adapter_fake,
        tags=_tag_cudagraph_unsafe,
    )
    _REFT_CUSTOM_OP_AVAILABLE = True
    logger.info("[ReFT-vLLM] Registered reft_apply_adapter custom op (cudagraph_unsafe)")
except Exception as e:
    _REFT_CUSTOM_OP_AVAILABLE = False
    logger.warning("[ReFT-vLLM] Failed to register reft_apply_adapter custom op: %s", e)

# Allow disabling the custom op for debugging.
if os.environ.get("VLLM_REFT_NO_CUSTOM_OP", "").lower() in {"1", "true", "yes"}:
    _REFT_CUSTOM_OP_AVAILABLE = False
    logger.info("[ReFT-vLLM] Custom op disabled via VLLM_REFT_NO_CUSTOM_OP")


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
    per_layer_adapters: dict = reft_spec.get("adapters", {})
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
                # Use per-layer adapter when available to preserve the correct
                # parametrization base buffer; fall back to sample_adapter.
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                # Cast to model dtype so linear layers and matmuls don't
                # allocate temporaries during CUDA graph capture.
                model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
                if model_dtype is None:
                    model_dtype = torch.bfloat16
                adapter_copy = copy.deepcopy(source).to(dev)
                if hasattr(adapter_copy, "learned_source"):
                    adapter_copy.learned_source.to(dtype=model_dtype)

                # Install CUDA-graph-safe caches via the adapter's own protocol.
                _install_adapter_caches(adapter_copy, model_dtype=model_dtype)
                _init_reft_debug_buffers(self, dev)
                hidden_size = getattr(config, "hidden_size", 896)
                _init_reft_capture_buffers(self, hidden_size, dev)

                # Register as a proper nn.Module submodule so that
                # named_parameters() sees adapter weights and TRL's
                # sync_weights() can push them through the standard
                # update_named_param / load_weights path.
                self.reft_adapter = adapter_copy
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                _REFT_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Qwen2 layer %d "
                    "(adapter_type=%s, has_R_cache=%s)",
                    layer_idx,
                    type(adapter_copy).__name__,
                    hasattr(adapter_copy, "_R_cache"),
                )
            else:
                object.__setattr__(self, "_reft_layer_idx", -1)
                object.__setattr__(self, "_reft_debug_enabled", False)

            # Store position string as a normal attribute (not a parameter).
            object.__setattr__(self, "_reft_position", position)
            _maybe_log_reft_layer_init(
                arch="qwen2",
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

            # Inline adapter computation — pure tensor ops, fully
            # compilable.  No custom op, no graph splitting.
            # Adapter weights live at fixed addresses; in-place
            # .copy_() updates are visible to graph replays (same
            # as LoRA).
            delta = self.reft_adapter._compute_delta(
                h_full.unsqueeze(0)).squeeze(0)

            hidden_states = delta
            residual = h_full
            return hidden_states, residual

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
    per_layer_adapters: dict = reft_spec.get("adapters", {})
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
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                llama_config = getattr(vllm_config, "model_config", vllm_config)
                hf_config = getattr(llama_config, "hf_config",
                                    getattr(llama_config, "config", config))
                model_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
                if model_dtype is None:
                    model_dtype = torch.bfloat16
                # Move to device only (not dtype) — rotate_layer's
                # Householder parametrization requires all-float32.
                adapter_copy = copy.deepcopy(source).to(dev)
                # Cast learned_source to model dtype so nn.Linear
                # forward doesn't allocate float32 temporaries during
                # CUDA graph capture.
                if hasattr(adapter_copy, "learned_source"):
                    adapter_copy.learned_source.to(dtype=model_dtype)

                _install_adapter_caches(adapter_copy, model_dtype=model_dtype)
                _init_reft_debug_buffers(self, dev)
                hidden_size = getattr(hf_config, "hidden_size", 4096)
                _init_reft_capture_buffers(self, hidden_size, dev)

                self.reft_adapter = adapter_copy
                object.__setattr__(self, "_reft_layer_idx", layer_idx)
                object.__setattr__(self, "_reft_debug_enabled", debug_mask_enabled)
                _REFT_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[ReFT-vLLM] Attached adapter to Llama layer %d "
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
                arch="llama",
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

            hidden_states = delta
            residual = h_full
            return hidden_states, residual

        def get_reft_debug_stats(self) -> Optional[dict]:
            return _collect_layer_reft_debug_stats(self)

    ReFTLlamaDecoderLayer.__name__ = "ReFTLlamaDecoderLayer"
    ReFTLlamaDecoderLayer.__qualname__ = "ReFTLlamaDecoderLayer"
    return ReFTLlamaDecoderLayer
