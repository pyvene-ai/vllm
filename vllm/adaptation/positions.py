# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named position-mask registry.

A *position* names a per-token phase predicate ("which tokens of the
batch does this adaptation touch").  Each registered position maps to a
function::

    fn(positions, dtype, num_tokens, phase) -> Optional[torch.Tensor]

where ``positions`` is the per-token position-id tensor for the batch,
``phase`` is a :class:`PhaseInfo` describing the batch's decode/prefill
split, and the return value is a per-token float mask of shape
``(num_tokens,)`` — or ``None`` for "apply unconditionally".

Builtins mirror the historical ReFT position strings: ``all``,
``all_tokens``, ``prefill``, ``decode``, ``first``, ``last``.  Custom
adaptations register their own with :func:`register_position_mask`.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = [
    "PhaseInfo",
    "get_position_mask",
    "position_active_in_decode",
    "register_position_mask",
    "registered_positions",
]


@dataclass(frozen=True)
class PhaseInfo:
    """Prefill/decode split of the current batch.

    In a vLLM v1 batch, decode tokens come first and prefill tokens
    last.  ``num_prefill_tokens`` is ``None`` when no attention
    metadata was available (masks then fall back to position-id
    heuristics).
    """
    num_prefill_tokens: Optional[int]
    num_decodes: int
    num_prefills: int
    query_start_loc: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor]


PositionMaskFn = Callable[
    [torch.Tensor, torch.dtype, int, PhaseInfo], Optional[torch.Tensor]]


@dataclass
class _PositionEntry:
    fn: PositionMaskFn
    # Whether tokens in the decode region can receive a nonzero mask.
    # Used by the pure-decode fast path to skip adapters that can never
    # fire.  Custom positions default to True (never skipped — safe).
    active_in_decode: bool = True


_POSITION_REGISTRY: dict[str, _PositionEntry] = {}


def register_position_mask(name: str,
                           fn: PositionMaskFn,
                           *,
                           active_in_decode: bool = True,
                           override: bool = False) -> None:
    """Register a named position mask.

    Args:
        name: Position name used in adapter configs.
        fn: ``fn(positions, dtype, num_tokens, phase)`` returning a
            per-token float mask or ``None`` (= all tokens).
        active_in_decode: Set False only if the mask is provably zero
            for every decode-region token; enables the pure-decode
            batch skip.
        override: Allow replacing an existing registration.
    """
    if name in _POSITION_REGISTRY and not override:
        raise ValueError(
            f"Position {name!r} is already registered; pass override=True "
            "to replace it.")
    _POSITION_REGISTRY[name] = _PositionEntry(
        fn=fn, active_in_decode=active_in_decode)


def registered_positions() -> dict[str, _PositionEntry]:
    """The live registry mapping (mutable; pop() to unregister)."""
    return _POSITION_REGISTRY


def get_position_mask(name: str, positions: torch.Tensor,
                      dtype: torch.dtype, num_tokens: int,
                      phase: PhaseInfo) -> Optional[torch.Tensor]:
    """Compute the mask for position *name*, or ``None`` for all-tokens."""
    entry = _POSITION_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown position {name!r}. Registered positions: "
            f"{sorted(_POSITION_REGISTRY)}. Use "
            "vllm.adaptation.register_position_mask() to add custom ones.")
    return entry.fn(positions, dtype, num_tokens, phase)


def position_active_in_decode(name: str) -> bool:
    """Whether an adaptation with this position can fire on decode tokens.

    Unknown names return True (never skip what we don't understand)."""
    entry = _POSITION_REGISTRY.get(name)
    return True if entry is None else entry.active_in_decode


# ---------------------------------------------------------------------------
# Builtin positions
# ---------------------------------------------------------------------------

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
        return torch.zeros(num_prefill_tokens,
                           device=prefill_positions.device,
                           dtype=torch.long)

    # --- Preferred path: use query_start_loc from metadata ---
    if query_start_loc is not None and num_prefills > 0:
        # query_start_loc covers all requests (decode + prefill), shape
        # (N+1,).  Prefill requests start at index num_decodes.  The
        # prefill-region offsets are relative to the start of prefill
        # tokens.
        prefill_qsl = (query_start_loc[num_decodes:]
                       - query_start_loc[num_decodes])
        prefill_tok_idx = torch.arange(num_prefill_tokens,
                                       device=prefill_positions.device)
        req_idx = torch.searchsorted(prefill_qsl[1:], prefill_tok_idx,
                                     right=True)
        return req_idx.clamp(max=num_prefills - 1)

    # --- Fallback: detect request boundaries from position resets ---
    is_boundary = torch.zeros(num_prefill_tokens,
                              device=prefill_positions.device,
                              dtype=torch.long)
    is_boundary[1:] = (
        prefill_positions[1:] <= prefill_positions[:-1]).long()
    return is_boundary.cumsum(0)


def _all_mask(positions, dtype, num_tokens, phase):
    return None


def _prefill_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        token_idx = torch.arange(num_tokens, device=positions.device)
        return (token_idx >= num_decode_tokens).to(dtype)
    gate = (positions[0:1] == 0).to(dtype)
    return gate.expand(num_tokens)


def _decode_mask(positions, dtype, num_tokens, phase):
    # Exact complement of "prefill": the decode region at the start of a
    # v1 batch.  A request running its final prompt chunk is still in
    # the prefill region, so the decode adapter only kicks in from the
    # step after the prompt completes.
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        token_idx = torch.arange(num_tokens, device=positions.device)
        return (token_idx < num_decode_tokens).to(dtype)
    # Fallback: whole batch is decode iff the first token is not at
    # position 0 (mirrors the "prefill" gate).
    gate = (positions[0:1] != 0).to(dtype)
    return gate.expand(num_tokens)


def _first_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        # position == 0 only appears in the very first chunk of a
        # request, so a simple check is sufficient — no seq_lens needed.
        full_mask = torch.zeros(num_tokens, device=positions.device,
                                dtype=dtype)
        if phase.num_prefill_tokens > 0:
            prefill_positions = positions[num_decode_tokens:]
            full_mask[num_decode_tokens:] = (
                prefill_positions == 0).to(dtype)
        return full_mask
    in_prefill = (positions[0:1] == 0).to(dtype)
    return (positions == 0).to(dtype) * in_prefill


def _last_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        full_mask = torch.zeros(num_tokens, device=positions.device,
                                dtype=dtype)
        if phase.num_prefill_tokens == 0:
            return full_mask

        prefill_positions = positions[num_decode_tokens:]  # (P,)

        # --- Identify the last token of each request's query span ---
        req_idx = _prefill_request_indices(
            phase.num_prefill_tokens, phase.num_decodes,
            phase.num_prefills, phase.query_start_loc, prefill_positions)

        # last-in-query-span: token where req_idx changes or final token
        is_last_in_span = torch.zeros(phase.num_prefill_tokens,
                                      device=positions.device, dtype=dtype)
        if phase.num_prefill_tokens > 1:
            is_last_in_span[:-1] = (req_idx[1:] != req_idx[:-1]).to(dtype)
        is_last_in_span[-1] = 1.0

        # --- Filter to only true last prefill tokens ---
        if phase.seq_lens is not None:
            # seq_lens is ordered [decode_reqs..., prefill_reqs...]
            prefill_seq_lens = phase.seq_lens[phase.num_decodes:]
            expected_last_pos = prefill_seq_lens[req_idx] - 1
            is_true_last = (
                prefill_positions == expected_last_pos).to(dtype)
            full_mask[num_decode_tokens:] = is_last_in_span * is_true_last
        else:
            # No seq_lens available — fall back to last-in-span only.
            full_mask[num_decode_tokens:] = is_last_in_span

        return full_mask

    # Fallback tensor-only "last" mask (no prefill/decode split info).
    is_last = torch.zeros_like(positions, dtype=dtype)
    is_last[-1] = 1.0
    next_is_zero = (torch.roll(positions, -1) == 0).to(dtype)
    is_last = is_last + next_is_zero * (1.0 - is_last)
    in_prefill = (positions[0:1] == 0).to(dtype)
    return is_last * in_prefill


register_position_mask("all", _all_mask, active_in_decode=True)
register_position_mask("all_tokens", _all_mask, active_in_decode=True)
register_position_mask("prefill", _prefill_mask, active_in_decode=False)
register_position_mask("decode", _decode_mask, active_in_decode=True)
register_position_mask("first", _first_mask, active_in_decode=False)
register_position_mask("last", _last_mask, active_in_decode=False)
