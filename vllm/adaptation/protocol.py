# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptation protocol helpers: mount sites, blend, capability checks."""

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "MOUNT_SITES",
    "LINEAR_SITE_PREFIX",
    "apply_adaptation",
    "check_adaptation_supported",
    "resolve_site_submodule_path",
    "validate_site",
]

# Named mount points inside a decoder layer.
#   block_output — the residual stream after the whole block (default;
#                  historical ReFT mount).
#   block_input  — the residual stream before the block runs.
#   post_attn    — the attention module's output, before the residual
#                  add (hooked on layer.self_attn).
#   post_mlp     — the MLP module's output, before the residual add
#                  (hooked on layer.mlp).
# Additionally "linear:<submodule.path>" mounts on the output of any
# named submodule of the layer, e.g. "linear:self_attn.qkv_proj" or
# "linear:mlp.gate_up_proj" (note vLLM merges q/k/v and gate/up).
MOUNT_SITES = ("block_output", "block_input", "post_attn", "post_mlp")
LINEAR_SITE_PREFIX = "linear:"

# Hook targets for the fixed non-block sites.
_SITE_SUBMODULES = {
    "post_attn": "self_attn",
    "post_mlp": "mlp",
}


def validate_site(site: str) -> None:
    """Raise ValueError if *site* is not a recognized mount point."""
    if site in MOUNT_SITES:
        return
    if site.startswith(LINEAR_SITE_PREFIX):
        if not site[len(LINEAR_SITE_PREFIX):]:
            raise ValueError(
                "linear site must name a submodule, e.g. "
                "'linear:self_attn.qkv_proj'")
        return
    raise ValueError(
        f"Unknown mount site {site!r}. Expected one of {MOUNT_SITES} or "
        f"'{LINEAR_SITE_PREFIX}<submodule.path>'.")


def resolve_site_submodule_path(site: str) -> Optional[str]:
    """Submodule path (relative to the decoder layer) a site hooks onto.

    Returns ``None`` for the block-level sites, which are applied inside
    the layer's own forward rather than via a submodule hook.
    """
    validate_site(site)
    if site in ("block_output", "block_input"):
        return None
    if site.startswith(LINEAR_SITE_PREFIX):
        return site[len(LINEAR_SITE_PREFIX):]
    return _SITE_SUBMODULES[site]


def apply_adaptation(adaptation: nn.Module, hidden: torch.Tensor,
                     mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Blend one adaptation's computation into *hidden*.

    If the adaptation defines ``apply_masked(h, mask) -> h'`` that wins;
    otherwise the default additive blend runs:
    ``h + mask * _compute_delta(h)``.

    Args:
        adaptation: The adaptation module.
        hidden: ``(num_tokens, dim)`` stream at the mount site.
        mask: Per-token float mask ``(num_tokens,)``, or ``None`` for
            all-tokens.
    """
    if mask is None:
        mask = torch.ones(hidden.shape[0], device=hidden.device,
                          dtype=torch.float32)
    apply_masked = getattr(adaptation, "apply_masked", None)
    if apply_masked is not None:
        return apply_masked(hidden, mask)
    delta = adaptation._compute_delta(hidden.unsqueeze(0)).squeeze(0)
    return hidden + delta * mask.unsqueeze(-1).to(delta.dtype)


def needs_sequence_segmentation(adaptation: nn.Module) -> bool:
    """Whether this adaptation mixes information along the sequence axis.

    Sequence-mixing computations (cnn/bigram mixers, chunked adapters)
    must run per request span — vLLM's flattened batch concatenates
    unrelated requests, and mixing across the boundary leaks one
    request's hiddens into another's delta.

    An explicit ``sequence_mixing`` attribute wins; otherwise the
    presence of a ``mixer`` implies sequence mixing.
    """
    explicit = getattr(adaptation, "sequence_mixing", None)
    if explicit is not None:
        return bool(explicit)
    return getattr(adaptation, "mixer", None) is not None


def check_adaptation_supported(adaptation: nn.Module) -> None:
    """Reject adaptations that cannot run correctly under vLLM serving.

    Mixers with recurrent state across decode steps (``stateful``) or
    that need separate k/v streams (``needs_kv``) require per-request
    state that does not survive vLLM's batching, reordering, and
    preemption — loading them would corrupt generations silently.
    """
    mixer = getattr(adaptation, "mixer", None)
    if mixer is None:
        return
    if getattr(mixer, "stateful", False):
        raise ValueError(
            f"Adaptation {type(adaptation).__name__} uses a stateful mixer "
            f"({type(mixer).__name__}); recurrent decode-time state is not "
            "supported under vLLM serving (per-request state does not "
            "survive batching/reordering/preemption).")
    if getattr(mixer, "needs_kv", False):
        raise ValueError(
            f"Adaptation {type(adaptation).__name__} uses a needs_kv mixer "
            f"({type(mixer).__name__}); routing separate k/v streams into "
            "adaptations is not supported under vLLM serving.")
