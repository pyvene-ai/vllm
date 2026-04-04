"""vllm.reft – first-class ReFT (Representation Fine-Tuning) support in vLLM.

This package owns everything required to run ReFT-adapted models in vLLM:

  * Thread-local spec injection so model constructors can pick up a ReFT spec
    without the caller needing to thread it through every kwarg.
  * ReFT-aware decoder layer factories (see layer.py).

Usage (from trainer code):
    import vllm.reft as vllm_reft

    vllm_reft.set_reft_spec({
        "layer_indices": [0, 4, 8, ...],
        "position": "prefill",          # "prefill" | "first" | "last"
        "sample_adapter": adapter,      # nn.Module – architecture template
    })
    try:
        llm = LLM(model=model_name, ...)  # model constructor reads the spec
    finally:
        vllm_reft.clear_reft_spec()

    llm.sync_reft_state(state_dict_per_layer)   # load actual trained weights
"""

import threading
from typing import Any, Optional

__all__ = [
    "set_reft_spec",
    "get_reft_spec",
    "clear_reft_spec",
]

# ---------------------------------------------------------------------------
# Thread-local ReFT spec storage
# ---------------------------------------------------------------------------

_reft_spec_local = threading.local()


def set_reft_spec(spec: Optional[dict[str, Any]]) -> None:
    """Store *spec* as the active ReFT spec for this thread.

    Must be called before the vLLM ``LLM(...)`` constructor so that model
    constructors (``Qwen2ForCausalLM``, ``LlamaForCausalLM``, …) can read it
    and create ReFT-aware decoder layers.

    spec keys:
        layer_indices   list[int]   – which decoder layers get adapters
        position        str         – "prefill" | "first" | "last"
        sample_adapter  nn.Module   – one adapter instance (architecture only;
                                      actual weights are loaded later via
                                      ``LLM.sync_reft_state``)
    """
    _reft_spec_local.spec = spec


def get_reft_spec() -> Optional[dict[str, Any]]:
    """Return the active ReFT spec for this thread, or ``None``."""
    return getattr(_reft_spec_local, "spec", None)


def clear_reft_spec() -> None:
    """Clear the active ReFT spec for this thread."""
    _reft_spec_local.spec = None
