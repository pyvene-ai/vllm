"""vllm.reft – first-class ReFT (Representation Fine-Tuning) support in vLLM.

This package owns everything required to run ReFT-adapted models in vLLM:

  * Thread-local spec injection so model constructors can pick up a ReFT spec
    without the caller needing to thread it through every kwarg.
  * Cross-process fallback via a temp file so that vLLM v1's spawn-based
    worker processes (which do NOT inherit threading.local state) also see
    the spec when ``Qwen2ForCausalLM.__init__`` is called in a worker.
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

import copy
import os
import tempfile
import threading
from typing import Any, Optional

__all__ = [
    "set_reft_spec",
    "get_reft_spec",
    "clear_reft_spec",
]

# ---------------------------------------------------------------------------
# Thread-local ReFT spec storage (same-process / same-thread fast path)
# ---------------------------------------------------------------------------

_reft_spec_local = threading.local()

# Environment variable that holds the path to the pickled spec file.
# Set by set_reft_spec() in the parent process; inherited by spawned workers.
_SPEC_FILE_ENV_KEY = "_VLLM_REFT_SPEC_FILE"


def set_reft_spec(spec: Optional[dict[str, Any]]) -> None:
    """Store *spec* as the active ReFT spec for this thread.

    Must be called before the vLLM ``LLM(...)`` constructor so that model
    constructors (``Qwen2ForCausalLM``, ``LlamaForCausalLM``, …) can read it
    and create ReFT-aware decoder layers.

    The spec is also serialised to a temporary file whose path is stored in
    the ``_VLLM_REFT_SPEC_FILE`` environment variable.  Spawned worker
    processes inherit env vars but not threading.local, so they can still
    recover the spec via ``get_reft_spec()``.

    spec keys:
        layer_indices   list[int]   – which decoder layers get adapters
        position        str         – "prefill" | "first" | "last"
        sample_adapter  nn.Module   – one adapter instance (architecture only;
                                      actual weights are loaded later via
                                      ``LLM.sync_reft_state``)
    """
    _reft_spec_local.spec = spec
    if spec is not None:
        _write_spec_file(spec)
    else:
        _remove_spec_file()


def get_reft_spec() -> Optional[dict[str, Any]]:
    """Return the active ReFT spec for this thread, or ``None``.

    Falls back to deserialising from the temp file when called from a spawned
    worker process that did not inherit the thread-local.
    """
    spec = getattr(_reft_spec_local, "spec", None)
    if spec is not None:
        return spec
    # Cross-process fallback: spawned workers inherit env vars.
    return _read_spec_file()


def clear_reft_spec() -> None:
    """Clear the active ReFT spec for this thread and remove the temp file."""
    _reft_spec_local.spec = None
    _remove_spec_file()


# ---------------------------------------------------------------------------
# Temp-file helpers for cross-process spec passing
# ---------------------------------------------------------------------------

def _write_spec_file(spec: dict) -> None:
    """Pickle *spec* to a temp file and record the path in the environment."""
    import torch

    # Deep-copy the spec and move the sample_adapter to CPU so that torch.save
    # can serialise it before CUDA is initialised in the worker process.
    cpu_spec = dict(spec)
    adapter = cpu_spec.get("sample_adapter")
    if adapter is not None and hasattr(adapter, "cpu"):
        cpu_spec["sample_adapter"] = copy.deepcopy(adapter).cpu()

    fd, path = tempfile.mkstemp(suffix=".pt", prefix="vllm_reft_spec_")
    os.close(fd)
    try:
        torch.save(cpu_spec, path)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    os.environ[_SPEC_FILE_ENV_KEY] = path


def _read_spec_file() -> Optional[dict]:
    """Load and return the spec from the temp file, or ``None``."""
    import torch

    path = os.environ.get(_SPEC_FILE_ENV_KEY)
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _remove_spec_file() -> None:
    """Delete the temp file and clear the env var."""
    path = os.environ.pop(_SPEC_FILE_ENV_KEY, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
