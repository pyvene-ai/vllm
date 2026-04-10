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

    # Adapter weights are synced via TRL's standard sync_weights() pipeline.
    # After sync, call llm.collective_rpc("refresh_reft_caches") to recompute
    # derived inference caches.
"""

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

_SENTINEL = object()


def _adapter_to_blueprint(adapter) -> dict:
    """Extract a serializable blueprint from any ReFT adapter instance.

    Stores the class path and constructor kwargs so the adapter can be
    re-instantiated fresh in spawned worker processes.  This avoids two
    problems with saving the module directly:

    1. ``torch.save`` cannot pickle modules that have
       ``torch.nn.utils.parametrizations.orthogonal`` (or ``weight_norm``)
       applied.
    2. De-parametrizing before saving causes a key-name mismatch when
       ``sync_reft_state`` loads the HF state dict: the HF checkpoint stores
       ``rotate_layer.parametrizations.weight.original`` but a de-parametrized
       adapter only has ``rotate_layer.weight``, so the trained R matrix is
       silently skipped and the adapter runs with the wrong rotation.

    A fresh adapter instantiated from the blueprint has the full parametrized
    key tree, so the HF state dict loads without any remapping and all
    trained weights — including R — are correctly applied.

    Works for any adapter class; constructor kwargs are extracted by walking
    the MRO to find the first ``__init__`` with explicit named parameters, then
    matching them to stored instance attributes.
    """
    import inspect

    cls = type(adapter)

    # Walk the MRO and accumulate explicit named parameters from every
    # __init__ in the chain.  Subclasses like LoReFTRidgeAdapter define
    # ``def __init__(self, *args, lam=1e-3, **kwargs)`` — the old code
    # stopped at the first __init__ with *any* explicit param (``lam``)
    # and never reached W2Adapter.__init__ which defines ``hidden_size``,
    # ``low_rank_dim``, etc.  Now we collect params from all levels, with
    # subclass params taking priority on name collisions.
    init_params: dict = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        init_fn = klass.__dict__.get("__init__")
        if init_fn is None:
            continue
        sig = inspect.signature(init_fn)
        explicit = {
            k: v for k, v in sig.parameters.items()
            if k != "self" and v.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        init_params.update(explicit)

    kwargs: dict = {}
    for name, param in init_params.items():
        if name == "device":
            kwargs["device"] = "cpu"
            continue

        val = getattr(adapter, name, _SENTINEL)
        if val is _SENTINEL:
            # Attribute not stored; use the default if one exists.
            if param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
        elif isinstance(val, (int, float, str, bool, type(None))):
            kwargs[name] = val
        elif name == "dtype":
            # torch.dtype → string, e.g. "torch.bfloat16"
            kwargs[name] = str(val)
        elif name == "mixer":
            # Stored as an nn.Module instance; reverse-lookup the string key.
            kwargs[name] = _mixer_instance_to_name(val)
        elif param.default is not inspect.Parameter.empty:
            # Non-serializable value; fall back to the default.
            kwargs[name] = param.default
        # else: omit — constructor must not require it or will use its default

    # dtype is not always stored as a direct attribute; derive from weights.
    # Check for None too — the MRO walk may pick up the default (None) when
    # the adapter doesn't store dtype as self.dtype.
    if kwargs.get("dtype") is None:
        ls = getattr(adapter, "learned_source", None)
        if ls is not None and hasattr(ls, "weight"):
            kwargs["dtype"] = str(ls.weight.dtype)

    # Save trained weights alongside the blueprint so that worker processes
    # start with correct weights *before* CUDA graph capture / torch.compile
    # warmup.  Without this, the blueprint adapter is constructed with random
    # init weights, and post-init sync_reft_state() updates them in-place —
    # but compiled/captured graphs may not reflect the in-place updates.
    import torch
    adapter_state = {
        k: v.detach().cpu()
        for k, v in adapter.state_dict().items()
    }

    blueprint = {
        "__type__": "AdapterBlueprint",
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "kwargs": kwargs,
        "state_dict": adapter_state,
    }
    import sys
    print(
        f"[vllm.reft] _adapter_to_blueprint: "
        f"{cls.__qualname__} from {cls.__module__} | "
        f"kwargs={{{', '.join(f'{k}={v!r}' for k, v in sorted(kwargs.items()))}}} | "
        f"state_keys={sorted(adapter_state.keys())}",
        file=sys.stderr, flush=True,
    )
    return blueprint


def _mixer_instance_to_name(mixer_instance) -> Optional[str]:
    """Return the MIXER_REGISTRY key for *mixer_instance*, or ``None``."""
    if mixer_instance is None:
        return None
    try:
        from adaptors._mixer import MIXER_REGISTRY
        for name, mixer_cls in MIXER_REGISTRY.items():
            if isinstance(mixer_instance, mixer_cls):
                return name
    except ImportError:
        pass
    return None


def _blueprint_to_adapter(blueprint: dict):
    """Re-instantiate a fresh adapter from a saved blueprint dict.

    Imports the adapter class by its module path, converts the stored dtype
    string back to a ``torch.dtype``, then calls the constructor.  If the
    blueprint contains a ``state_dict``, loads trained weights immediately
    so the adapter is ready *before* CUDA graph capture / torch.compile.
    """
    import importlib
    import sys
    import torch

    mod_name = blueprint["__module__"]
    qual_name = blueprint["__qualname__"]
    has_state = "state_dict" in blueprint and blueprint["state_dict"]
    print(
        f"[vllm.reft] _blueprint_to_adapter: "
        f"module={mod_name} qualname={qual_name} "
        f"kwargs_keys={sorted(blueprint['kwargs'].keys())} "
        f"has_state_dict={has_state}",
        file=sys.stderr, flush=True,
    )

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, qual_name)

    kwargs = dict(blueprint["kwargs"])
    dtype_val = kwargs.get("dtype")
    if isinstance(dtype_val, str):
        # "torch.bfloat16" → torch.bfloat16
        kwargs["dtype"] = getattr(torch, dtype_val.split(".")[-1], None)

    adapter = cls(**kwargs)

    # Load trained weights if available in the blueprint.
    saved_state = blueprint.get("state_dict")
    if saved_state:
        missing, unexpected = adapter.load_state_dict(saved_state, strict=False)
        allowed_missing = {"_R_cache", "_w2_pinv_cache", "_w2_ridge_cache"}
        real_missing = [k for k in missing if k not in allowed_missing]
        if real_missing or unexpected:
            print(
                f"[vllm.reft] Blueprint state_dict load: "
                f"missing={real_missing} unexpected={unexpected}",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[vllm.reft] Blueprint state_dict loaded OK "
                f"({len(saved_state)} keys)",
                file=sys.stderr, flush=True,
            )
        # Refresh inference caches (e.g. _R_cache) from loaded weights.
        if hasattr(adapter, "install_inference_caches"):
            adapter.install_inference_caches()
        if hasattr(adapter, "refresh_inference_caches"):
            adapter.refresh_inference_caches()

    return adapter


def _write_spec_file(spec: dict) -> None:
    """Serialise *spec* to a temp file and record the path in the environment.

    The sample_adapter is replaced with a plain blueprint dict (constructor
    kwargs) before saving.  See ``_adapter_to_blueprint`` for why this is
    preferable to de-parametrizing the adapter module.
    """
    import torch

    saveable_spec = dict(spec)
    adapter = saveable_spec.get("sample_adapter")
    if adapter is not None:
        saveable_spec["sample_adapter"] = _adapter_to_blueprint(adapter)

    fd, path = tempfile.mkstemp(suffix=".pt", prefix="vllm_reft_spec_")
    os.close(fd)
    try:
        torch.save(saveable_spec, path)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    os.environ[_SPEC_FILE_ENV_KEY] = path


def _read_spec_file() -> Optional[dict]:
    """Load and return the spec from the temp file, or ``None``.

    If the spec was written by a newer version that stores a blueprint dict
    instead of a module, the adapter is reconstructed before returning.
    """
    import torch

    path = os.environ.get(_SPEC_FILE_ENV_KEY)
    if not path or not os.path.exists(path):
        return None
    try:
        spec = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    # Reconstruct adapter from blueprint if this is a new-style spec file.
    adapter = spec.get("sample_adapter")
    if isinstance(adapter, dict) and adapter.get("__type__") == "AdapterBlueprint":
        import sys
        try:
            spec["sample_adapter"] = _blueprint_to_adapter(adapter)
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "[vllm.reft] Failed to reconstruct adapter from blueprint: %s. "
                "Blueprint: %s", e, adapter,
            )
            # Also print to stderr — logger may not be configured in worker
            print(
                f"[vllm.reft] CRITICAL: Blueprint adapter reconstruction FAILED: "
                f"{type(e).__name__}: {e}\n"
                f"  module={adapter.get('__module__')}\n"
                f"  qualname={adapter.get('__qualname__')}\n"
                f"  kwargs_keys={sorted(adapter.get('kwargs', {}).keys())}",
                file=sys.stderr, flush=True,
            )
            return None
        else:
            rebuilt = spec["sample_adapter"]
            print(
                f"[vllm.reft] Blueprint reconstructed OK: "
                f"{type(rebuilt).__name__} | "
                f"state_keys={sorted(rebuilt.state_dict().keys())}",
                file=sys.stderr, flush=True,
            )

    return spec


def _remove_spec_file() -> None:
    """Delete the temp file and clear the env var."""
    path = os.environ.pop(_SPEC_FILE_ENV_KEY, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
