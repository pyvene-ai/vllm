"""ReFT adapter model manager with LRU eviction.

Manages the lifecycle of ReFT adapters: registration (CPU cache),
activation (GPU layers), deactivation, and LRU eviction.  Mirrors the
``LRUCacheLoRAModelManager`` pattern from ``vllm.lora.models`` but is
much simpler because ReFT adapters are tiny (~16K params/layer).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from vllm.reft.layer import _add_adapter_to_layer, _prepare_adapter
from vllm.utils import LRUCache

logger = logging.getLogger("vllm.reft.models")


@dataclass
class ReFTModel:
    """CPU-side representation of a registered ReFT adapter."""

    id: int
    """Unique integer ID (>= 1).  0 is reserved for base model."""
    position: str
    """Position mode: ``"first"``, ``"last"``, ``"prefill"``,
    ``"decode"``, or ``"all"``.  ``"decode"`` is the exact complement of
    ``"prefill"``."""
    adapter_config: dict
    """Serializable config dict (from ``spec_to_reft_config``)."""
    layer_indices: frozenset[int] = field(default_factory=frozenset)
    """Which decoder layers this adapter applies to."""

    def __post_init__(self):
        if self.id < 1:
            raise ValueError(
                f"ReFT adapter id must be >= 1, got {self.id}")


class _ReFTLRUCache(LRUCache[int, ReFTModel]):
    """LRU cache that calls *deactivate_fn* when an entry is evicted."""

    def __init__(self, capacity: int, deactivate_fn):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: Optional[ReFTModel]):
        logger.debug("LRU evicting ReFT adapter id=%d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


class ReFTModelManager:
    """Centralized lifecycle manager for ReFT adapters.

    Two-tier LRU caching:
      - **registered** (CPU, ``max_cpu_refts``): holds ``ReFTModel``
        descriptors.  Cheap to store — just metadata + serializable config.
      - **active** (GPU, ``max_refts``): adapters whose weights are
        loaded into the decoder layer ``nn.ModuleDict``.  When full,
        the least-recently-used adapter is evicted from GPU.

    The manager reuses the existing per-layer functions from
    ``vllm.reft.layer``: ``_prepare_adapter`` and ``_add_adapter_to_layer``
    for loading, and the deletion logic from ``worker_base.py`` for
    unloading.
    """

    def __init__(
        self,
        reft_layers: list[nn.Module],
        max_refts: int = 256,
        max_cpu_refts: int = 1024,
        device: torch.device = torch.device("cuda"),
        model_dtype: torch.dtype = torch.bfloat16,
    ):
        self.reft_layers = reft_layers
        self.max_refts = max_refts
        self.device = device
        self.model_dtype = model_dtype

        # Slot map: index → adapter id (None = free).
        self.reft_index_to_id: list[Optional[int]] = [None] * max_refts

        # CPU cache — eviction here just logs, no GPU teardown needed.
        self._registered: _ReFTLRUCache = _ReFTLRUCache(
            max_cpu_refts, self._on_cpu_evict)

        # GPU cache — eviction triggers _deactivate_adapter.
        self._active: _ReFTLRUCache = _ReFTLRUCache(
            max_refts, self._deactivate_adapter)

    # ------------------------------------------------------------------
    # Registration (CPU cache)
    # ------------------------------------------------------------------

    def add_adapter(self, reft_model: ReFTModel) -> bool:
        """Register an adapter in the CPU cache.

        Returns True if the adapter was newly registered, False if it was
        already registered (LRU order is updated either way).
        """
        if reft_model.id in self._registered:
            self._registered.touch(reft_model.id)
            return False
        self._registered[reft_model.id] = reft_model
        logger.debug("Registered ReFT adapter id=%d (%d registered)",
                     reft_model.id, len(self._registered))
        return True

    def remove_adapter(self, reft_id: int) -> bool:
        """Remove an adapter from both CPU and GPU caches."""
        if reft_id in self._active:
            del self._active[reft_id]
        if reft_id in self._registered:
            del self._registered[reft_id]
            return True
        return False

    def list_adapters(self) -> dict[int, ReFTModel]:
        """Return all registered adapters."""
        return dict(self._registered.cache)

    # ------------------------------------------------------------------
    # Activation (GPU)
    # ------------------------------------------------------------------

    def activate_adapter(self, reft_id: int) -> bool:
        """Load an adapter onto GPU layers.

        If the GPU cache is full, the LRU adapter is automatically evicted.
        Returns True on success.
        """
        if reft_id in self._active:
            self._active.touch(reft_id)
            return True

        reft_model = self._registered.get(reft_id)
        if reft_model is None:
            raise ValueError(
                f"ReFT adapter id={reft_id} is not registered. "
                f"Call add_adapter() first.")

        # Find a free slot (LRU auto-evicts if full).
        if len(self._active) >= self.max_refts:
            self._active.remove_oldest()

        slot = self._find_free_slot()
        if slot is None:
            raise RuntimeError("No free ReFT slots after LRU eviction — "
                               "this should not happen.")

        self.reft_index_to_id[slot] = reft_id

        # Load weights into decoder layers.
        count = self._load_adapter_to_layers(reft_model)
        logger.debug("Activated ReFT adapter id=%d in slot %d "
                     "(%d layers)", reft_id, slot, count)

        self._active[reft_id] = reft_model
        return True

    def _deactivate_adapter(self, reft_id: int) -> None:
        """Remove adapter weights from GPU layers and free the slot."""
        # Free the slot.
        try:
            idx = self.reft_index_to_id.index(reft_id)
            self.reft_index_to_id[idx] = None
        except ValueError:
            pass

        # Remove from all layers.
        key = str(reft_id)
        for layer in self.reft_layers:
            if key in layer.reft_adapters:
                del layer.reft_adapters[key]
                layer._reft_adapter_positions.pop(reft_id, None)
                layer._reft_combined_masks.pop(reft_id, None)
                # Update backward compat reft_adapter reference.
                if getattr(layer, "reft_adapter", None) is not None:
                    remaining = list(layer.reft_adapters.values())
                    layer.reft_adapter = remaining[0] if remaining else None

        logger.debug("Deactivated ReFT adapter id=%d", reft_id)

    def _on_cpu_evict(self, reft_id: int) -> None:
        """Called when an adapter is evicted from the CPU cache."""
        # If still on GPU, deactivate first.
        if reft_id in self._active:
            del self._active[reft_id]
        logger.debug("CPU-evicted ReFT adapter id=%d", reft_id)

    # ------------------------------------------------------------------
    # Batch-level API (called from model runner each step)
    # ------------------------------------------------------------------

    def ensure_active(self, batch_reft_ids: set[int]) -> None:
        """Ensure all adapters needed for this batch are on GPU.

        Activates missing adapters (with LRU eviction if necessary) and
        bumps LRU order for already-active ones.  Adapters that exist on
        layers but were never registered with the manager (e.g. baked-in
        adapter id=1 from ``reft_config``) are silently skipped.
        """
        for reft_id in batch_reft_ids:
            if reft_id in self._active:
                self._active.touch(reft_id)
            elif reft_id in self._registered:
                self.activate_adapter(reft_id)
            # else: adapter may have been loaded directly onto layers
            # (e.g. baked-in via reft_config=), skip silently.

    def is_active(self, reft_id: int) -> bool:
        return reft_id in self._active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_free_slot(self) -> Optional[int]:
        """Return the index of the first free slot, or None."""
        for i, v in enumerate(self.reft_index_to_id):
            if v is None:
                return i
        return None

    def _load_adapter_to_layers(self, reft_model: ReFTModel) -> int:
        """Instantiate adapter weights and add to relevant layers.

        Returns the number of layers loaded.
        """
        from vllm.reft import reft_config_to_spec

        spec = reft_config_to_spec(reft_model.adapter_config)
        if spec is None:
            return 0

        count = 0
        for layer in self.reft_layers:
            # Determine layer index from the layer's prefix attribute.
            layer_idx = getattr(layer, "_reft_layer_idx", None)
            if layer_idx is None or layer_idx < 0:
                continue
            if reft_model.layer_indices and layer_idx not in reft_model.layer_indices:
                continue

            source = spec.get("adapters", {}).get(
                layer_idx, spec.get("sample_adapter"))
            if source is None:
                continue

            dev = self.device
            adapter_copy = _prepare_adapter(source, dev, self.model_dtype)
            _add_adapter_to_layer(layer, reft_model.id, adapter_copy,
                                  reft_model.position, dev)
            count += 1

        return count
