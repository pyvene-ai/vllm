# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs must be re-captured when the ReFT adapter set changes.

Captured decode graphs bake in the adapter modules that existed at
capture time; loading or unloading an adapter afterwards would silently
not take effect inside replayed graphs.  The fix: every CUDAGraphWrapper
registers itself in a weak registry; changing the adapter set clears all
captured entries and re-enables capturing, so the next dispatch lazily
re-captures with the current adapter set.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation import monitor
from vllm.compilation.cuda_graph import (_cudagraph_wrappers,
                                         invalidate_all_cudagraphs)
from vllm.reft.layer import _add_adapter_to_layer, _init_multi_reft_state
from vllm.worker.worker_base import WorkerBase


class FakeWrapper:
    """Duck-typed stand-in for CUDAGraphWrapper (weakref-able)."""

    def __init__(self, num_entries: int):
        self.concrete_cudagraph_entries = {
            ("desc", i): object() for i in range(num_entries)
        }


@pytest.fixture
def capture_flag_guard():
    saved = monitor.cudagraph_capturing_enabled
    yield
    monitor.cudagraph_capturing_enabled = saved


class TestInvalidateAll:

    def test_clears_entries_and_reenables_capture(self, capture_flag_guard):
        w1, w2 = FakeWrapper(3), FakeWrapper(1)
        _cudagraph_wrappers.add(w1)
        _cudagraph_wrappers.add(w2)
        try:
            monitor.cudagraph_capturing_enabled = False
            cleared = invalidate_all_cudagraphs()
            assert cleared == 4
            assert not w1.concrete_cudagraph_entries
            assert not w2.concrete_cudagraph_entries
            # Lazy re-capture must be allowed again.
            assert monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w1)
            _cudagraph_wrappers.discard(w2)

    def test_noop_when_nothing_captured(self, capture_flag_guard):
        w = FakeWrapper(0)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            assert invalidate_all_cudagraphs() == 0
            # Nothing cleared -> no need to unfreeze capturing.
            assert not monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w)

    def test_registry_is_weak(self):
        before = len(_cudagraph_wrappers)
        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        del w
        import gc
        gc.collect()
        assert len(_cudagraph_wrappers) == before


class _ConstAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)

    def _compute_delta(self, h):
        return torch.zeros_like(h)


def _fake_worker_with_layers(num_layers=2):
    layers = []
    for _ in range(num_layers):
        layer = nn.Module()
        layer._reft_layer_idx = 0
        _init_multi_reft_state(layer, torch.device("cpu"), 8)
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    worker = SimpleNamespace(get_model=lambda: model,
                             model_runner=SimpleNamespace())
    worker._get_reft_manager = lambda: None
    return worker, layers


class TestWorkerIntegration:

    def test_load_adapter_invalidates_graphs(self, monkeypatch,
                                             capture_flag_guard):
        import vllm.reft as vllm_reft
        monkeypatch.setattr(
            vllm_reft, "reft_config_to_spec", lambda cfg: {
                "layer_indices": [0, 1],
                "sample_adapter": _ConstAdapter(),
                "position": "all",
            })
        worker, layers = _fake_worker_with_layers()
        for i, layer in enumerate(layers):
            layer._reft_layer_idx = i

        w = FakeWrapper(2)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            count = WorkerBase.load_reft_adapter(worker, 1, {},
                                                 position="decode")
            assert count == 2
            assert not w.concrete_cudagraph_entries
            assert monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w)

    def test_unload_adapter_invalidates_graphs(self, capture_flag_guard):
        worker, layers = _fake_worker_with_layers()
        for layer in layers:
            _add_adapter_to_layer(layer, 1, _ConstAdapter(), "all",
                                  torch.device("cpu"))
        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            removed = WorkerBase.unload_reft_adapter(worker, 1)
            assert removed == 2
            assert not w.concrete_cudagraph_entries
            assert monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w)

    def test_manager_activate_invalidates_graphs(self, monkeypatch,
                                                 capture_flag_guard):
        import vllm.reft as vllm_reft
        from vllm.reft.models import ReFTModel, ReFTModelManager
        monkeypatch.setattr(
            vllm_reft, "reft_config_to_spec", lambda cfg: {
                "layer_indices": [0],
                "sample_adapter": _ConstAdapter(),
                "position": "all",
            })
        layer = nn.Module()
        layer._reft_layer_idx = 0
        _init_multi_reft_state(layer, torch.device("cpu"), 8)
        manager = ReFTModelManager([layer], max_refts=2, max_cpu_refts=2,
                                   device=torch.device("cpu"),
                                   model_dtype=torch.float32)
        manager.add_adapter(
            ReFTModel(id=1, position="all", adapter_config={},
                      layer_indices=frozenset([0])))

        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            manager.activate_adapter(1)
            assert not w.concrete_cudagraph_entries
            assert monitor.cudagraph_capturing_enabled

            # Deactivation (LRU eviction) also changes the captured set.
            w.concrete_cudagraph_entries = {("desc", 0): object()}
            monitor.cudagraph_capturing_enabled = False
            manager.remove_adapter(1)
            assert not w.concrete_cudagraph_entries
            assert monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w)
