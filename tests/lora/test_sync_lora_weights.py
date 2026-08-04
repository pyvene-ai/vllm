# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for in-memory LoRA weight sync (training-time weight updates).

Exercises WorkerBase.sync_lora_weights against a real
LRUCacheWorkerLoRAManager + LRUCacheLoRAModelManager on CPU:

  - registering and activating a synced adapter,
  - two concurrently synced adapters (e.g. prefill-only + decode-only
    pair trained together),
  - re-syncing an adapter mid-training must refresh the GPU-slot
    buffers AND invalidate the cached punica mapping,
  - synced adapters have no disk backing, so they must never be
    LRU-evicted (a later reload from their fake path would fail).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.models import LoRAMapping, LRUCacheLoRAModelManager
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.worker.worker_base import WorkerBase

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}
EMBEDDING_PADDING_MODULES = ["lm_head"]

RANK = 8
# Modules of the dummy_model fixture we attach LoRA to, with their
# (in_features, out_features).
TARGET_MODULES = {
    "layer1.dense1": (100, 10),
    "dense2": (100, 50),
}


def _make_worker_manager(dummy_model,
                         max_loras: int = 2,
                         max_cpu_loras: int = 2) -> LRUCacheWorkerLoRAManager:
    lora_config = LoRAConfig(
        max_lora_rank=RANK,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
        lora_dtype=torch.float32,
    )
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4
    vllm_config.scheduler_config.max_num_batched_tokens = 128
    vllm_config.model_config.get_vocab_size.return_value = 512
    vllm_config.lora_config = lora_config
    text_config = MagicMock()
    text_config.max_position_embeddings = 4096
    vllm_config.model_config.hf_config.get_text_config.return_value = \
        text_config

    worker_mgr = LRUCacheWorkerLoRAManager(vllm_config, torch.device("cpu"),
                                           EMBEDDING_MODULES,
                                           EMBEDDING_PADDING_MODULES)
    worker_mgr.create_lora_manager(dummy_model)
    return worker_mgr


def _fake_worker(worker_mgr) -> SimpleNamespace:
    """A minimal stand-in for a Worker exposing model_runner.lora_manager."""
    return SimpleNamespace(
        model_runner=SimpleNamespace(lora_manager=worker_mgr))


def _peft_state_dict(fill_value: float) -> dict[str, torch.Tensor]:
    tensors = {}
    for module, (in_f, out_f) in TARGET_MODULES.items():
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.full(
            (RANK, in_f), fill_value, dtype=torch.float32)
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.full(
            (out_f, RANK), fill_value, dtype=torch.float32)
    return tensors


PEFT_CONFIG = {
    "r": RANK,
    "lora_alpha": 16,
    "target_modules": ["dense1", "dense2"],
}


def _sync(worker, fill_value: float, lora_int_id: int) -> bool:
    return WorkerBase.sync_lora_weights(worker,
                                        _peft_state_dict(fill_value),
                                        PEFT_CONFIG,
                                        lora_int_id=lora_int_id)


def _slot_of(model_mgr, lora_id: int) -> int:
    assert lora_id in model_mgr.lora_index_to_id, (
        f"adapter {lora_id} not in any active slot: "
        f"{model_mgr.lora_index_to_id}")
    return model_mgr.lora_index_to_id.index(lora_id)


def _stacked_fill_value(model_mgr, lora_id: int) -> float:
    """The distinctive fill constant found in a module's GPU-slot buffer."""
    slot = _slot_of(model_mgr, lora_id)
    module = model_mgr.modules["layer1.dense1"]
    slot_weights = module.lora_a_stacked[0][slot]
    nonzero = slot_weights[slot_weights != 0]
    assert nonzero.numel() > 0, "slot buffer is all zeros"
    values = torch.unique(nonzero)
    assert values.numel() == 1, f"expected uniform fill, got {values}"
    return values.item()


@pytest.fixture
def worker_and_mgr(dist_init, dummy_model):
    worker_mgr = _make_worker_manager(dummy_model)
    return _fake_worker(worker_mgr), worker_mgr


class TestBasicSync:

    def test_sync_registers_and_activates(self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        assert 1 in worker_mgr.list_adapters()
        model_mgr = worker_mgr._adapter_manager
        assert _stacked_fill_value(model_mgr, 1) == pytest.approx(0.5)

    def test_sync_returns_false_without_lora_manager(self):
        worker = SimpleNamespace(model_runner=SimpleNamespace())
        assert not _sync(worker, 0.5, lora_int_id=1)

    def test_sync_two_adapters_concurrently(self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        assert _sync(worker, 0.25, lora_int_id=2)

        assert worker_mgr.list_adapters() == {1, 2}
        model_mgr = worker_mgr._adapter_manager
        assert _slot_of(model_mgr, 1) != _slot_of(model_mgr, 2)
        assert _stacked_fill_value(model_mgr, 1) == pytest.approx(0.5)
        assert _stacked_fill_value(model_mgr, 2) == pytest.approx(0.25)


class TestResync:

    def test_resync_updates_gpu_slot_weights(self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        # Training step happened; push new weights for the same id.
        assert _sync(worker, 0.125, lora_int_id=1)
        model_mgr = worker_mgr._adapter_manager
        assert _stacked_fill_value(model_mgr, 1) == pytest.approx(0.125)

    def test_resync_both_adapters_of_a_pair(self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        assert _sync(worker, 0.25, lora_int_id=2)
        assert _sync(worker, 0.0625, lora_int_id=1)
        assert _sync(worker, 0.03125, lora_int_id=2)
        model_mgr = worker_mgr._adapter_manager
        assert _stacked_fill_value(model_mgr, 1) == pytest.approx(0.0625)
        assert _stacked_fill_value(model_mgr, 2) == pytest.approx(0.03125)

    def test_resync_invalidates_cached_punica_mapping(self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        model_mgr = worker_mgr._adapter_manager

        mapping = LoRAMapping((1, ), (1, ), is_prefill=True)
        model_mgr.set_adapter_mapping(mapping)
        assert model_mgr._last_mapping is not None

        # Re-sync may reassign slots; the cached mapping must not be
        # trusted afterwards, otherwise punica keeps routing tokens to
        # the old slot.
        assert _sync(worker, 0.125, lora_int_id=1)
        assert model_mgr._last_mapping is None


class TestEvictionProtection:

    def test_synced_adapters_survive_lru_eviction(self, dist_init,
                                                  dummy_model):
        worker_mgr = _make_worker_manager(dummy_model,
                                          max_loras=2,
                                          max_cpu_loras=2)
        worker = _fake_worker(worker_mgr)
        assert _sync(worker, 0.5, lora_int_id=1)
        assert _sync(worker, 0.25, lora_int_id=2)

        model_mgr = worker_mgr._adapter_manager
        assert isinstance(model_mgr, LRUCacheLoRAModelManager)
        # Capacity is full; an eviction attempt (as triggered by a
        # disk-backed adapter being added) must NOT silently drop a
        # synced adapter — synced adapters cannot be reloaded from disk.
        with pytest.raises(RuntimeError):
            model_mgr.remove_oldest_adapter()
        assert worker_mgr.list_adapters() == {1, 2}

    def test_serving_synced_adapter_does_not_reload_from_disk(
            self, worker_and_mgr):
        worker, worker_mgr = worker_and_mgr
        assert _sync(worker, 0.5, lora_int_id=1)
        assert _sync(worker, 0.25, lora_int_id=2)

        # The batch references both adapters by fake paths (as LoRARequest
        # objects created by the serving layer).  This must not attempt
        # any disk IO because both ids are registered in memory.
        requests = {
            LoRARequest(lora_name="prefill-adapter",
                        lora_int_id=1,
                        lora_path="/nonexistent/path1",
                        lora_position="prefill"),
            LoRARequest(lora_name="decode-adapter",
                        lora_int_id=2,
                        lora_path="/nonexistent/path2",
                        lora_position="decode"),
        }
        mapping = LoRAMapping((1, 2), (1, 2), is_prefill=False)
        worker_mgr.set_active_adapters(requests, mapping)
        assert worker_mgr.list_adapters() == {1, 2}

    def test_resync_preserves_eviction_protection(self, dist_init,
                                                  dummy_model):
        worker_mgr = _make_worker_manager(dummy_model,
                                          max_loras=2,
                                          max_cpu_loras=2)
        worker = _fake_worker(worker_mgr)
        assert _sync(worker, 0.5, lora_int_id=1)
        assert _sync(worker, 0.25, lora_int_id=2)
        # Re-sync (remove + re-add) must re-establish the protection.
        assert _sync(worker, 0.125, lora_int_id=1)

        model_mgr = worker_mgr._adapter_manager
        with pytest.raises(RuntimeError):
            model_mgr.remove_oldest_adapter()
        assert worker_mgr.list_adapters() == {1, 2}
