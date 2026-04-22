# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar,
                    Union)

import cloudpickle
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, run_method,
                        update_environment_variables,
                        warn_for_unimplemented_methods)
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

_R = TypeVar("_R")


@warn_for_unimplemented_methods
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        from vllm.platforms import current_platform
        self.current_platform = current_platform

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def refresh_reft_caches(self, reft_int_id: Optional[int] = None) -> None:
        """Recompute derived ReFT caches after adapter weights are updated.

        Called via ``collective_rpc("refresh_reft_caches")`` after TRL's
        ``sync_weights()`` pushes new adapter parameters through
        ``load_weights()``.

        Args:
            reft_int_id: If given, only refresh caches for this adapter.
                If None, refresh all adapters.
        """
        model = self.get_model()
        for layer in model.model.layers:
            if not hasattr(layer, "reft_adapters"):
                continue
            for str_id, adapter in layer.reft_adapters.items():
                if reft_int_id is not None and int(str_id) != reft_int_id:
                    continue
                if hasattr(adapter, "install_inference_caches"):
                    adapter.install_inference_caches()

    def sync_reft_weights(self, weight_dict: dict[int, dict[str, bytes]],
                          reft_int_id: int = 1,
                          refresh_caches: bool = True) -> int:
        """Load ReFT adapter weights for a specific adapter ID.

        Called via ``collective_rpc("sync_reft_weights",
        args=(weight_dict, reft_int_id))`` from pyreft's ``sync_to_vllm``.

        Args:
            weight_dict: Mapping from layer index to serialized state_dict.
                Values are ``{param_name: tensor}`` dicts with tensors on CPU.
            reft_int_id: Which adapter to update (default 1 for backward compat).
            refresh_caches: Whether to recompute inference caches after loading.

        Returns:
            Number of adapter layers synced.
        """
        model = self.get_model()
        key = str(reft_int_id)
        count = 0
        for idx, state_dict in weight_dict.items():
            layer = model.model.layers[idx]
            if not hasattr(layer, "reft_adapters"):
                continue
            adapter = layer.reft_adapters.get(key)
            if adapter is None:
                continue
            device = next(adapter.parameters()).device
            sd = {k: v.to(device) for k, v in state_dict.items()}
            adapter.load_state_dict(sd)
            count += 1
        if refresh_caches and count:
            self.refresh_reft_caches(reft_int_id)
        return count

    def load_reft_adapter(self, reft_int_id: int, adapter_config: dict,
                          position: str = "prefill") -> int:
        """Load a new ReFT adapter into all relevant layers.

        Called via ``collective_rpc("load_reft_adapter",
        args=(reft_int_id, adapter_config, position))``.

        Args:
            reft_int_id: Unique integer ID for this adapter (>= 1).
            adapter_config: Serializable config dict (from spec_to_reft_config).
            position: Position mode for this adapter.

        Returns:
            Number of layers the adapter was loaded into.
        """
        import copy as _copy
        from vllm.reft import reft_config_to_spec
        from vllm.reft.layer import (_prepare_adapter, _add_adapter_to_layer,
                                     _install_adapter_caches)

        spec = reft_config_to_spec(adapter_config)
        if spec is None:
            return 0

        model = self.get_model()
        count = 0
        for layer_idx in spec["layer_indices"]:
            layer = model.model.layers[layer_idx]
            if not hasattr(layer, "reft_adapters"):
                continue
            try:
                dev = next(layer.parameters()).device
            except StopIteration:
                dev = torch.device("cuda")
            source = spec.get("adapters", {}).get(layer_idx,
                                                   spec["sample_adapter"])
            model_dtype = torch.bfloat16
            adapter_copy = _prepare_adapter(source, dev, model_dtype)
            _add_adapter_to_layer(layer, reft_int_id, adapter_copy,
                                  position, dev)
            count += 1
        return count

    def unload_reft_adapter(self, reft_int_id: int) -> int:
        """Remove a ReFT adapter from all layers.

        Returns:
            Number of layers the adapter was removed from.
        """
        model = self.get_model()
        key = str(reft_int_id)
        count = 0
        for layer in model.model.layers:
            if not hasattr(layer, "reft_adapters"):
                continue
            if key in layer.reft_adapters:
                del layer.reft_adapters[key]
                layer._reft_adapter_positions.pop(reft_int_id, None)
                layer._reft_combined_masks.pop(reft_int_id, None)
                # Update backward compat reft_adapter reference
                if layer.reft_adapter is not None:
                    remaining = list(layer.reft_adapters.values())
                    layer.reft_adapter = remaining[0] if remaining else None
                count += 1
        return count

    def get_reft_debug_stats(self) -> dict:
        """Return per-layer ReFT debug stats from the model."""
        model = self.get_model()
        if hasattr(model, "get_reft_debug_stats"):
            return model.get_reft_debug_stats()
        return {}

    def get_reft_weight_fingerprints(self, layer_indices=None) -> dict:
        """Return adapter param/buffer fingerprints for diagnostic comparison."""
        model = self.get_model()
        if hasattr(model, "get_reft_weight_fingerprints"):
            return model.get_reft_weight_fingerprints(layer_indices)
        return {}

    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        raise NotImplementedError

    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        with self.current_platform.inference_mode():
            while True:
                output = self.execute_model(execute_model_req=None)
                if output is None:
                    return None

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rpc_rank: int = 0,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.worker: Optional[WorkerBase] = None
        self.vllm_config: Optional[VllmConfig] = None
        # do not store this `vllm_config`, `init_worker` will set the final
        # one. TODO: investigate if we can remove this field in
        # `WorkerWrapperBase`, `init_cached_hf_modules` should be
        # unnecessary now.
        if vllm_config.model_config is not None:
            # it can be None in tests
            trust_remote_code = vllm_config.model_config.trust_remote_code
            if trust_remote_code:
                # note: lazy import to avoid importing torch before initializing
                from vllm.utils import init_cached_hf_modules
                init_cached_hf_modules()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: List[Dict[str,
                                                                str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config")
        assert self.vllm_config is not None, (
            "vllm_config is required to initialize the worker")
        enable_trace_function_call_for_thread(self.vllm_config)

        from vllm.plugins import load_general_plugins
        load_general_plugins()

        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls)
        else:
            logger.warning(
                "passing worker_cls as a class object is strongly deprecated,"
                " as the serialization of class objects can be tricky and"
                " error-prone. To be safe, please keep the class in a separate"
                " module and pass the qualified name of the class as a string."
            )
            assert isinstance(self.vllm_config.parallel_config.worker_cls,
                              bytes)
            worker_class = cloudpickle.loads(
                self.vllm_config.parallel_config.worker_cls)
        if self.vllm_config.parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_extension_cls)
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}.")
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls, )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls, worker_class, extended_calls)
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)
            assert self.worker is not None

    def initialize_from_config(self, kv_cache_configs: List[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.rpc_rank]
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)
