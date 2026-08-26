# SPDX-License-Identifier: Apache-2.0
"""Per-request KV prefix injection — the attn_kv-site serving member.

Serves precomputed, re-rotated KV rows as the first ``prefix_len``
positions of a request's paged cache. The prompt carries
``prefix_len`` placeholder token ids at the front (never embedded:
the scheduler counts them as externally computed); the row store is
keyed by the md5 of those placeholder ids, so a unique placeholder
tag per request selects that request's rows.

Unlike SharedStorageConnector (which matches/loads the whole aligned
prompt), this connector matches and injects exactly the first
``prefix_len`` positions and computes everything after them normally.
``prefix_len`` must be a multiple of the cache block size.
"""
import os
from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (  # noqa: E501
    ReqMeta, SharedStorageConnector, SharedStorageConnectorMetadata)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class PrefixInjectionConnector(SharedStorageConnector):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        transfer_config = vllm_config.kv_transfer_config
        self._prefix_len = int(
            transfer_config.get_from_extra_config("prefix_len", 0))
        if self._prefix_len % self._block_size != 0:
            raise ValueError(
                f"prefix_len {self._prefix_len} must be a multiple of "
                f"the cache block size {self._block_size}")
        self._pending_age: dict[str, int] = {}
        self._max_pending_steps = int(
            transfer_config.get_from_extra_config(
                "max_pending_steps", 100000))

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # Validate the store before the parent injects: fail with
        # request context instead of a bare safetensors error, and
        # never let a wrong-sized store write partial rows.
        import safetensors.torch
        metadata = self._get_connector_metadata()
        for request in metadata.requests:
            if request.is_store:
                continue
            folder = self._generate_foldername_debug(
                request.token_ids, request.mm_hashes,
                create_folder=False)
            if not os.path.exists(folder):
                raise FileNotFoundError(
                    f"prefix injection: row store folder missing at "
                    f"load time: {folder}")
            n_rows = len(request.slot_mapping)
            probe = os.path.join(
                folder, "model.layers.0.self_attn.attn.safetensors")
            kv = safetensors.torch.load_file(probe)["kv_cache"]
            if kv.shape[1] != n_rows:
                raise RuntimeError(
                    f"prefix injection: store {folder} has "
                    f"{kv.shape[1]} rows; request needs {n_rows} "
                    "(prefix_len mismatch between store and config)")
        super().start_load_kv(forward_context, **kwargs)

    def _found_match_for_request(self, request: "Request") -> bool:
        n = self._prefix_len
        if n == 0 or len(request.prompt_token_ids) <= n:
            return False
        foldername = self._generate_foldername_debug(
            torch.tensor(request.prompt_token_ids)[:n],
            [f.identifier for f in request.mm_features],
            create_folder=False)
        return os.path.exists(foldername)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        if num_computed_tokens >= self._prefix_len:
            return 0, False
        if not self._found_match_for_request(request):
            return 0, False
        return self._prefix_len - num_computed_tokens, False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> "SharedStorageConnectorMetadata":
        meta = SharedStorageConnectorMetadata()
        n = self._prefix_len
        nblk = n // self._block_size

        def add_load(req_id, prompt_ids, block_ids, mm_hashes):
            block_ids_t = torch.tensor(block_ids[:nblk])
            offs = torch.arange(0, self._block_size)
            slot_mapping = (offs.reshape(1, -1) +
                            block_ids_t.reshape(-1, 1) *
                            self._block_size).flatten()[:n]
            meta.requests.append(
                ReqMeta(token_ids=torch.tensor(prompt_ids)[:n],
                        slot_mapping=slot_mapping,
                        is_store=False,
                        mm_hashes=mm_hashes))

        served: list[str] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                if len(new_req.block_ids[0]) < nblk:
                    raise RuntimeError(
                        f"prefix injection: request {new_req.req_id} "
                        f"scheduled with {len(new_req.block_ids[0])} "
                        f"blocks; prefix needs {nblk}")
                add_load(new_req.req_id, new_req.prompt_token_ids,
                         new_req.block_ids[0],
                         [f.identifier for f in new_req.mm_features])
                served.append(new_req.req_id)

        # Resumed-from-preemption requests may appear anywhere in
        # scheduled_cached_reqs — scan them all (the parent's
        # break-on-first-non-resumed assumption does not survive
        # heavy preemption under large prefixes).
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if not cached_reqs.resumed_from_preemption[i]:
                continue
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                new_ids = cached_reqs.new_block_ids[i]
                if new_ids is None or len(new_ids[0]) < nblk:
                    raise RuntimeError(
                        f"prefix injection: resumed request {req_id} "
                        f"has block ids {new_ids and len(new_ids[0])} "
                        f"< prefix blocks {nblk}")
                add_load(req_id, list(request.prompt_token_ids),
                         new_ids[0],
                         [f.identifier for f in request.mm_features])
                served.append(req_id)

        for req_id in served:
            self._requests_need_load.pop(req_id, None)
            self._pending_age.pop(req_id, None)

        # Strict accounting for what remains: a pending entry is
        # legitimate ONLY if this step's schedule did not mention it
        # at all (its allocation was registered but it runs in a
        # later step, or it awaits resumption). Anything mentioned
        # but unserved is an inconsistency; anything pending
        # "forever" is starvation. Both fail loudly.
        if self._requests_need_load:
            mentioned = ({r.req_id for r in
                          scheduler_output.scheduled_new_reqs}
                         | set(cached_reqs.req_ids))
            for req_id in self._requests_need_load:
                if req_id in mentioned:
                    raise RuntimeError(
                        f"prefix injection: request {req_id} was "
                        "scheduled this step but its prefix load "
                        "could not be built — scheduler/connector "
                        "state is inconsistent")
                age = self._pending_age.get(req_id, 0) + 1
                self._pending_age[req_id] = age
                if age > self._max_pending_steps:
                    raise RuntimeError(
                        f"prefix injection: request {req_id} pending "
                        f"for {age} scheduler steps without being "
                        "scheduled — starvation or leaked state")
        return meta
