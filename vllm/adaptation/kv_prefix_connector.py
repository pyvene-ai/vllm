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
                if new_ids is None:
                    continue
                add_load(req_id, list(request.prompt_token_ids),
                         new_ids[0],
                         [f.identifier for f in request.mm_features])
                served.append(req_id)

        # Entries not scheduled this step stay pending for a later
        # step instead of tripping an assert.
        for req_id in served:
            self._requests_need_load.pop(req_id, None)
        return meta
