# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-aware LoRA mapping tests for InputBatch.make_lora_inputs.

Covers prefill-only, decode-only, and paired prefill+decode adapters on
the same request.  All tests run on CPU.

Phase semantics (exact complement):
  - A request is in the *prefill* phase while num_computed_tokens <
    num_prompt_tokens (includes the final prompt chunk, whose forward
    produces the first sampled token).
  - A request is in the *decode* phase once num_computed_tokens >=
    num_prompt_tokens.
  - A prefill-position adapter applies only during the prefill phase; a
    decode-position adapter applies only during the decode phase.  Their
    union covers every step exactly once.
"""

import numpy as np
import pytest
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

VOCAB_SIZE = 128
MAX_NUM_REQS = 16
MAX_MODEL_LEN = 512


def _make_batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=VOCAB_SIZE,
        block_sizes=[16],
    )


def _lora(idx: int, position: str = "all") -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter-{idx}-{position}",
        lora_int_id=idx,
        lora_path=f"/fake/{idx}",
        lora_position=position,
    )


_REQ_COUNTER = [0]


def _req(
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_output_tokens: int = 0,
    lora_request: LoRARequest = None,
    decode_lora_request: LoRARequest = None,
) -> CachedRequestState:
    _REQ_COUNTER[0] += 1
    return CachedRequestState(
        req_id=f"req-{_REQ_COUNTER[0]}",
        prompt_token_ids=list(range(num_prompt_tokens)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=[1] * num_output_tokens,
        lora_request=lora_request,
        decode_lora_request=decode_lora_request,
    )


def _mappings(batch: InputBatch, num_scheduled: list[int]):
    ns = np.array(num_scheduled, dtype=np.int32)
    prompt_mapping, token_mapping, active = batch.make_lora_inputs(ns)
    return prompt_mapping, token_mapping, active


class TestSingleAdapterPositions:

    def test_all_position_applies_everywhere(self):
        batch = _make_batch()
        # Request mid-prefill.
        batch.add_request(_req(10, 4, lora_request=_lora(1, "all")))
        # Request in decode.
        batch.add_request(
            _req(6, 7, num_output_tokens=2, lora_request=_lora(1, "all")))
        prompt, token, active = _mappings(batch, [4, 1])
        assert prompt == (1, 1)
        assert token == (1, 1, 1, 1, 1)
        assert {r.lora_int_id for r in active} == {1}

    def test_prefill_only_active_during_prefill(self):
        batch = _make_batch()
        batch.add_request(_req(10, 0, lora_request=_lora(1, "prefill")))
        prompt, token, _ = _mappings(batch, [10])
        assert prompt == (1, )
        assert token == tuple([1] * 10)

    def test_prefill_only_active_on_final_prompt_chunk(self):
        batch = _make_batch()
        # 4 of 10 prompt tokens remain: still prefill phase.
        batch.add_request(_req(10, 6, lora_request=_lora(1, "prefill")))
        prompt, token, _ = _mappings(batch, [4])
        assert prompt == (1, )
        assert token == (1, 1, 1, 1)

    def test_prefill_only_masked_during_decode(self):
        batch = _make_batch()
        batch.add_request(
            _req(10, 10, num_output_tokens=1,
                 lora_request=_lora(1, "prefill")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (0, )
        assert token == (0, )

    def test_decode_only_masked_during_prefill(self):
        batch = _make_batch()
        batch.add_request(_req(10, 0, lora_request=_lora(2, "decode")))
        prompt, token, _ = _mappings(batch, [10])
        assert prompt == (0, )
        assert token == tuple([0] * 10)

    def test_decode_only_masked_on_final_prompt_chunk(self):
        # Exact complement: the final prompt chunk (which samples the
        # first output token) still belongs to the prefill phase.
        batch = _make_batch()
        batch.add_request(_req(10, 9, lora_request=_lora(2, "decode")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (0, )
        assert token == (0, )

    def test_decode_only_active_during_decode(self):
        batch = _make_batch()
        batch.add_request(
            _req(10, 10, num_output_tokens=1, lora_request=_lora(2,
                                                                 "decode")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (2, )
        assert token == (2, )

    def test_decode_only_active_on_later_decode_steps(self):
        batch = _make_batch()
        batch.add_request(
            _req(10, 14, num_output_tokens=5, lora_request=_lora(2,
                                                                 "decode")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (2, )
        assert token == (2, )


class TestPairedAdapters:

    def test_pair_prefill_phase_uses_prefill_adapter(self):
        batch = _make_batch()
        batch.add_request(
            _req(10,
                 0,
                 lora_request=_lora(1, "prefill"),
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, active = _mappings(batch, [10])
        assert prompt == (1, )
        assert token == tuple([1] * 10)
        assert {r.lora_int_id for r in active} == {1, 2}

    def test_pair_decode_phase_uses_decode_adapter(self):
        batch = _make_batch()
        batch.add_request(
            _req(10,
                 10,
                 num_output_tokens=1,
                 lora_request=_lora(1, "prefill"),
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, active = _mappings(batch, [1])
        assert prompt == (2, )
        assert token == (2, )
        assert {r.lora_int_id for r in active} == {1, 2}

    def test_pair_exact_complement_boundary(self):
        batch = _make_batch()
        # One token of prompt left: prefill adapter owns this step.
        batch.add_request(
            _req(10,
                 9,
                 lora_request=_lora(1, "prefill"),
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (1, )
        assert token == (1, )

        # After the prompt completes, the decode adapter takes over.
        batch2 = _make_batch()
        batch2.add_request(
            _req(10,
                 10,
                 num_output_tokens=1,
                 lora_request=_lora(1, "prefill"),
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, _ = _mappings(batch2, [1])
        assert prompt == (2, )
        assert token == (2, )

    def test_pair_chunked_prefill_stays_on_prefill_adapter(self):
        batch = _make_batch()
        batch.add_request(
            _req(100,
                 32,
                 lora_request=_lora(1, "prefill"),
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, _ = _mappings(batch, [32])
        assert prompt == (1, )
        assert token == tuple([1] * 32)


class TestMixedBatch:

    def test_mixed_configs_and_phases(self):
        batch = _make_batch()
        # Req A: "all" adapter 1, decoding.
        batch.add_request(
            _req(4, 4, num_output_tokens=1, lora_request=_lora(1, "all")))
        # Req B: prefill-only adapter 2, mid-prefill.
        batch.add_request(_req(8, 0, lora_request=_lora(2, "prefill")))
        # Req C: decode-only adapter 3, decoding.
        batch.add_request(
            _req(4, 5, num_output_tokens=2, lora_request=_lora(3, "decode")))
        # Req D: pair (4 prefill / 5 decode), decoding.
        batch.add_request(
            _req(4,
                 4,
                 num_output_tokens=1,
                 lora_request=_lora(4, "prefill"),
                 decode_lora_request=_lora(5, "decode")))
        # Req E: no adapter, mid-prefill.
        batch.add_request(_req(6, 0))

        prompt, token, active = _mappings(batch, [1, 8, 1, 1, 6])
        assert prompt == (1, 2, 3, 5, 0)
        expected_tokens = ((1, ) + tuple([2] * 8) + (3, ) + (5, ) +
                           tuple([0] * 6))
        assert token == expected_tokens
        assert {r.lora_int_id for r in active} == {1, 2, 3, 4, 5}

    def test_no_position_features_fast_path(self):
        # Pure "all" adapters: mapping must match the naive repeat.
        batch = _make_batch()
        batch.add_request(_req(4, 0, lora_request=_lora(1, "all")))
        batch.add_request(_req(4, 4, num_output_tokens=1))
        prompt, token, _ = _mappings(batch, [4, 1])
        assert prompt == (1, 0)
        assert token == (1, 1, 1, 1, 0)


class TestBookkeeping:

    def test_remove_request_clears_decode_mapping(self):
        batch = _make_batch()
        req = _req(10,
                   0,
                   lora_request=_lora(1, "prefill"),
                   decode_lora_request=_lora(2, "decode"))
        batch.add_request(req)
        batch.remove_request(req.req_id)
        assert 1 not in batch.lora_id_to_lora_request
        assert 2 not in batch.lora_id_to_lora_request
        assert 1 not in batch.lora_id_to_request_ids
        assert 2 not in batch.lora_id_to_request_ids

    def test_remove_one_of_two_requests_sharing_decode_adapter(self):
        batch = _make_batch()
        r1 = _req(10,
                  0,
                  lora_request=_lora(1, "prefill"),
                  decode_lora_request=_lora(2, "decode"))
        r2 = _req(10,
                  0,
                  lora_request=_lora(1, "prefill"),
                  decode_lora_request=_lora(2, "decode"))
        batch.add_request(r1)
        batch.add_request(r2)
        batch.remove_request(r1.req_id)
        assert 2 in batch.lora_id_to_lora_request
        assert batch.lora_id_to_request_ids[2] == {r2.req_id}

    def test_condense_preserves_decode_mapping(self):
        batch = _make_batch()
        r1 = _req(10, 10, num_output_tokens=1, lora_request=_lora(1, "all"))
        r2 = _req(10, 10, num_output_tokens=1, lora_request=_lora(2, "all"))
        r3 = _req(10,
                  10,
                  num_output_tokens=1,
                  lora_request=_lora(3, "prefill"),
                  decode_lora_request=_lora(4, "decode"))
        batch.add_request(r1)
        batch.add_request(r2)
        batch.add_request(r3)
        batch.remove_request(r2.req_id)
        batch.condense()
        # r3 moved into r2's old slot; its decode adapter must follow.
        prompt, token, _ = _mappings(batch, [1, 1])
        r3_index = batch.req_id_to_index[r3.req_id]
        assert prompt[r3_index] == 4
        assert token[r3_index] == 4

    def test_swap_states_preserves_decode_mapping(self):
        batch = _make_batch()
        r1 = _req(10, 10, num_output_tokens=1, lora_request=_lora(1, "all"))
        r2 = _req(10,
                  10,
                  num_output_tokens=1,
                  lora_request=_lora(3, "prefill"),
                  decode_lora_request=_lora(4, "decode"))
        batch.add_request(r1)
        batch.add_request(r2)
        batch.swap_states(0, 1)
        prompt, token, _ = _mappings(batch, [1, 1])
        r2_index = batch.req_id_to_index[r2.req_id]
        r1_index = batch.req_id_to_index[r1.req_id]
        assert prompt[r2_index] == 4
        assert prompt[r1_index] == 1


class TestDecodeLoraValidation:

    def test_decode_slot_must_have_decode_position(self):
        batch = _make_batch()
        with pytest.raises(ValueError, match="decode"):
            batch.add_request(
                _req(10,
                     0,
                     lora_request=_lora(1, "prefill"),
                     decode_lora_request=_lora(2, "prefill")))

    def test_primary_must_be_prefill_when_paired(self):
        # A primary "all" adapter would overlap the decode adapter on
        # decode steps; reject the combination.
        batch = _make_batch()
        with pytest.raises(ValueError, match="prefill"):
            batch.add_request(
                _req(10,
                     0,
                     lora_request=_lora(1, "all"),
                     decode_lora_request=_lora(2, "decode")))

    def test_decode_slot_without_primary_is_allowed(self):
        batch = _make_batch()
        batch.add_request(
            _req(10,
                 10,
                 num_output_tokens=1,
                 decode_lora_request=_lora(2, "decode")))
        prompt, token, active = _mappings(batch, [1])
        assert prompt == (2, )
        assert token == (2, )
        assert {r.lora_int_id for r in active} == {2}
