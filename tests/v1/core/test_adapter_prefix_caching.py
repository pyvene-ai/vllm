# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache block hashing must be adapter-aware.

KV content depends on every adapter that touches *prefill* hidden
states, so those adapter ids must be part of the block hash (reuse
across different prefill adapters would be silently wrong).  Decode-only
adapters never touch prefill KV, so they are deliberately EXCLUDED —
a decode-adapter request shares cached prefills with the base model.
"""

from types import SimpleNamespace

from vllm.lora.request import LoRARequest
from vllm.adaptation.request import ReFTRequest
from vllm.v1.core.kv_cache_utils import (generate_block_hash_extra_keys,
                                         need_extra_keys)


def _req(lora_request=None, decode_lora_request=None, reft_request=None,
         decode_reft_request=None, cache_salt=None):
    return SimpleNamespace(
        mm_features=[],
        lora_request=lora_request,
        decode_lora_request=decode_lora_request,
        reft_request=reft_request,
        decode_reft_request=decode_reft_request,
        cache_salt=cache_salt,
    )


def _lora(idx, position="all"):
    return LoRARequest(lora_name=f"l{idx}-{position}", lora_int_id=idx,
                       lora_path=f"/l/{idx}", lora_position=position)


def _reft(idx, position=None):
    return ReFTRequest(reft_name=f"r{idx}", reft_int_id=idx,
                       reft_path=f"/r/{idx}", reft_position=position)


def _keys(request):
    keys, _ = generate_block_hash_extra_keys(request, 0, 16, 0)
    return keys


class TestLoraHashKeys:

    def test_all_position_included(self):
        assert _keys(_req(lora_request=_lora(1, "all"))) == (1, )

    def test_prefill_position_included(self):
        assert _keys(_req(lora_request=_lora(1, "prefill"))) == (1, )

    def test_decode_position_excluded(self):
        # Decode-only LoRA never touches prefill KV: it must share the
        # prefix cache with the base model.
        assert _keys(_req(lora_request=_lora(1, "decode"))) is None

    def test_decode_slot_excluded(self):
        keys = _keys(_req(lora_request=_lora(1, "prefill"),
                          decode_lora_request=_lora(2, "decode")))
        assert keys == (1, )

    def test_base_request_matches_decode_only_request(self):
        assert _keys(_req()) == _keys(_req(lora_request=_lora(1, "decode")))


class TestReftHashKeys:

    def test_unknown_position_conservatively_included(self):
        # The adapter's position lives worker-side; without a declared
        # reft_position the hasher must assume it can touch prefill KV.
        assert _keys(_req(reft_request=_reft(5))) == ("reft:5", )

    def test_prefill_position_included(self):
        assert _keys(_req(reft_request=_reft(5, "prefill"))) == ("reft:5", )

    def test_decode_position_excluded(self):
        assert _keys(_req(reft_request=_reft(5, "decode"))) is None

    def test_decode_slot_follows_declared_position(self):
        # Pairing convention loads the decode slot's adapter with
        # position="decode"; declaring it on the request unlocks cache
        # sharing.  Undeclared stays conservative.
        keys = _keys(_req(reft_request=_reft(5, "prefill"),
                          decode_reft_request=_reft(6, "decode")))
        assert keys == ("reft:5", )
        keys = _keys(_req(reft_request=_reft(5, "prefill"),
                          decode_reft_request=_reft(6)))
        assert keys == ("reft:5", "reft:6")

    def test_different_prefill_adapters_hash_differently(self):
        a = _keys(_req(reft_request=_reft(5, "prefill")))
        b = _keys(_req(reft_request=_reft(7, "prefill")))
        assert a != b

    def test_lora_and_reft_keys_do_not_collide(self):
        lora_keys = _keys(_req(lora_request=_lora(5, "all")))
        reft_keys = _keys(_req(reft_request=_reft(5, "prefill")))
        assert lora_keys != reft_keys


class TestNeedExtraKeys:

    def test_reft_request_triggers_extra_keys(self):
        assert need_extra_keys(_req(reft_request=_reft(5)))

    def test_decode_reft_slot_triggers_extra_keys(self):
        assert need_extra_keys(_req(decode_reft_request=_reft(6)))

    def test_plain_request_does_not(self):
        assert not need_extra_keys(_req())


class TestReftRequestPositionField:

    def test_default_none(self):
        assert _reft(1).reft_position is None

    def test_roundtrip(self):
        import msgspec
        req = _reft(2, "decode")
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(req),
                                     type=ReFTRequest)
        assert out.reft_position == "decode"
