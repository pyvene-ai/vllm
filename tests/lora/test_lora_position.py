# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoRARequest.lora_position ("all" / "prefill" / "decode")."""

import msgspec
import pytest

from vllm.lora.request import LoRARequest


def _make(position=None, **kwargs):
    defaults = dict(lora_name="a", lora_int_id=1, lora_path="/fake/path")
    defaults.update(kwargs)
    if position is not None:
        defaults["lora_position"] = position
    return LoRARequest(**defaults)


def test_default_position_is_all():
    req = _make()
    assert req.lora_position == "all"


@pytest.mark.parametrize("position", ["all", "prefill", "decode"])
def test_valid_positions_accepted(position):
    req = _make(position)
    assert req.lora_position == position


@pytest.mark.parametrize("position",
                         ["", "Prefill", "DECODE", "first", "last", "both"])
def test_invalid_positions_rejected(position):
    with pytest.raises(ValueError, match="lora_position"):
        _make(position)


@pytest.mark.parametrize("position", ["all", "prefill", "decode"])
def test_msgspec_roundtrip_preserves_position(position):
    req = _make(position)
    encoded = msgspec.msgpack.encode(req)
    decoded = msgspec.msgpack.decode(encoded, type=LoRARequest)
    assert decoded.lora_position == position
    assert decoded.lora_int_id == req.lora_int_id


def test_invalid_id_still_rejected():
    with pytest.raises(ValueError, match="id must be > 0"):
        _make(lora_int_id=0)
