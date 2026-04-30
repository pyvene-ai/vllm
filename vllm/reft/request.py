"""Per-request ReFT adapter identifier for multi-ReFT serving."""

import msgspec


class ReFTRequest(msgspec.Struct, frozen=True):
    """Identifies which ReFT adapter a request should use.

    Analogous to :class:`vllm.lora.request.LoRARequest` but for ReFT adapters.

    Attributes:
        reft_name: Human-readable adapter name.
        reft_int_id: Unique integer ID. **0 is reserved for "no adapter"
            (base model passthrough).**  Must be >= 1 for real adapters.
        reft_path: Path to the saved adapter checkpoint directory.
    """

    reft_name: str
    reft_int_id: int
    reft_path: str

    def __post_init__(self):
        if self.reft_int_id < 1:
            raise ValueError(
                f"reft_int_id must be >= 1 (0 is reserved for base model), "
                f"got {self.reft_int_id}")

    @property
    def adapter_id(self) -> int:
        return self.reft_int_id

    @property
    def name(self) -> str:
        return self.reft_name

    @property
    def path(self) -> str:
        return self.reft_path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReFTRequest):
            return NotImplemented
        return self.reft_name == other.reft_name

    def __hash__(self) -> int:
        return hash(self.reft_name)
