"""Per-request adapter identifier for multi-adapter serving."""

from typing import Optional

import msgspec


class AdapterRequest(msgspec.Struct, frozen=True):
    """Identifies which adapter a request should use.

    Analogous to :class:`vllm.lora.request.LoRARequest` but for adapters.

    Attributes:
        adapter_name: Human-readable adapter name.
        adapter_int_id: Unique integer ID. **0 is reserved for "no adapter"
            (base model passthrough).**  Must be >= 1 for real adapters.
        adapter_path: Path to the saved adapter checkpoint directory.
        adapter_position: Optional declaration of the adapter's position
            mode (positions live on the loaded adapter, worker-side).
            Used by prefix caching: adapters declared ``"decode"`` never
            touch prefill KV, so their requests share cached prefills
            with the base model; undeclared positions are conservatively
            assumed to affect prefill.
    """

    adapter_name: str
    adapter_int_id: int
    adapter_path: str
    adapter_position: Optional[str] = None

    def __post_init__(self):
        if self.adapter_int_id < 1:
            raise ValueError(
                f"adapter_int_id must be >= 1 (0 is reserved for base model), "
                f"got {self.adapter_int_id}")

    @property
    def adapter_id(self) -> int:
        return self.adapter_int_id

    @property
    def name(self) -> str:
        return self.adapter_name

    @property
    def path(self) -> str:
        return self.adapter_path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterRequest):
            return NotImplemented
        # identity includes the int id: N-member requests may carry
        # members sharing a name, and set-dedup by name alone would
        # silently drop them
        return (self.adapter_name == other.adapter_name
                and self.adapter_int_id == other.adapter_int_id)

    def __hash__(self) -> int:
        return hash((self.adapter_name, self.adapter_int_id))
