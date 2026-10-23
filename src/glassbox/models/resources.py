"""Memory checks and the practical model choice for this machine."""

from __future__ import annotations

QWEN_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"

# Published parameter counts, used only for a load-time memory estimate.
PARAMETER_COUNTS: dict[str, int] = {
    QWEN_0_5B: 494_000_000,
    QWEN_1_5B: 1_540_000_000,
    QWEN_3B: 3_090_000_000,
}

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}

# Activations, allocator overhead, and the tokenizer sit on top of the weights.
_WEIGHT_OVERHEAD = 1.35
_HEADROOM_BYTES = 1_000_000_000
_PREFERRED_ORDER = (QWEN_3B, QWEN_1_5B, QWEN_0_5B)


class ModelMemoryError(MemoryError):
    """The selected weights do not fit in the memory available right now."""


def available_ram_bytes() -> int:
    """Return MemAvailable from ``/proc/meminfo``."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError as exc:
        raise ModelMemoryError("Could not read available memory.") from exc
    raise ModelMemoryError("MemAvailable is missing from /proc/meminfo.")


def estimate_weight_bytes(model_name: str, dtype_name: str) -> int | None:
    """Estimate resident weight bytes, or ``None`` when the model size is unknown."""
    count = PARAMETER_COUNTS.get(model_name)
    width = DTYPE_BYTES.get(dtype_name)
    if count is None or width is None:
        return None
    return int(count * width * _WEIGHT_OVERHEAD)


def resolve_dtype(model_name: str, requested: str, available_bytes: int) -> str:
    """Pick float32 when it fits, otherwise bfloat16.

    ``requested`` may be ``auto``, ``float32``, ``float16``, or ``bfloat16``.
    A named dtype is returned as-is; the caller still refuses a load that does
    not fit.
    """
    if requested != "auto":
        if requested not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype {requested!r}")
        return requested
    float32 = estimate_weight_bytes(model_name, "float32")
    if float32 is not None and available_bytes >= float32 + _HEADROOM_BYTES:
        return "float32"
    return "bfloat16"


def recommended_model(available_bytes: int) -> tuple[str, str]:
    """Return the largest Qwen2.5 instruct model that leaves 1 GiB of headroom.

    The estimate uses bfloat16 weights. Float32 is chosen later, and only for a
    model whose full-precision weights also fit.
    """
    for model_name in _PREFERRED_ORDER:
        needed = estimate_weight_bytes(model_name, "bfloat16")
        assert needed is not None
        if available_bytes >= needed + _HEADROOM_BYTES:
            gib = available_bytes / (1024**3)
            return model_name, (
                f"{model_name} is the largest Qwen2.5 instruct model whose "
                f"bfloat16 weights fit in {gib:.1f} GiB available RAM with 1 GiB headroom."
            )
    return QWEN_0_5B, (
        f"{QWEN_0_5B} is the smallest supported Qwen candidate. "
        "Available RAM is still below the estimate, so loading may fail."
    )


def assert_weights_fit(model_name: str, dtype_name: str, available_bytes: int) -> None:
    """Raise before a load that is estimated to exhaust RAM."""
    needed = estimate_weight_bytes(model_name, dtype_name)
    if needed is None:
        return
    if available_bytes < needed + _HEADROOM_BYTES:
        raise ModelMemoryError(
            f"{model_name} in {dtype_name} needs about {needed / (1024**3):.1f} GiB "
            f"plus 1 GiB headroom, and {available_bytes / (1024**3):.1f} GiB is available. "
            f"Use {QWEN_0_5B} for this machine, or free RAM and retry."
        )
