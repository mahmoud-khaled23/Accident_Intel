"""Per-model USD pricing table and cost computation helpers."""
import logging

logger = logging.getLogger(__name__)

PRICING: dict[str, dict] = {
    "gpt-4o-mini": {
        "input_per_1m_usd": 0.150,
        "output_per_1m_usd": 0.600,
        "source": "api",
    },
    "gemini-1.5-flash": {
        "input_per_1m_usd": 0.075,
        "output_per_1m_usd": 0.300,
        "source": "api",
    },
    "gemini-flash-lite-latest": {
        "input_per_1m_usd": 0.100,
        "output_per_1m_usd": 0.400,
        "source": "api",
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "local",
    },
    "BAAI/bge-m3": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "local",
    },
    "BAAI/bge-reranker-v2-m3": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "local",
    },
    "yolo11x": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "compute",
    },
    "damage_segmentation": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "compute",
    },
    "parts_segmentation": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "compute",
    },
    "default/unknown": {
        "input_per_1m_usd": 0.0,
        "output_per_1m_usd": 0.0,
        "source": "api",
    },
}

_STRIP_PREFIXES = ("openai/", "google/", "anthropic/")


def _normalize_prefix(model_id: str) -> str:
    """Strip leading provider prefixes from a model id.

    Args:
        model_id: Raw model identifier, potentially prefixed with
            ``openai/``, ``google/``, or ``anthropic/``.

    Returns:
        The model id with any known provider prefix removed.
    """
    for prefix in _STRIP_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def _lookup_pricing(model_id: str) -> dict:
    """Look up the pricing entry for a model id, with prefix normalization.

    Args:
        model_id: Raw model identifier.

    Returns:
        The matching PRICING entry, or ``PRICING["default/unknown"]`` on miss.
    """
    norm = _normalize_prefix(model_id)
    return PRICING.get(norm, PRICING["default/unknown"])


def compute_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the USD cost for a model call.

    Normalizes the model id, looks up its pricing, and returns the total cost
    rounded to 6 decimal places.  Logs a WARNING for unknown model ids.

    Args:
        model_id: The model identifier (provider prefix is stripped).
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.

    Returns:
        The computed cost in USD, rounded to 6 decimal places.
    """
    norm = _normalize_prefix(model_id)
    if norm not in PRICING:
        logger.warning("model_cost: unknown model id '%s', cost recorded as 0", model_id)
    entry = PRICING.get(norm, PRICING["default/unknown"])
    cost = (
        prompt_tokens / 1_000_000 * entry["input_per_1m_usd"]
        + completion_tokens / 1_000_000 * entry["output_per_1m_usd"]
    )
    return round(cost, 6)
