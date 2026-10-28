"""Token counting helpers using cached HuggingFace tokenizers."""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _tokenizer(model_id: str):
    """Load and cache an AutoTokenizer for the given model id.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        An ``AutoTokenizer`` instance for the model.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(model_id)


def count_tokens(model_id: str, text: str) -> int:
    """Return the token count of ``text`` for a local HuggingFace model.

    Non-raising: on any failure (model not found, offline, etc.) logs a
    WARNING and returns ``0``.

    Args:
        model_id: HuggingFace model identifier used to select the tokenizer.
        text: The text to tokenize.

    Returns:
        Number of tokens, or ``0`` on any failure.
    """
    try:
        tok = _tokenizer(model_id)
        return len(tok.encode(text))
    except Exception as exc:
        logger.warning("model_cost: token count failed for '%s': %s", model_id, exc)
        return 0
