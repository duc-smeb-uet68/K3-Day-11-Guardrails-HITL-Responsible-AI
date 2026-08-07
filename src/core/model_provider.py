"""Shared, non-secret model configuration for the VinBank lab.

All live model calls use GPT-4o mini through OpenRouter. Keeping this in one
place prevents individual agents from silently selecting another provider.
"""
from __future__ import annotations

import os
from typing import Any

from core.adk_compat import ADK_AVAILABLE


# OpenRouter's catalog slug is ``openai/gpt-4o-mini``. LiteLLM requires its
# provider prefix in front of that slug when it routes the request.
OPENROUTER_MODEL_SLUG = "openai/gpt-4o-mini"
LITELLM_OPENROUTER_MODEL = f"openrouter/{OPENROUTER_MODEL_SLUG}"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def get_openrouter_api_key(*, required: bool = False) -> str | None:
    """Return the OpenRouter key from the environment without logging it."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key
    if required:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for live model calls. "
            "Add it to .env or set it in the environment."
        )
    return None


def has_openrouter_api_key() -> bool:
    """Return whether live OpenRouter calls are configured."""
    return get_openrouter_api_key() is not None


def build_openrouter_model() -> Any:
    """Create the ADK model adapter for GPT-4o mini on OpenRouter.

    The compatibility stand-in remains a string on machines without Google ADK,
    so deterministic guardrail tests can run without optional model packages.
    A real ADK runtime requires LiteLLM, declared in requirements.txt.
    """
    if not ADK_AVAILABLE:
        return LITELLM_OPENROUTER_MODEL

    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as exc:  # pragma: no cover - depends on local install.
        raise RuntimeError(
            "OpenRouter support requires LiteLLM. Run: pip install -r requirements.txt"
        ) from exc

    return LiteLlm(
        model=LITELLM_OPENROUTER_MODEL,
        api_key=get_openrouter_api_key(required=True),
        api_base=OPENROUTER_API_BASE,
    )
