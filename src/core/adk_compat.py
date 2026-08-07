"""Small ADK compatibility boundary for deterministic local policy tests.

The assignment uses Google ADK in production.  Public policy tests should also
be able to import and exercise deterministic guards on a machine where optional
model dependencies have not yet been installed.  This module exposes the real
ADK objects when available and deliberately tiny data-only stand-ins otherwise.
The stand-ins cannot call a model or a tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


try:  # pragma: no cover - depends on the student's installed environment.
    from google.adk import runners
    from google.adk.agents import llm_agent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.plugins import base_plugin
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - behaviour is covered through guards.
    ADK_AVAILABLE = False

    @dataclass
    class _Part:
        text: str | None = None

        @classmethod
        def from_text(cls, *, text: str):
            return cls(text=text)

    @dataclass
    class _Content:
        role: str | None = None
        parts: list[_Part] = field(default_factory=list)

    class _BasePlugin:
        def __init__(self, *, name: str | None = None):
            self.name = name or self.__class__.__name__

    @dataclass
    class InvocationContext:
        user_id: str | None = None

    class _LlmAgent:
        """Metadata-only placeholder; it never makes a network request."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _InMemoryRunner:
        """Placeholder used only so judge initialization can fail closed later."""

        def __init__(self, *, agent, app_name: str, plugins=None):
            self.agent = agent
            self.app_name = app_name
            self.plugins = list(plugins or [])

    types = SimpleNamespace(Content=_Content, Part=_Part)
    base_plugin = SimpleNamespace(BasePlugin=_BasePlugin)
    llm_agent = SimpleNamespace(LlmAgent=_LlmAgent)
    runners = SimpleNamespace(InMemoryRunner=_InMemoryRunner)
