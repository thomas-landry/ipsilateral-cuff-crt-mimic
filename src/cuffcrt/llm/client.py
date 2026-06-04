"""Chat-completion clients for the local MedGemma (oMLX) server.

Two clients implement the same minimal protocol:

- :class:`OMLXClient` wraps the ``openai`` SDK and talks to a local,
  OpenAI-compatible server (oMLX serving MedGemma). The ``openai`` package is
  imported lazily so the rest of the package (and its tests) does not require
  the ``llm`` extra.
- :class:`StubClient` returns canned, well-formed JSON in process. It performs
  no network or server access and powers the ``--dry-run`` path and the tests.

Both return a plain string (the model's message content). Decoding parameters
(temperature, ``max_tokens``, seed) are passed per call so the caller controls
determinism.

Credentials come from the environment (``OMLX_API_KEY``, ``OMLX_BASE_URL``).
The key is never logged, printed, or written to any output artifact.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, cast

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "mlx-community/medgemma-1.5-4b-it-bf16"
# oMLX is local and does not authenticate; the SDK still requires a non-empty
# key, so this placeholder is used only when OMLX_API_KEY is unset.
_LOCAL_PLACEHOLDER_KEY = "not-needed-local"

# A chat-message payload: a list of role/content dicts. The ``content`` value is
# either a plain string (text turn) or a list of typed parts (e.g. a text part
# plus an ``image_url`` part for the adjudicate mode).
ChatMessages = list[dict[str, Any]]


class ChatClient(Protocol):
    """Minimal chat-completion protocol shared by the real and stub clients."""

    model: str

    def complete(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        """Return the model's message-content string for ``messages``."""
        ...


def resolve_base_url(override: str | None = None) -> str:
    """Resolve the server base URL: explicit override, then env, then default."""
    if override:
        return override
    return os.environ.get("OMLX_BASE_URL") or DEFAULT_BASE_URL


def _resolve_api_key() -> str:
    """Read ``OMLX_API_KEY`` from the environment; never log or return it elsewhere."""
    return os.environ.get("OMLX_API_KEY") or _LOCAL_PLACEHOLDER_KEY


class OMLXClient:
    """OpenAI-SDK client pointed at a local oMLX server.

    Parameters
    ----------
    model : str
        Served model id (default :data:`DEFAULT_MODEL`).
    base_url : str, optional
        Server base URL. Falls back to ``OMLX_BASE_URL`` then
        :data:`DEFAULT_BASE_URL`.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "The 'openai' package is required for live inference. "
                "Install the optional extra: uv sync --extra llm"
            ) from exc
        self.model = model
        self.base_url = resolve_base_url(base_url)
        # api_key is read from the environment and held only inside the SDK
        # client; it is never stored on this object nor logged.
        self._client = OpenAI(base_url=self.base_url, api_key=_resolve_api_key())

    def complete(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        """Issue one chat-completions call and return the message content.

        All three decoding parameters (``temperature``, ``max_tokens``,
        ``seed``) are passed through to the server so the values recorded in the
        run log are the values actually sent. This is load-bearing for the
        reproducibility claim; see ``tests/test_medgemma_inference.py``.
        """
        from openai.types.chat import ChatCompletionMessageParam

        # ``messages`` is our loose ``list[dict]`` payload; narrow it to the
        # openai SDK's expected type at the call boundary without copying.
        typed_messages = cast("list[ChatCompletionMessageParam]", messages)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=typed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        content = response.choices[0].message.content
        return content or ""


class StubClient:
    """In-process client returning canned, well-formed JSON. No network.

    Used by ``--dry-run`` and the test suite. It inspects the message payload
    to decide whether an extraction or an adjudication response is expected
    (an image part implies adjudication), so the canned output always parses
    under the matching schema.

    Parameters
    ----------
    model : str
        Reported model id, so run logs from a dry run are structurally
        identical to live runs.
    base_url : str
        Reported base URL (not contacted).
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = resolve_base_url(base_url)

    @staticmethod
    def _is_image_request(messages: ChatMessages) -> bool:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def complete(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        """Return canned JSON matching the requested mode.

        The decoding arguments (``temperature``, ``max_tokens``, ``seed``) are
        accepted to satisfy the :class:`ChatClient` protocol but are
        intentionally unused: the stub queries no model, so there is nothing for
        them to influence. The live :class:`OMLXClient` is what forwards them to
        the server.
        """
        del temperature, max_tokens, seed  # unused by design (no model queried)
        if self._is_image_request(messages):
            return json.dumps(
                {
                    "observed": "stub trace, no model queried",
                    "call": "indeterminate",
                    "confidence": 0.5,
                    "rationale": (
                        "Stub client response (--dry-run): no model was queried. "
                        "This is a placeholder adjudication."
                    ),
                }
            )
        return json.dumps(
            {
                "phenotype": "unknown",
                "vasopressor_use": None,
                "shock_state": None,
                "notes": "Stub client response (--dry-run): no model was queried.",
            }
        )
