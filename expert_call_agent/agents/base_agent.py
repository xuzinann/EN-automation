import json
import re
from abc import ABC, abstractmethod

import config
from clients.gemini_client import GeminiClient
from clients.claude_client import ClaudeClient
from models import CallSession


class BaseAgent(ABC):
    name: str
    system_prompt: str
    temperature: float | None = None  # subclasses override; None = provider default

    def __init__(self, gemini_client: GeminiClient, claude_client: ClaudeClient):
        self.gemini = gemini_client
        self.claude = claude_client
        provider, model = config.AGENT_MODELS[self.name]
        self.model_provider = provider
        self.model_name = model

    async def _call_model(self, user_prompt: str) -> str:
        if self.model_provider == "gemini":
            return await self.gemini.generate(
                model=self.model_name,
                prompt=user_prompt,
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                temperature=self.temperature,
            )
        else:
            return await self.claude.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=self.system_prompt,
                temperature=self.temperature,
                model=self.model_name,
            )

    async def _call_model_text(self, user_prompt: str) -> str:
        """Plain-text (non-JSON) model call, for agents that speak prose.

        Mirrors _call_model but never requests JSON mode, so the chosen provider
        returns a natural spoken turn rather than a structured object.
        """
        if self.model_provider == "gemini":
            return await self.gemini.generate(
                model=self.model_name,
                prompt=user_prompt,
                system_instruction=self.system_prompt,
                temperature=self.temperature,
            )
        return await self.claude.generate(
            messages=[{"role": "user", "content": user_prompt}],
            system=self.system_prompt,
            temperature=self.temperature,
            model=self.model_name,
        )

    def _parse_json(self, text: str) -> dict | list:
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return json.loads(cleaned)

    def _safe_parse_json(self, text: str | None, default):
        """Parse model JSON, returning `default` on empty/None/malformed output.

        Live-call agents must degrade gracefully: a single bad turn should fall
        back to a safe default, never raise and stall the call.
        """
        if not text:
            return default
        try:
            return self._parse_json(text)
        except ValueError:  # json.JSONDecodeError is a subclass of ValueError
            return default

    @abstractmethod
    async def run(self, session: CallSession, **kwargs):
        ...
