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
            )
        else:
            return await self.claude.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=self.system_prompt,
            )

    def _parse_json(self, text: str) -> dict | list:
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return json.loads(cleaned)

    @abstractmethod
    async def run(self, session: CallSession, **kwargs):
        ...
