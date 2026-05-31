import httpx

import config
from clients.gcp_auth import get_auth


class ClaudeClient:
    def __init__(self):
        self.project = config.GCP_PROJECT
        self.region = config.CLAUDE_REGION
        self.model = config.CLAUDE_MODEL
        self._http_client = httpx.AsyncClient(timeout=60.0)

    async def _get_headers(self) -> dict:
        token, _ = await get_auth()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> str:
        url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.region}/"
            f"publishers/anthropic/models/{self.model}:rawPredict"
        )
        body = {
            "anthropic_version": config.ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature

        headers = await self._get_headers()
        resp = await self._http_client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    async def close(self):
        await self._http_client.aclose()
