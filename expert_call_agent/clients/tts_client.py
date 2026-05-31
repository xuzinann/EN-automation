import base64

import httpx

import config
from clients.gcp_auth import get_auth


class TTSClient:
    def __init__(self):
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        token, project = await get_auth()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project,
        }
        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": config.TTS_LANGUAGE,
                "name": voice or config.TTS_VOICE,
            },
            "audioConfig": {"audioEncoding": "MP3"},
        }
        resp = await self._http_client.post(
            config.TTS_ENDPOINT, headers=headers, json=body
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()["audioContent"])

    async def close(self):
        await self._http_client.aclose()
