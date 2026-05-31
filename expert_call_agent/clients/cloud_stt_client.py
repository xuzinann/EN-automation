import base64

import httpx

import config
from clients.gcp_auth import get_auth


class CloudSTTClient:
    """Google Cloud Speech-to-Text v2 (Chirp 2) — a purpose-built ASR model.

    Drop-in replacement for the Gemini-based ``STTClient``: same ``transcribe``
    signature, so ``routes_call.py`` is unchanged. Chirp 2 is regional (not
    ``global``), so this hits the regional endpoint and uses the implicit
    recognizer (``recognizers/_``) with inline config. ``autoDecodingConfig``
    lets the service detect the encoding, so the browser's webm/opus and our
    mp3 test fixtures both work without specifying ``mime_type``.
    """

    def __init__(self):
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",  # accepted for interface parity; auto-detected
    ) -> str:
        token, _ = await get_auth()
        location = config.SPEECH_LOCATION
        url = (
            f"https://{location}-speech.googleapis.com/v2/"
            f"projects/{config.GCP_PROJECT}/locations/{location}/recognizers/_:recognize"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": config.GCP_PROJECT,
        }
        body = {
            "config": {
                "model": config.SPEECH_MODEL,
                "languageCodes": config.SPEECH_LANGUAGE_CODES,
                "autoDecodingConfig": {},
                "features": {"enableAutomaticPunctuation": True},
            },
            "content": base64.b64encode(audio_bytes).decode(),
        }
        resp = await self._http_client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        parts = []
        for result in data.get("results", []):
            alternatives = result.get("alternatives") or []
            transcript = alternatives[0].get("transcript") if alternatives else ""
            if transcript:
                parts.append(transcript)
        return " ".join(parts).strip()

    async def close(self):
        await self._http_client.aclose()
