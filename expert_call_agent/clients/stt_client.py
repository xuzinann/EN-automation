import config
from clients.gemini_client import GeminiClient
from observability import op


class STTClient:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client

    @op()
    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/mp3",
    ) -> str:
        return await self.gemini.generate_with_audio(
            model=config.GEMINI_FLASH_MODEL,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            prompt="Transcribe this audio exactly. Return only the transcription, no commentary.",
        )

    async def close(self):
        # No own resources (shares the GeminiClient); present for interface parity.
        pass
