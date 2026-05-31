import asyncio

from google import genai
from google.genai import types

import config
from observability import op


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT,
            location=config.GEMINI_REGION,
            http_options=types.HttpOptions(timeout=config.GEMINI_HTTP_TIMEOUT_MS),
        )

    @op()
    async def generate(
        self,
        model: str,
        prompt: str,
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
        temperature: float | None = None,
    ) -> str:
        gen_config = types.GenerateContentConfig()
        if system_instruction:
            gen_config.system_instruction = system_instruction
        if response_mime_type:
            gen_config.response_mime_type = response_mime_type
        if temperature is not None:
            gen_config.temperature = temperature

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
            config=gen_config,
        )
        return response.text

    @op()
    async def generate_with_audio(
        self,
        model: str,
        audio_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        text_part = types.Part.from_text(text=prompt)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=[audio_part, text_part],
        )
        return response.text
