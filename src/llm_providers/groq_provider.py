"""Groq API provider for the PM RAID Agent LLMProvider contract."""
import os
import time
from llm_providers.base import (
    LLMProvider, LLMProviderType, LLMRequest, LLMResponse,
    LLMProviderError, LLMAuthenticationError, LLMRateLimitError,
)


class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    @property
    def provider_type(self):
        return LLMProviderType.GROQ

    @property
    def provider_name(self):
        return "Groq"

    def is_configured(self):
        return bool(self.api_key and self.api_key not in {
            "your_groq_api_key", "gsk_xxxxxxxxxxxxxxxxx", "replace_with_your_groq_api_key"
        })

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        if not self.is_configured():
            raise LLMAuthenticationError("GROQ_API_KEY is missing", provider="groq")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMProviderError(
                "Install Groq SDK: python -m pip install groq",
                provider="groq", original_error=exc
            ) from exc

        model = request.model or self.model
        messages = list(request.messages)
        kwargs = {"model": model, "messages": messages}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_tokens

        start = time.monotonic()
        try:
            with Groq(api_key=self.api_key, max_retries=0) as client:
                result = client.chat.completions.create(**kwargs)
            choice = result.choices[0]
            content = choice.message.content
            if not content:
                raise LLMProviderError(
                    "Groq returned no text (possibly token limit or empty response)",
                    provider="groq"
                )
            usage = getattr(result, "usage", None)
            return LLMResponse(
                content=content, provider="groq", model=model,
                tokens_used=getattr(usage, "total_tokens", None),
                finish_reason=getattr(choice, "finish_reason", None),
                response_time=time.monotonic() - start,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            message = str(exc)
            if status == 429:
                # Deliberately no automatic retries: daily/token quotas may not recover quickly.
                raise LLMRateLimitError(
                    f"Groq rate/quota limit: {message}", provider="groq", original_error=exc
                ) from exc
            if status in (401, 403):
                raise LLMAuthenticationError(
                    f"Groq authentication failed: {message}", provider="groq", original_error=exc
                ) from exc
            raise LLMProviderError(
                f"Groq request failed: {message}", provider="groq", original_error=exc
            ) from exc

    def test_connection(self):
        if not self.is_configured():
            return False
        try:
            self.generate_response(LLMRequest(
                messages=[{"role": "user", "content": "Reply OK"}], max_tokens=32
            ))
            return True
        except LLMProviderError:
            return False
