"""Gemini implementation of the existing LLMProvider contract."""
import os
import time
import logging
import re
from llm_providers.base import (LLMProvider, LLMProviderType, LLMRequest,
                                LLMResponse, LLMProviderError, LLMAuthenticationError,
                                LLMRateLimitError)


logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY', '').strip()
        self.model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip()

    @property
    def provider_type(self):
        return LLMProviderType.GEMINI

    @property
    def provider_name(self):
        return 'Gemini'

    def is_configured(self):
        return bool(self.api_key and self.api_key not in {'your_gemini_api_key', 'replace_with_your_gemini_api_key'})

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        if not self.is_configured():
            raise LLMAuthenticationError('GEMINI_API_KEY is missing', provider='gemini')
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMProviderError('Install Gemini SDK: python -m pip install google-genai', provider='gemini', original_error=exc) from exc

        model = request.model or self.model
        system = '\n\n'.join(m['content'] for m in request.messages if m['role'] == 'system')
        # Flatten role messages for compatibility with the project's request schema.
        content = '\n\n'.join(f"{m['role'].upper()}: {m['content']}" for m in request.messages if m['role'] != 'system')
        config_kwargs = {}
        if system:
            config_kwargs['system_instruction'] = system
        if request.temperature is not None:
            config_kwargs['temperature'] = request.temperature
        if request.max_tokens is not None:
            config_kwargs['max_output_tokens'] = request.max_tokens
        start = time.monotonic()
        # Three retries after the initial request: 2s, 4s, 8s.
        delays = (2, 4, 8)
        for attempt in range(len(delays) + 1):
            try:
                with genai.Client(api_key=self.api_key) as client:
                    result = client.models.generate_content(
                        model=model, contents=content,
                        config=types.GenerateContentConfig(**config_kwargs))
                response_text = result.text
                if not response_text:
                    raise LLMProviderError('Gemini returned no text (possibly blocked or truncated)', provider='gemini')
                usage = getattr(result, 'usage_metadata', None)
                return LLMResponse(content=response_text, provider='gemini', model=model,
                                   tokens_used=getattr(usage, 'total_token_count', None),
                                   response_time=time.monotonic()-start)
            except LLMProviderError:
                raise
            except Exception as exc:
                status = getattr(exc, 'code', None)
                if status is None:
                    status = getattr(exc, 'status_code', None)
                try:
                    status = int(status)
                except (TypeError, ValueError):
                    match = re.search(r'\b(400|401|403|404|408|429|500|502|503|504)\b', str(exc))
                    status = int(match.group(1)) if match else None
                msg = str(exc)
                upper_msg = msg.upper()
                # An exhausted billing balance or daily quota is not transient.
                quota_exhausted = any(x in upper_msg for x in (
                    'CREDIT_BALANCE_EXHAUSTED', 'INSUFFICIENT_QUOTA',
                    'QUOTA EXCEEDED', 'DAILY LIMIT', 'PER DAY'))
                temporary = (status in (408, 429, 500, 502, 503, 504)
                             or 'UNAVAILABLE' in upper_msg
                             or isinstance(exc, TimeoutError))
                if status == 429 and quota_exhausted:
                    temporary = False
                if temporary and attempt < len(delays):
                    delay = delays[attempt]
                    logger.warning('Gemini temporary failure (status=%s), retry %s/%s in %ss',
                                   status, attempt + 1, len(delays), delay)
                    time.sleep(delay)
                    continue
                if status == 429 or 'RESOURCE_EXHAUSTED' in upper_msg:
                    raise LLMRateLimitError(f'Gemini quota/rate limit: {msg}', provider='gemini', original_error=exc) from exc
                if status in (401, 403) or 'API_KEY_INVALID' in upper_msg:
                    raise LLMAuthenticationError(f'Gemini authentication failed: {msg}', provider='gemini', original_error=exc) from exc
                raise LLMProviderError(f'Gemini request failed: {msg}', provider='gemini', original_error=exc) from exc

    def test_connection(self):
        if not self.is_configured():
            return False
        try:
            self.generate_response(LLMRequest(messages=[{'role':'user','content':'Reply OK'}], max_tokens=20))
            return True
        except LLMProviderError:
            return False
