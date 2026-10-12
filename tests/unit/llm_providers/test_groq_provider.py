"""Mocked Groq provider tests: no live API calls or credentials."""
from unittest.mock import Mock, patch
import pytest
from llm_providers.groq_provider import GroqProvider
from llm_providers.base import LLMRequest, LLMRateLimitError, LLMAuthenticationError, LLMProviderError, LLMProviderType


@pytest.fixture
def provider():
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test-only", "GROQ_MODEL": "openai/gpt-oss-120b"}):
        return GroqProvider()


def test_properties(provider):
    assert provider.is_configured()
    assert provider.provider_type == LLMProviderType.GROQ
    assert provider.provider_name == "Groq"


def test_missing_key():
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
        p = GroqProvider()
        assert not p.is_configured()
        with pytest.raises(LLMAuthenticationError):
            p.generate_response(LLMRequest(messages=[{"role": "user", "content": "hi"}]))


@patch("groq.Groq")
def test_success(mock_groq, provider):
    completion = Mock()
    completion.choices = [Mock()]
    completion.choices[0].message.content = '{"risks": []}'
    completion.choices[0].finish_reason = "stop"
    completion.usage.total_tokens = 42
    mock_groq.return_value.__enter__.return_value.chat.completions.create.return_value = completion
    response = provider.generate_response(LLMRequest(
        messages=[{"role": "system", "content": "JSON only"}, {"role": "user", "content": "Find risks"}],
        temperature=0.1, max_tokens=300,
    ))
    assert response.content == '{"risks": []}'
    assert response.tokens_used == 42
    kwargs = mock_groq.return_value.__enter__.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-120b"
    assert kwargs["max_completion_tokens"] == 300
    assert kwargs["messages"][0]["role"] == "system"


@pytest.mark.parametrize("status,exception", [
    (429, LLMRateLimitError), (401, LLMAuthenticationError),
    (403, LLMAuthenticationError), (503, LLMProviderError),
])
@patch("groq.Groq")
def test_api_errors(mock_groq, provider, status, exception):
    error = Exception("simulated error")
    error.status_code = status
    mock_groq.return_value.__enter__.return_value.chat.completions.create.side_effect = error
    with pytest.raises(exception):
        provider.generate_response(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
