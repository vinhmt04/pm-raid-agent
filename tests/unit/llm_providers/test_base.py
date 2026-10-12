"""Unit tests for LLM provider base classes and interfaces."""

import pytest
from unittest.mock import Mock, patch
import time

from llm_providers.base import (
    LLMProvider, LLMRequest, LLMResponse, LLMProviderType,
    LLMProviderError, LLMRateLimitError, LLMAuthenticationError,
    RetryConfig, LLMProviderWithRetry
)


class TestLLMRequest:
    """Test LLMRequest data class."""
    
    def test_request_creation(self):
        """Test basic LLMRequest creation."""
        messages = [{"role": "user", "content": "Hello"}]
        request = LLMRequest(messages=messages)
        
        assert request.messages == messages
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.model is None
        assert request.system_prompt is None
    
    def test_request_with_system_prompt(self):
        """Test LLMRequest with system prompt."""
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "You are a helpful assistant."
        
        request = LLMRequest(messages=messages, system_prompt=system_prompt)
        
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"] == system_prompt
        assert request.messages[1] == {"role": "user", "content": "Hello"}
    
    def test_request_empty_messages(self):
        """Test LLMRequest with empty messages."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            LLMRequest(messages=[])
    
    def test_request_with_parameters(self):
        """Test LLMRequest with all parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        request = LLMRequest(
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            model="gpt-4"
        )
        
        assert request.messages == messages
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.model == "gpt-4"


class TestLLMResponse:
    """Test LLMResponse data class."""
    
    def test_response_creation(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            content="Hello world",
            provider="test",
            model="test-model"
        )
        
        assert response.content == "Hello world"
        assert response.provider == "test"
        assert response.model == "test-model"
        assert response.tokens_used is None
        assert response.finish_reason is None
        assert response.response_time is None
        assert response.metadata == {}
    
    def test_response_with_metadata(self):
        """Test LLMResponse with metadata."""
        metadata = {"test_key": "test_value"}
        response = LLMResponse(
            content="Hello",
            provider="test",
            model="test-model",
            tokens_used=10,
            finish_reason="stop",
            response_time=1.5,
            metadata=metadata
        )
        
        assert response.tokens_used == 10
        assert response.finish_reason == "stop"
        assert response.response_time == 1.5
        assert response.metadata == metadata


class TestLLMProviderType:
    """Test LLMProviderType enum."""
    
    def test_provider_types(self):
        """Test available provider types."""
        assert LLMProviderType.OPENAI.value == "openai"


class TestLLMProviderErrors:
    """Test LLM provider exception classes."""
    
    def test_base_error(self):
        """Test LLMProviderError."""
        error = LLMProviderError("Test error", provider="test")
        assert str(error) == "Test error"
        assert error.provider == "test"
        assert error.original_error is None
    
    def test_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("Rate limited", retry_after=60, provider="test")
        assert str(error) == "Rate limited"
        assert error.retry_after == 60
        assert error.provider == "test"
    
    def test_authentication_error(self):
        """Test LLMAuthenticationError."""
        original = ValueError("Invalid key")
        error = LLMAuthenticationError("Auth failed", provider="test", original_error=original)
        assert str(error) == "Auth failed"
        assert error.original_error == original


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_factor=3.0
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 3.0
    
    def test_delay_calculation(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=10.0)
        
        assert config.get_delay(0) == 1.0  # 1.0 * 2^0
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2
        assert config.get_delay(3) == 8.0  # 1.0 * 2^3
        assert config.get_delay(4) == 10.0  # capped at max_delay


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, should_fail=False, fail_with=None, response_content="Mock response"):
        self.should_fail = should_fail
        self.fail_with = fail_with or LLMProviderError("Mock error")
        self.response_content = response_content
        self.call_count = 0
    
    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI
    
    @property
    def provider_name(self) -> str:
        return "Mock"
    
    def is_configured(self) -> bool:
        return True
    
    def test_connection(self) -> bool:
        return not self.should_fail
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        
        if self.should_fail:
            raise self.fail_with
        
        return LLMResponse(
            content=self.response_content,
            provider=self.provider_name,
            model="mock-model",
            tokens_used=10,
            finish_reason="stop"
        )


class TestLLMProviderWithRetry:
    """Test LLMProviderWithRetry wrapper."""
    
    def test_successful_request(self):
        """Test successful request without retries."""
        mock_provider = MockLLMProvider()
        retry_wrapper = LLMProviderWithRetry(mock_provider, RetryConfig(max_retries=2))
        
        request = LLMRequest(messages=[{"role": "user", "content": "test"}])
        response = retry_wrapper.generate_response(request)
        
        assert response.content == "Mock response"
        assert mock_provider.call_count == 1
    
    def test_retry_on_provider_error(self):
        """Test retry logic for general provider errors."""
        mock_provider = MockLLMProvider(
            should_fail=True, 
            fail_with=LLMProviderError("Temporary error")
        )
        retry_config = RetryConfig(max_retries=2, base_delay=0.01)  # Fast retries for testing
        retry_wrapper = LLMProviderWithRetry(mock_provider, retry_config)
        
        request = LLMRequest(messages=[{"role": "user", "content": "test"}])
        
        with pytest.raises(LLMProviderError):
            retry_wrapper.generate_response(request)
        
        # Should have tried 3 times (initial + 2 retries)
        assert mock_provider.call_count == 3
    
    def test_retry_on_rate_limit(self):
        """Test retry logic for rate limit errors."""
        mock_provider = MockLLMProvider(
            should_fail=True,
            fail_with=LLMRateLimitError("Rate limited", retry_after=1)
        )
        retry_config = RetryConfig(max_retries=2, base_delay=0.01)
        retry_wrapper = LLMProviderWithRetry(mock_provider, retry_config)
        
        request = LLMRequest(messages=[{"role": "user", "content": "test"}])
        
        with pytest.raises(LLMRateLimitError):
            retry_wrapper.generate_response(request)
        
        assert mock_provider.call_count == 3
    
    def test_no_retry_on_auth_error(self):
        """Test that authentication errors are not retried."""
        mock_provider = MockLLMProvider(
            should_fail=True,
            fail_with=LLMAuthenticationError("Invalid API key")
        )
        retry_wrapper = LLMProviderWithRetry(mock_provider, RetryConfig(max_retries=2))
        
        request = LLMRequest(messages=[{"role": "user", "content": "test"}])
        
        with pytest.raises(LLMAuthenticationError):
            retry_wrapper.generate_response(request)
        
        # Should not retry authentication errors
        assert mock_provider.call_count == 1
    
    def test_eventual_success_after_retries(self):
        """Test success after some failed attempts."""
        # Mock provider that fails twice then succeeds
        mock_provider = Mock()
        mock_provider.provider_name = "Mock"
        mock_provider.generate_response = Mock()
        
        # First two calls fail, third succeeds
        mock_provider.generate_response.side_effect = [
            LLMProviderError("Error 1"),
            LLMProviderError("Error 2"), 
            LLMResponse(content="Success", provider="Mock", model="test")
        ]
        
        retry_config = RetryConfig(max_retries=3, base_delay=0.01)
        retry_wrapper = LLMProviderWithRetry(mock_provider, retry_config)
        
        request = LLMRequest(messages=[{"role": "user", "content": "test"}])
        response = retry_wrapper.generate_response(request)
        
        assert response.content == "Success"
        assert mock_provider.generate_response.call_count == 3


class TestLLMProviderAbstract:
    """Test abstract LLMProvider methods."""
    
    def test_create_error_response(self):
        """Test error response creation."""
        provider = MockLLMProvider()
        error_response = provider._create_error_response("Test error")
        
        assert "Error: Test error" in error_response.content
        assert error_response.provider == "Mock"
        assert error_response.finish_reason == "error"
        assert error_response.metadata["error"] == "Test error"