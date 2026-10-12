"""Unit tests for OpenAI LLM provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import openai
import httpx

from llm_providers.openai_provider import OpenAIProvider
from llm_providers.base import (
    LLMRequest, LLMResponse, LLMProviderType,
    LLMProviderError, LLMRateLimitError, LLMAuthenticationError
)


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "sk-test123"
        self.model = "gpt-3.5-turbo"
    
    def test_provider_properties(self):
        """Test provider type and name."""
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        
        assert provider.provider_type == LLMProviderType.OPENAI
        assert provider.provider_name == "OpenAI"
    
    def test_is_configured_valid_key(self):
        """Test is_configured with valid API key."""
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        assert provider.is_configured() is True
    
    def test_is_configured_invalid_key(self):
        """Test is_configured with invalid API key."""
        with patch("llm_providers.openai_provider.config") as mock_config:
            mock_config.get_llm_provider_config.side_effect = ValueError("No credentials")
            provider = OpenAIProvider(api_key="", model=self.model)
            assert provider.is_configured() is False
        
        provider = OpenAIProvider(api_key="your_openai_api_key_here", model=self.model)
        assert provider.is_configured() is False
    
    def test_initialization_without_params(self):
        """Test initialization without explicit parameters (uses config)."""
        with patch('llm_providers.openai_provider.config') as mock_config:
            # Mock the config to provide values
            mock_llm_config = Mock()
            mock_llm_config.get_connection_params.return_value = {
                "api_key": "sk-config123",
                "model": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.2
            }
            mock_config.get_llm_provider_config.return_value = mock_llm_config
            
            provider = OpenAIProvider()
            
            assert provider._api_key == "sk-config123"
            assert provider._default_model == "gpt-4"
            assert provider._max_tokens == 2000
            assert provider._temperature == 0.2
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_successful_generation(self, mock_openai):
        """Test successful response generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock completion response
        mock_completion = Mock()
        mock_completion.id = "chatcmpl-123"
        mock_completion.created = 1234567890
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Hello, how can I help you?"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 25
        mock_completion.usage.model_dump.return_value = {"total_tokens": 25}
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Test the provider
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
        
        response = provider.generate_response(request)
        
        assert response.content == "Hello, how can I help you?"
        assert response.provider == "OpenAI"
        assert response.model == self.model
        assert response.tokens_used == 25
        assert response.finish_reason == "stop"
        assert response.response_time is not None
        assert "completion_id" in response.metadata
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == self.model
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_rate_limit_error(self, mock_openai):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock rate limit error
        mock_response = httpx.Response(
            429,
            headers={"retry-after": "30"},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        rate_error = openai.RateLimitError(
            "Rate limit exceeded", response=mock_response, body=None
        )
        
        mock_client.chat.completions.create.side_effect = rate_error
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(LLMRateLimitError) as excinfo:
            provider.generate_response(request)
        
        assert "rate limit exceeded" in str(excinfo.value).lower()
        assert excinfo.value.retry_after == 30
        assert excinfo.value.provider == "openai"
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_authentication_error(self, mock_openai):
        """Test handling of authentication errors."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        auth_error = openai.AuthenticationError(
            "Invalid API key",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = auth_error
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(LLMAuthenticationError) as excinfo:
            provider.generate_response(request)
        
        assert "authentication failed" in str(excinfo.value).lower()
        assert excinfo.value.provider == "openai"
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_api_error(self, mock_openai):
        """Test handling of general API errors."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        api_error = openai.APIError(
            "Service unavailable",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = api_error
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(LLMProviderError) as excinfo:
            provider.generate_response(request)
        
        assert "api error" in str(excinfo.value).lower()
        assert excinfo.value.provider == "openai"
    
    def test_generate_without_configuration(self):
        """Test generation without proper configuration."""
        with patch("llm_providers.openai_provider.config") as mock_config:
            mock_config.get_llm_provider_config.side_effect = ValueError("No credentials")
            provider = OpenAIProvider(api_key="", model=self.model)
        request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(LLMAuthenticationError) as excinfo:
            provider.generate_response(request)
        
        assert "not properly configured" in str(excinfo.value)
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_request_parameters(self, mock_openai):
        """Test that request parameters are properly passed to API."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock successful response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = None
        mock_client.chat.completions.create.return_value = mock_completion
        
        provider = OpenAIProvider(api_key=self.api_key, model="gpt-4")
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=500,
            temperature=0.8,
            model="gpt-3.5-turbo"  # Override default
        )
        
        provider.generate_response(request)
        
        # Verify API call parameters
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-3.5-turbo"  # Request override
        assert call_args["max_tokens"] == 500
        assert call_args["temperature"] == 0.8
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_test_connection_success(self, mock_openai):
        """Test successful connection test."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock successful minimal response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = None
        mock_client.chat.completions.create.return_value = mock_completion
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        assert provider.test_connection() is True
    
    def test_test_connection_not_configured(self):
        """Test connection test with invalid configuration."""
        with patch("llm_providers.openai_provider.config") as mock_config:
            mock_config.get_llm_provider_config.side_effect = ValueError("No credentials")
            provider = OpenAIProvider(api_key="", model=self.model)
        assert provider.test_connection() is False
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_test_connection_failure(self, mock_openai):
        """Test connection test with API failure."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        assert provider.test_connection() is False
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_get_available_models(self, mock_openai):
        """Test getting available models."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock models response
        mock_models = Mock()
        mock_models.data = [
            Mock(id="gpt-4"),
            Mock(id="gpt-3.5-turbo"),
            Mock(id="text-embedding-ada-002"),  # Non-GPT model
            Mock(id="gpt-4-turbo")
        ]
        mock_client.models.list.return_value = mock_models
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        models = provider.get_available_models()
        
        # Should only return GPT models
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "gpt-4-turbo" in models
        assert "text-embedding-ada-002" not in models
    
    @patch('llm_providers.openai_provider.OpenAI')
    def test_get_available_models_failure(self, mock_openai):
        """Test handling of models list failure."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_client.models.list.side_effect = Exception("API Error")
        
        provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        
        with pytest.raises(LLMProviderError) as excinfo:
            provider.get_available_models()
        
        assert "Failed to get available models" in str(excinfo.value)