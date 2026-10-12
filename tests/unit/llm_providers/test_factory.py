"""Unit tests for LLM provider factory."""

import pytest
from unittest.mock import Mock, patch

from llm_providers.factory import LLMProviderFactory
from llm_providers.base import LLMProviderType, LLMProviderError, RetryConfig
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.groq_provider import GroqProvider


class TestLLMProviderFactory:
    """Test LLM provider factory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = LLMProviderFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization with available providers."""
        assert LLMProviderType.OPENAI in self.factory._providers
        assert self.factory._providers[LLMProviderType.OPENAI] == OpenAIProvider
        assert self.factory._providers[LLMProviderType.GEMINI] == GeminiProvider
        assert self.factory._providers[LLMProviderType.GROQ] == GroqProvider
    
    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = self.factory.get_available_providers()
        assert LLMProviderType.OPENAI in providers
        assert LLMProviderType.GEMINI in providers
        assert LLMProviderType.GROQ in providers
        assert len(providers) == 3
    
    @patch('llm_providers.factory.config')
    def test_get_provider_from_config(self, mock_config):
        """Test getting provider based on configuration."""
        mock_config.app.llm_provider = "openai"
        
        provider = self.factory.get_provider()
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_type == LLMProviderType.OPENAI
    
    def test_get_provider_explicit_type(self):
        """Test getting provider with explicit type."""
        provider = self.factory.get_provider(LLMProviderType.OPENAI)
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_type == LLMProviderType.OPENAI
    
    @patch('llm_providers.factory.config')
    def test_get_provider_unsupported_config(self, mock_config):
        """Test error handling for unsupported configured provider."""
        mock_config.app.llm_provider = "unsupported_provider"
        
        with pytest.raises(LLMProviderError) as excinfo:
            self.factory.get_provider()
        
        assert "Unsupported LLM provider: unsupported_provider" in str(excinfo.value)
    
    def test_get_provider_unsupported_type(self):
        """Test error for unsupported provider type."""
        # Create a mock enum value that's not in the factory
        class UnsupportedType:
            pass
        
        unsupported_type = UnsupportedType()
        
        with pytest.raises(LLMProviderError) as excinfo:
            self.factory.get_provider(unsupported_type)
        
        assert "No implementation available" in str(excinfo.value)
    
    @patch('llm_providers.factory.config')
    def test_get_provider_with_retry(self, mock_config):
        """Test getting provider with retry wrapper."""
        mock_config.app.llm_provider = "openai"
        
        retry_config = RetryConfig(max_retries=5)
        provider_with_retry = self.factory.get_provider_with_retry(
            retry_config=retry_config
        )
        
        assert hasattr(provider_with_retry, 'generate_response')
        assert provider_with_retry.retry_config.max_retries == 5
        assert isinstance(provider_with_retry.provider, OpenAIProvider)
    
    @patch('llm_providers.factory.config')
    def test_get_provider_with_retry_default_config(self, mock_config):
        """Test getting provider with retry using default config."""
        mock_config.app.llm_provider = "openai"
        
        provider_with_retry = self.factory.get_provider_with_retry()
        
        assert provider_with_retry.retry_config.max_retries == 3  # Default
        assert isinstance(provider_with_retry.provider, OpenAIProvider)
    
    @patch('llm_providers.factory.config')
    def test_is_provider_configured_true(self, mock_config):
        """Test checking if provider is configured (true case)."""
        mock_config.app.llm_provider = "openai"
        
        with patch.object(OpenAIProvider, 'is_configured', return_value=True):
            assert self.factory.is_provider_configured() is True
    
    @patch('llm_providers.factory.config')
    def test_is_provider_configured_false(self, mock_config):
        """Test checking if provider is configured (false case)."""
        mock_config.app.llm_provider = "openai"
        
        with patch.object(OpenAIProvider, 'is_configured', return_value=False):
            assert self.factory.is_provider_configured() is False
    
    @patch('llm_providers.factory.config')
    def test_is_provider_configured_error(self, mock_config):
        """Test checking provider configuration with error."""
        mock_config.app.llm_provider = "invalid"
        
        # Should return False when provider creation fails
        assert self.factory.is_provider_configured() is False
    
    @patch('llm_providers.factory.config')
    def test_test_provider_connection_success(self, mock_config):
        """Test provider connection test (success)."""
        mock_config.app.llm_provider = "openai"
        
        with patch.object(OpenAIProvider, 'test_connection', return_value=True):
            assert self.factory.test_provider_connection() is True
    
    @patch('llm_providers.factory.config')
    def test_test_provider_connection_failure(self, mock_config):
        """Test provider connection test (failure)."""
        mock_config.app.llm_provider = "openai"
        
        with patch.object(OpenAIProvider, 'test_connection', return_value=False):
            assert self.factory.test_provider_connection() is False
    
    @patch('llm_providers.factory.config')
    def test_test_provider_connection_error(self, mock_config):
        """Test provider connection test with error."""
        mock_config.app.llm_provider = "invalid"
        
        # Should return False when provider creation fails
        assert self.factory.test_provider_connection() is False
    
    def test_explicit_provider_type_override_config(self):
        """Test that explicit provider type overrides config."""
        # Even if config says something else, explicit type should work
        with patch('llm_providers.factory.config') as mock_config:
            mock_config.app.llm_provider = "some_other_provider"
            
            provider = self.factory.get_provider(LLMProviderType.OPENAI)
            assert isinstance(provider, OpenAIProvider)
    
    def test_provider_creation_error_handling(self):
        """Test error handling during provider creation."""
        with patch.object(OpenAIProvider, '__init__', side_effect=Exception("Init failed")):
            with pytest.raises(LLMProviderError) as excinfo:
                self.factory.get_provider(LLMProviderType.OPENAI)
            
            assert "Failed to create LLM provider" in str(excinfo.value)

class TestGeminiFactory:
    def test_explicit_gemini_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        provider = LLMProviderFactory().get_provider(LLMProviderType.GEMINI)
        assert isinstance(provider, GeminiProvider)
        assert provider.is_configured()

    def test_gemini_selected_by_config(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        with patch("llm_providers.factory.config") as mock_config:
            mock_config.app.llm_provider = "gemini"
            provider = LLMProviderFactory().get_provider()
            assert isinstance(provider, GeminiProvider)


class TestGroqFactory:
    @patch("llm_providers.factory.config")
    def test_groq_selection(self, mock_config):
        mock_config.app.llm_provider = "groq"
        provider = LLMProviderFactory().get_provider()
        assert isinstance(provider, GroqProvider)
        assert provider.provider_type == LLMProviderType.GROQ
