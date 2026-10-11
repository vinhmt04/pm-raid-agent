"""Unit tests for configuration management."""

import os
import pytest
from unittest.mock import patch

# Import using sys.path manipulation to match main.py approach
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import Config, ApplicationConfig, OpenAIProviderConfig


class TestApplicationConfig:
    """Test application configuration."""
    
    def test_default_configuration(self):
        """Test that default configuration loads without errors."""
        app_config = ApplicationConfig()
        assert app_config.llm_provider == "openai"
        assert app_config.log_level == "INFO"
        assert app_config.streamlit_port == 8501
        assert app_config.supported_formats == ["pdf", "docx", "xlsx"]
    
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "STREAMLIT_PORT": "9000"}):
            app_config = ApplicationConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501"))
            )
            assert app_config.log_level == "DEBUG"
            assert app_config.streamlit_port == 9000


class TestOpenAIProviderConfig:
    """Test OpenAI provider configuration."""
    
    def test_valid_configuration(self):
        """Test valid OpenAI configuration."""
        config = OpenAIProviderConfig(api_key="sk-test123")
        assert config.provider_name == "openai"
        assert config.validate_credentials() is True
    
    def test_invalid_api_key(self):
        """Test invalid API key handling."""
        config = OpenAIProviderConfig(api_key="")
        assert config.validate_credentials() is False
        
        config = OpenAIProviderConfig(api_key="your_openai_api_key_here")
        assert config.validate_credentials() is False
    
    def test_connection_params_with_valid_key(self):
        """Test connection parameters with valid key."""
        config = OpenAIProviderConfig(api_key="sk-test123")
        params = config.get_connection_params()
        assert params["api_key"] == "sk-test123"
        assert params["model"] == "gpt-4"
    
    def test_connection_params_with_invalid_key(self):
        """Test connection parameters with invalid key raises error."""
        config = OpenAIProviderConfig(api_key="")
        with pytest.raises(ValueError, match="Invalid or missing OpenAI API key"):
            config.get_connection_params()


class TestConfig:
    """Test main configuration class."""
    
    def test_lazy_initialization_without_api_key(self):
        """Test that Config can be created without API key."""
        # This should not raise an error during initialization
        config = Config()
        assert config.app.llm_provider == "openai"
    
    def test_llm_provider_lazy_loading(self):
        """Test that LLM provider config is loaded lazily."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = Config()
            # Should not have initialized LLM config yet
            assert config._llm_config is None
            
            # Now it should initialize
            llm_config = config.get_llm_provider_config()
            assert llm_config.provider_name == "openai"
            assert config._llm_config is not None
    
    def test_validation_with_invalid_credentials(self):
        """Test validation fails with invalid credentials."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            config = Config()
            with pytest.raises(ValueError, match="Invalid credentials"):
                config.validate_configuration()
    
    def test_validation_with_valid_credentials(self):
        """Test validation passes with valid credentials."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=False):
            config = Config()
            assert config.validate_configuration() is True
