"""
Configuration management for PM RAID Agent MVP.
Implements loosely coupled LLM provider configuration with lazy initialization.
"""

import os
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProviderConfig(BaseModel, ABC):
    """Abstract base class for LLM provider configurations."""
    
    provider_name: str = Field(..., description="Name of the LLM provider")
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate provider credentials."""
        pass
    
    @abstractmethod
    def get_connection_params(self) -> Dict[str, Any]:
        """Get parameters needed for provider connection."""
        pass


class OpenAIProviderConfig(LLMProviderConfig):
    """OpenAI provider configuration implementation."""
    
    provider_name: str = Field(default="openai", description="Provider name")
    api_key: str = Field(default="", description="API key for authentication")
    model: str = Field(default="gpt-4", description="Model to use")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="Temperature for response generation")
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    def validate_credentials(self) -> bool:
        """Validate OpenAI credentials."""
        return bool(self.api_key and self.api_key != "your_openai_api_key_here" and len(self.api_key.strip()) > 0)
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get OpenAI connection parameters."""
        if not self.validate_credentials():
            raise ValueError("Invalid or missing OpenAI API key")
        
        return {
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


class ApplicationConfig(BaseModel):
    """Application-level configuration."""
    
    # LLM Provider Selection
    llm_provider: str = Field(default="openai", description="Selected LLM provider")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="structured", description="Log format")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_host: str = Field(default="localhost", description="Streamlit server host")
    
    # Document Processing Configuration
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    supported_formats: list[str] = Field(
        default=["pdf", "docx", "xlsx"], 
        description="Supported document formats"
    )
    
    # Session Configuration
    session_cleanup: bool = Field(default=True, description="Enable session cleanup")
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    
    # Test Configuration
    test_mode: bool = Field(default=False, description="Enable test mode")


class Config:
    """
    Main configuration class with lazy initialization and loosely coupled LLM provider interface.
    
    Supports one real LLM provider for MVP while maintaining interface
    abstraction for future provider additions.
    """
    
    def __init__(self):
        self.app = ApplicationConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "structured"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
            streamlit_host=os.getenv("STREAMLIT_HOST", "localhost"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            supported_formats=os.getenv("SUPPORTED_FORMATS", "pdf,docx,xlsx").split(","),
            session_cleanup=os.getenv("SESSION_CLEANUP", "true").lower() == "true",
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
            test_mode=os.getenv("TEST_MODE", "false").lower() == "true",
        )
        
        # Lazy initialization - provider config created only when needed
        self._llm_config: Optional[LLMProviderConfig] = None
    
    def _initialize_llm_provider(self) -> LLMProviderConfig:
        """Lazily initialize the selected LLM provider configuration."""
        if self._llm_config is not None:
            return self._llm_config
            
        provider = self.app.llm_provider.lower()
        
        if provider == "openai":
            self._llm_config = OpenAIProviderConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            )
        else:
            supported_providers = ["openai"]
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {supported_providers}"
            )
        
        return self._llm_config
    
    def get_llm_provider_config(self) -> LLMProviderConfig:
        """
        Get the configured LLM provider interface.
        
        Returns:
            LLM provider configuration implementing the common interface
            
        Raises:
            ValueError: If provider is not supported
        """
        return self._initialize_llm_provider()
    
    def validate_configuration(self) -> bool:
        """
        Validate the complete configuration including LLM provider credentials.
        
        Returns:
            True if all configuration is valid
            
        Raises:
            ValueError: If any configuration is invalid
        """
        try:
            # Validate application config
            self.app.dict()
            
            # Validate LLM provider config through interface (triggers lazy init)
            llm_config = self.get_llm_provider_config()
            llm_config.dict()
            
            if not llm_config.validate_credentials():
                raise ValueError(f"Invalid credentials for {llm_config.provider_name} provider")
            
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")


# Global configuration instance - safe to import without API key
config = Config()
