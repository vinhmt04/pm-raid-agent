"""
LLM provider factory for environment-based provider selection.
Provides unified interface for creating and managing LLM providers.
"""

from typing import Optional, Dict, Type
from llm_providers.base import LLMProvider, LLMProviderType, LLMProviderError, LLMProviderWithRetry, RetryConfig
from llm_providers.openai_provider import OpenAIProvider
from config import config
from logging_config import get_logger

logger = get_logger(__name__)


class LLMProviderFactory:
    """
    Factory class for creating and managing LLM providers.
    Uses environment configuration to select and configure providers.
    """
    
    def __init__(self):
        """Initialize factory with available providers."""
        self._providers: Dict[LLMProviderType, Type[LLMProvider]] = {
            LLMProviderType.OPENAI: OpenAIProvider
        }
        
        logger.debug("LLM provider factory initialized", 
                    providers=list(self._providers.keys()))
    
    def get_provider(self, provider_type: Optional[LLMProviderType] = None) -> LLMProvider:
        """
        Get LLM provider instance based on configuration.
        
        Args:
            provider_type: Specific provider type to get (if None, uses config)
            
        Returns:
            Configured LLM provider instance
            
        Raises:
            LLMProviderError: If provider type not supported or configuration invalid
        """
        try:
            # Use configured provider if not specified
            if provider_type is None:
                configured_provider = config.app.llm_provider.lower()
                
                # Map string to enum
                provider_mapping = {
                    "openai": LLMProviderType.OPENAI
                }
                
                if configured_provider not in provider_mapping:
                    supported = list(provider_mapping.keys())
                    raise LLMProviderError(
                        f"Unsupported LLM provider: {configured_provider}. "
                        f"Supported providers: {supported}"
                    )
                
                provider_type = provider_mapping[configured_provider]
            
            # Get provider class
            provider_class = self._providers.get(provider_type)
            if provider_class is None:
                raise LLMProviderError(f"No implementation available for provider: {provider_type}")
            
            # Create provider instance
            provider = provider_class()
            
            logger.debug("LLM provider created", 
                        provider_type=provider_type,
                        provider_name=provider.provider_name,
                        is_configured=provider.is_configured())
            
            return provider
            
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Failed to create LLM provider: {str(e)}", original_error=e)
    
    def get_provider_with_retry(self, 
                               provider_type: Optional[LLMProviderType] = None,
                               retry_config: Optional[RetryConfig] = None) -> LLMProviderWithRetry:
        """
        Get LLM provider with retry logic.
        
        Args:
            provider_type: Specific provider type to get (if None, uses config)
            retry_config: Retry configuration (if None, uses default)
            
        Returns:
            LLM provider wrapped with retry logic
            
        Raises:
            LLMProviderError: If provider creation fails
        """
        provider = self.get_provider(provider_type)
        return LLMProviderWithRetry(provider, retry_config)
    
    def get_available_providers(self) -> list:
        """
        Get list of available provider types.
        
        Returns:
            List of available LLMProviderType values
        """
        return list(self._providers.keys())
    
    def is_provider_configured(self, provider_type: Optional[LLMProviderType] = None) -> bool:
        """
        Check if a provider is properly configured.
        
        Args:
            provider_type: Provider type to check (if None, uses config)
            
        Returns:
            True if provider is configured, False otherwise
        """
        try:
            provider = self.get_provider(provider_type)
            return provider.is_configured()
        except Exception as e:
            logger.warning("Failed to check provider configuration", 
                         provider_type=provider_type, 
                         error=str(e))
            return False
    
    def test_provider_connection(self, provider_type: Optional[LLMProviderType] = None) -> bool:
        """
        Test connection to LLM provider.
        
        Args:
            provider_type: Provider type to test (if None, uses config)
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            provider = self.get_provider(provider_type)
            return provider.test_connection()
        except Exception as e:
            logger.warning("Provider connection test failed",
                         provider_type=provider_type,
                         error=str(e))
            return False


# Global factory instance
llm_factory = LLMProviderFactory()