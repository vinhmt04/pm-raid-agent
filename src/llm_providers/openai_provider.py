"""
OpenAI LLM provider implementation.
Handles OpenAI API integration with authentication, rate limiting, and error handling.
"""

from typing import Optional, Dict, Any
import openai
from openai import OpenAI
import time

from llm_providers.base import (
    LLMProvider, 
    LLMRequest, 
    LLMResponse, 
    LLMProviderType,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError
)
from config import config
from logging_config import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.
    Integrates with OpenAI API for chat completions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if not provided, uses config)
            model: Default model to use (if not provided, uses config)
        """
        self._client = None
        self._api_key = api_key
        self._default_model = model
        self._max_tokens = 4000
        self._temperature = 0.1
        
        # Get configuration if not provided
        if not self._api_key or not self._default_model:
            try:
                openai_config = config.get_llm_provider_config()
                connection_params = openai_config.get_connection_params()
                
                if not self._api_key:
                    self._api_key = connection_params["api_key"]
                if not self._default_model:
                    self._default_model = connection_params["model"]
                    
                self._max_tokens = connection_params.get("max_tokens", 4000)
                self._temperature = connection_params.get("temperature", 0.1)
                
            except Exception as e:
                logger.warning("Could not load OpenAI config", error=str(e))
                self._max_tokens = 4000
                self._temperature = 0.1
    
    @property
    def provider_type(self) -> LLMProviderType:
        """Return OpenAI provider type."""
        return LLMProviderType.OPENAI
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "OpenAI"
    
    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise LLMAuthenticationError("OpenAI API key not configured", provider="openai")
            
            self._client = OpenAI(api_key=self._api_key)
            logger.debug("OpenAI client initialized")
        
        return self._client
    
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        return bool(self._api_key and self._api_key != "your_openai_api_key_here")
    
    def test_connection(self) -> bool:
        """
        Test connection to OpenAI API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.is_configured():
                return False
            
            client = self._get_client()
            
            # Make a minimal test request
            test_request = LLMRequest(
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0
            )
            
            response = self.generate_response(test_request)
            return response.finish_reason != "error"
            
        except Exception as e:
            logger.warning("OpenAI connection test failed", error=str(e))
            return False
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using OpenAI API.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response
            
        Raises:
            LLMProviderError: If generation fails
            LLMRateLimitError: If rate limit is hit
            LLMAuthenticationError: If authentication fails
        """
        if not self.is_configured():
            raise LLMAuthenticationError("OpenAI provider not properly configured", provider="openai")
        
        try:
            client = self._get_client()
            
            # Prepare request parameters
            model = request.model or self._default_model
            max_tokens = request.max_tokens or self._max_tokens
            temperature = request.temperature if request.temperature is not None else self._temperature
            
            logger.debug("Making OpenAI API request", 
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages_count=len(request.messages))
            
            start_time = time.time()
            
            # Make API call
            completion = client.chat.completions.create(
                model=model,
                messages=request.messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response_time = time.time() - start_time
            
            # Extract response content
            content = completion.choices[0].message.content or ""
            finish_reason = completion.choices[0].finish_reason
            
            # Extract token usage if available
            tokens_used = None
            if completion.usage:
                tokens_used = completion.usage.total_tokens
            
            response = LLMResponse(
                content=content,
                provider=self.provider_name,
                model=model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                response_time=response_time,
                metadata={
                    "completion_id": completion.id,
                    "created": completion.created,
                    "usage": completion.usage.model_dump() if completion.usage else None
                }
            )
            
            logger.info("OpenAI API request successful",
                       model=model,
                       tokens_used=tokens_used,
                       response_time=response_time,
                       finish_reason=finish_reason)
            
            return response
            
        except openai.RateLimitError as e:
            error_msg = f"OpenAI rate limit exceeded: {str(e)}"
            logger.warning("OpenAI rate limit hit", error=str(e))
            
            # Extract retry-after header if available
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    retry_after = int(retry_after)
            
            raise LLMRateLimitError(error_msg, retry_after=retry_after, provider="openai", original_error=e)
            
        except openai.AuthenticationError as e:
            error_msg = f"OpenAI authentication failed: {str(e)}"
            logger.error("OpenAI authentication failed", error=str(e))
            raise LLMAuthenticationError(error_msg, provider="openai", original_error=e)
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error("OpenAI API error", error=str(e))
            raise LLMProviderError(error_msg, provider="openai", original_error=e)
            
        except Exception as e:
            error_msg = f"Unexpected OpenAI error: {str(e)}"
            logger.error("Unexpected OpenAI error", error=str(e))
            raise LLMProviderError(error_msg, provider="openai", original_error=e)
    
    def get_available_models(self) -> list:
        """
        Get list of available models from OpenAI.
        
        Returns:
            List of available model names
            
        Raises:
            LLMProviderError: If unable to retrieve models
        """
        try:
            client = self._get_client()
            models = client.models.list()
            return [model.id for model in models.data if "gpt" in model.id.lower()]
            
        except Exception as e:
            logger.error("Failed to get OpenAI models", error=str(e))
            raise LLMProviderError(f"Failed to get available models: {str(e)}", provider="openai", original_error=e)