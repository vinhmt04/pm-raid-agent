"""
Base classes and interfaces for LLM provider integration.
Defines abstract LLMProvider interface and data models for responses and configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time

from logging_config import get_logger

logger = get_logger(__name__)


class LLMProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.
    """
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class LLMRequest:
    """
    Standardized request to LLM providers.
    """
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize request."""
        if not self.messages:
            raise ValueError("Messages list cannot be empty")
        
        # Add system prompt as first message if provided
        if self.system_prompt:
            system_msg = {"role": "system", "content": self.system_prompt}
            self.messages = [system_msg] + self.messages


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        self.message = message
        self.provider = provider  
        self.original_error = original_error
        super().__init__(self.message)


class LLMRateLimitError(LLMProviderError):
    """Exception raised when hitting rate limits."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMProviderError):
    """Exception raised for authentication failures."""
    pass


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Defines the interface that all LLM provider implementations must follow.
    """
    
    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type this implementation supports."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return human-readable provider name."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured with credentials."""
        pass
    
    @abstractmethod
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from LLM provider.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response
            
        Raises:
            LLMProviderError: If generation fails
            LLMRateLimitError: If rate limit is hit
            LLMAuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to LLM provider.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    def _create_error_response(self, error_message: str) -> LLMResponse:
        """
        Create an error response for failed requests.
        
        Args:
            error_message: Description of the error
            
        Returns:
            LLMResponse with error information
        """
        return LLMResponse(
            content=f"Error: {error_message}",
            provider=self.provider_name,
            model="unknown",
            finish_reason="error",
            metadata={"error": error_message}
        )


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class LLMProviderWithRetry:
    """
    Wrapper that adds retry logic to any LLM provider.
    """
    
    def __init__(self, provider: LLMProvider, retry_config: Optional[RetryConfig] = None):
        self.provider = provider
        self.retry_config = retry_config or RetryConfig()
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response with retry logic for transient failures.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
            
        Raises:
            LLMProviderError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()
                response = self.provider.generate_response(request)
                response.response_time = time.time() - start_time
                
                logger.info("LLM request successful", 
                          provider=self.provider.provider_name,
                          attempt=attempt + 1,
                          response_time=response.response_time)
                
                return response
                
            except LLMRateLimitError as e:
                last_error = e
                if attempt < self.retry_config.max_retries:
                    # Use provided retry_after or calculated delay
                    delay = e.retry_after or self.retry_config.get_delay(attempt)
                    logger.warning("Rate limit hit, retrying",
                                 provider=self.provider.provider_name, 
                                 attempt=attempt + 1,
                                 delay=delay)
                    time.sleep(delay)
                    continue
                else:
                    logger.error("Rate limit exceeded, no more retries",
                               provider=self.provider.provider_name)
                    raise
                    
            except LLMAuthenticationError as e:
                # Don't retry authentication errors
                logger.error("Authentication failed", 
                           provider=self.provider.provider_name,
                           error=str(e))
                raise
                
            except LLMProviderError as e:
                last_error = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning("LLM request failed, retrying",
                                 provider=self.provider.provider_name,
                                 attempt=attempt + 1, 
                                 error=str(e),
                                 delay=delay)
                    time.sleep(delay)
                    continue
                else:
                    logger.error("LLM request failed, no more retries",
                               provider=self.provider.provider_name,
                               error=str(e))
                    raise
        
        # Should not reach here, but just in case
        raise last_error or LLMProviderError("All retries failed", self.provider.provider_name)