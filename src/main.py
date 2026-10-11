"""
Main entry point for PM RAID Agent MVP.
Initializes logging and provides application startup utilities.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from logging_config import get_logger
from config import config

logger = get_logger(__name__)


def main() -> None:
    """
    Main application entry point.
    Sets up logging and validates configuration.
    """
    logger.info("PM RAID Agent MVP starting up")
    
    try:
        # Validate basic application configuration (excluding LLM credentials)
        config.app.dict()
        logger.info("Application configuration loaded", provider=config.app.llm_provider)
        
        # Note: LLM provider validation happens when actually used, not at startup
        logger.info("Ready to start document analysis workflow")
        
    except Exception as e:
        logger.error("Configuration validation failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
