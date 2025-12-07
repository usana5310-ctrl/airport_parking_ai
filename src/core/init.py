"""
Core Module - Initialization File
Project: AI Commercial Platform for Airport Parking
Company: Go Airport Parking (Multi-Airport, Multi-Franchise Network)

This module provides the foundational infrastructure for the AI Commercial Platform,
including configuration management, database connections, utilities, and exception handling.

Reference: [About The Role: Building one of the most advanced AI-driven commercial platforms]
"""

__version__ = "1.0.0"
__author__ = "AI Commercial Platform Team"
__description__ = "Core infrastructure for AI-driven commercial, pricing, marketing, and operations platform"

# Core module exports
from .config import settings, constants, feature_flags
from .database import models, connections, repositories
from .utils import logger, validators, safety_overrides, date_helpers
from .exceptions import business_exceptions, system_exceptions

# Public API - these are the main entry points for the core module
__all__ = [
    # Configuration
    "settings",
    "constants",
    "feature_flags",
    
    # Database
    "models",
    "connections",
    "repositories",
    
    # Utilities
    "logger",
    "validators",
    "safety_overrides",
    "date_helpers",
    
    # Exceptions
    "business_exceptions",
    "system_exceptions",
    
    # Version info
    "__version__",
    "__author__",
    "__description__",
]

# Initialize core module on import
def _initialize_core_module():
    """
    Initialize core module components when the module is imported.
    This ensures proper setup before any other modules use core functionality.
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    from .utils.logger import setup_logging
    from .config.settings import initialize_settings
    
    # Set up logging first (critical for debugging)
    setup_logging()
    
    # Initialize application settings
    initialize_settings()
    
    # Log successful initialization
    import logging
    core_logger = logging.getLogger("core")
    core_logger.info(
        f"Core module initialized - Version {__version__} - {__description__}",
        extra={
            "module": "core",
            "version": __version__,
            "author": __author__
        }
    )

# Run initialization
_initialize_core_module()

# Module metadata for package discovery
module_metadata = {
    "name": "core",
    "version": __version__,
    "description": __description__,
    "dependencies": [
        "python>=3.8",
        "pydantic>=2.0",
        "sqlalchemy>=2.0",
        "python-dotenv>=1.0",
    ],
    "exports": __all__,
}

def get_module_info() -> dict:
    """
    Returns metadata about the core module.
    
    Returns:
        dict: Module metadata including version, description, and exports
        
    Reference: [Absolutely Essential: Must be able to explain AI decisions to non-technical management]
    """
    return module_metadata.copy()

def validate_environment() -> bool:
    """
    Validates that the core module environment is properly configured.
    
    Returns:
        bool: True if environment is valid, False otherwise
        
    Reference: [Required Experience: Production engineering role]
    """
    from .config.settings import validate_settings
    from .database.connections import test_connections
    
    try:
        # Validate settings
        if not validate_settings():
            return False
        
        # Test database connections
        if not test_connections():
            return False
            
        return True
    except Exception as e:
        import logging
        logging.error(f"Environment validation failed: {str(e)}")
        return False

# Add convenience imports for common operations
def get_logger(name: str):
    """
    Convenience function to get a logger instance.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger instance
        
    Reference: [Absolutely Essential: Design with safety, logging, overrides]
    """
    from .utils.logger import get_logger as _get_logger
    return _get_logger(name)

def get_db_session():
    """
    Convenience function to get a database session.
    
    Returns:
        sqlalchemy.orm.Session: Database session
        
    Reference: [Technical Stack: MySQL / PostgreSQL]
    """
    from .database.connections import get_session
    return get_session()

# Register module with package system
def register_module():
    """
    Registers the core module with the application package system.
    This is called by the main application during startup.
    
    Reference: [Why This Role Is Different: Building a live commercial brain]
    """
    # This would typically register the module with a central registry
    # For now, we just log the registration
    import logging
    logger = logging.getLogger("core")
    logger.info("Core module registered with application package system")
