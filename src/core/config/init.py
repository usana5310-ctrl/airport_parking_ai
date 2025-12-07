"""
Configuration Submodule - Initialization File
Project: AI Commercial Platform for Airport Parking

This submodule manages all configuration aspects of the AI Commercial Platform,
including environment settings, business constants, and feature flags.

Reference: [Technical Stack: Full system integration requirements]
"""

__version__ = "1.0.0"
__description__ = "Configuration management for AI Commercial Platform"

# Import configuration components
from .settings import (
    Settings,
    get_settings,
    initialize_settings,
    validate_settings,
    reload_settings
)

from .constants import (
    # Airport-related constants
    AIRPORT_CODES,
    AIRPORT_NAMES,
    AIRPORT_TIMEZONES,
    AIRPORT_REGIONS,
    
    # Business constants
    MINIMUM_PRICE_FLOORS,
    MAXIMUM_PRICE_CAPS,
    DEFAULT_CPC_LIMITS,
    
    # Time constants
    DEFAULT_BOOKING_WINDOW_DAYS,
    MINIMUM_STAY_HOURS,
    MAXIMUM_STAY_DAYS,
    
    # Currency constants
    CURRENCY_CODE,
    CURRENCY_SYMBOL,
    TAX_RATES_BY_STATE,
    
    # Validation constants
    PRICE_VALIDATION_RULES,
    DATE_VALIDATION_RULES,
    
    # Helper functions
    get_airport_info,
    validate_airport_code,
    get_business_day_range,
    convert_currency
)

from .feature_flags import (
    FeatureFlags,
    get_feature_flags,
    is_feature_enabled,
    enable_feature,
    disable_feature,
    set_feature_override,
    
    # Feature names (defined as constants)
    FEATURE_DYNAMIC_PRICING,
    FEATURE_COMPETITOR_MONITORING,
    FEATURE_MARKETING_AUTOMATION,
    FEATURE_FRAUD_DETECTION,
    FEATURE_WEBSITE_PERSONALIZATION,
    FEATURE_CALL_CENTER_AI,
    
    # Feature groups
    FEATURE_GROUP_PRICING,
    FEATURE_GROUP_MARKETING,
    FEATURE_GROUP_OPERATIONS,
    
    # Helper functions
    get_enabled_features,
    get_disabled_features,
    validate_feature_config
)

# Public API - expose only what should be used externally
__all__ = [
    # Settings management
    "Settings",
    "get_settings",
    "initialize_settings",
    "validate_settings",
    "reload_settings",
    
    # Constants
    "AIRPORT_CODES",
    "AIRPORT_NAMES",
    "AIRPORT_TIMEZONES",
    "AIRPORT_REGIONS",
    "MINIMUM_PRICE_FLOORS",
    "MAXIMUM_PRICE_CAPS",
    "DEFAULT_CPC_LIMITS",
    "DEFAULT_BOOKING_WINDOW_DAYS",
    "MINIMUM_STAY_HOURS",
    "MAXIMUM_STAY_DAYS",
    "CURRENCY_CODE",
    "CURRENCY_SYMBOL",
    "TAX_RATES_BY_STATE",
    "PRICE_VALIDATION_RULES",
    "DATE_VALIDATION_RULES",
    "get_airport_info",
    "validate_airport_code",
    "get_business_day_range",
    "convert_currency",
    
    # Feature flags
    "FeatureFlags",
    "get_feature_flags",
    "is_feature_enabled",
    "enable_feature",
    "disable_feature",
    "set_feature_override",
    "FEATURE_DYNAMIC_PRICING",
    "FEATURE_COMPETITOR_MONITORING",
    "FEATURE_MARKETING_AUTOMATION",
    "FEATURE_FRAUD_DETECTION",
    "FEATURE_WEBSITE_PERSONALIZATION",
    "FEATURE_CALL_CENTER_AI",
    "FEATURE_GROUP_PRICING",
    "FEATURE_GROUP_MARKETING",
    "FEATURE_GROUP_OPERATIONS",
    "get_enabled_features",
    "get_disabled_features",
    "validate_feature_config",
    
    # Module info
    "__version__",
    "__description__",
]

# Module initialization
def _initialize_config_module():
    """
    Initialize the configuration module.
    This ensures all configuration is loaded and validated before use.
    
    Reference: [Required Experience: Production AI/data engineering - must handle configuration properly]
    """
    import logging
    from .settings import initialize_settings as _init_settings
    
    # Get logger
    logger = logging.getLogger("config")
    
    try:
        # Initialize settings from environment
        settings = _init_settings()
        
        # Log successful initialization
        logger.info(
            "Configuration module initialized successfully",
            extra={
                "module": "config",
                "version": __version__,
                "environment": settings.ENVIRONMENT,
                "airports_configured": len(AIRPORT_CODES)
            }
        )
        
        # Validate feature flags configuration
        from .feature_flags import validate_feature_config as _validate_features
        feature_status = _validate_features()
        
        if feature_status["valid"]:
            logger.info(
                "Feature flags validated",
                extra={
                    "enabled_features": feature_status["enabled_count"],
                    "disabled_features": feature_status["disabled_count"]
                }
            )
        else:
            logger.warning(
                "Feature flag validation issues detected",
                extra={"issues": feature_status["issues"]}
            )
            
    except Exception as e:
        logger.error(
            f"Failed to initialize configuration module: {str(e)}",
            extra={"module": "config", "error": str(e)}
        )
        raise

# Configuration validation functions
def validate_full_configuration() -> dict:
    """
    Perform comprehensive validation of all configuration components.
    
    Returns:
        dict: Validation results including status and any issues
        
    Reference: [Absolutely Essential: Design with safety, logging, overrides, and compliance]
    """
    import logging
    from .settings import validate_settings as _validate_settings
    from .feature_flags import validate_feature_config as _validate_features
    
    logger = logging.getLogger("config")
    validation_result = {
        "valid": True,
        "components": {},
        "issues": []
    }
    
    try:
        # Validate settings
        settings_valid = _validate_settings()
        validation_result["components"]["settings"] = {
            "valid": settings_valid,
            "message": "Settings validated" if settings_valid else "Settings validation failed"
        }
        
        if not settings_valid:
            validation_result["valid"] = False
            validation_result["issues"].append("Settings validation failed")
        
        # Validate feature flags
        feature_status = _validate_features()
        validation_result["components"]["feature_flags"] = feature_status
        
        if not feature_status["valid"]:
            validation_result["valid"] = False
            validation_result["issues"].extend(feature_status["issues"])
        
        # Validate constants (basic sanity checks)
        from .constants import AIRPORT_CODES, MINIMUM_PRICE_FLOORS
        
        if not AIRPORT_CODES:
            validation_result["valid"] = False
            validation_result["issues"].append("No airport codes configured")
            validation_result["components"]["constants"] = {"valid": False, "message": "Missing airport codes"}
        else:
            validation_result["components"]["constants"] = {"valid": True, "message": f"{len(AIRPORT_CODES)} airports configured"}
        
        # Check price floors for all airports
        missing_floors = [code for code in AIRPORT_CODES if code not in MINIMUM_PRICE_FLOORS]
        if missing_floors:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Missing price floors for airports: {missing_floors}")
        
        logger.info(
            "Configuration validation completed",
            extra={
                "overall_valid": validation_result["valid"],
                "component_count": len(validation_result["components"]),
                "issue_count": len(validation_result["issues"])
            }
        )
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Validation error: {str(e)}")
        logger.error(f"Configuration validation failed: {str(e)}")
    
    return validation_result

def get_configuration_summary() -> dict:
    """
    Get a summary of the current configuration for reporting purposes.
    
    Returns:
        dict: Configuration summary including counts and status
        
    Reference: [Absolutely Essential: Must be able to explain AI decisions to non-technical management]
    """
    from .settings import get_settings
    from .feature_flags import get_feature_flags, get_enabled_features
    
    settings = get_settings()
    feature_flags = get_feature_flags()
    
    return {
        "version": __version__,
        "environment": settings.ENVIRONMENT,
        "airports": {
            "total": len(AIRPORT_CODES),
            "sample": list(AIRPORT_CODES.keys())[:5] if AIRPORT_CODES else []
        },
        "features": {
            "total": len(get_enabled_features()) + len(getattr(feature_flags, 'disabled_features', [])),
            "enabled": len(get_enabled_features()),
            "enabled_list": get_enabled_features()
        },
        "business_rules": {
            "price_floors_configured": len(MINIMUM_PRICE_FLOORS),
            "price_caps_configured": len(MAXIMUM_PRICE_CAPS),
            "tax_rates_configured": len(TAX_RATES_BY_STATE)
        },
        "status": "active" if settings.ENVIRONMENT != "disabled" else "disabled"
    }

# Helper function to check if configuration supports a specific airport
def supports_airport(airport_code: str) -> bool:
    """
    Check if the system is configured to support a specific airport.
    
    Args:
        airport_code (str): IATA airport code (e.g., "JFK", "LAX")
        
    Returns:
        bool: True if airport is supported, False otherwise
        
    Reference: [Core Responsibilities: 1. Commercial & Pricing AI - Airport-level demand forecasting models]
    """
    return airport_code.upper() in AIRPORT_CODES

def get_airport_configuration(airport_code: str) -> dict:
    """
    Get complete configuration for a specific airport.
    
    Args:
        airport_code (str): IATA airport code
        
    Returns:
        dict: Airport configuration including all relevant settings
        
    Reference: [Core Responsibilities: Airport-level models and configurations]
    """
    if not supports_airport(airport_code):
        raise ValueError(f"Airport {airport_code} not configured in the system")
    
    airport_code = airport_code.upper()
    
    return {
        "code": airport_code,
        "name": AIRPORT_NAMES.get(airport_code, "Unknown"),
        "timezone": AIRPORT_TIMEZONES.get(airport_code, "UTC"),
        "region": AIRPORT_REGIONS.get(airport_code, "Unknown"),
        "price_floor": MINIMUM_PRICE_FLOORS.get(airport_code, 0.0),
        "price_cap": MAXIMUM_PRICE_CAPS.get(airport_code, float('inf')),
        "cpc_limit": DEFAULT_CPC_LIMITS.get(airport_code, 5.0),
        "tax_rate": TAX_RATES_BY_STATE.get(AIRPORT_REGIONS.get(airport_code, ""), 0.0),
        "features_enabled": {
            "dynamic_pricing": is_feature_enabled(FEATURE_DYNAMIC_PRICING),
            "competitor_monitoring": is_feature_enabled(FEATURE_COMPETITOR_MONITORING)
        }
    }

# Initialize the module
_initialize_config_module()

# Module metadata
module_metadata = {
    "name": "core.config",
    "version": __version__,
    "description": __description__,
    "exports_count": len(__all__),
    "initialized": True,
}

def get_module_info() -> dict:
    """
    Get information about the configuration module.
    
    Returns:
        dict: Module metadata
    """
    return module_metadata.copy()

# Export initialization status
INITIALIZATION_STATUS = {
    "module": "config",
    "version": __version__,
    "initialized": True,
    "validation_passed": validate_full_configuration()["valid"]
}
