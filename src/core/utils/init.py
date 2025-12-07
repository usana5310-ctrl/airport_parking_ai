"""
Utilities Module - Initialization File
Project: AI Commercial Platform for Airport Parking

This module provides shared utilities for logging, validation, safety overrides,
date handling, and other common functionality used across the AI platform.

Reference: [Absolutely Essential: Design with safety, logging, overrides, and compliance]
"""

__version__ = "1.0.0"
__description__ = "Shared utilities for AI Commercial Platform"

# Import utility components
from .logger import (
    # Logging setup and configuration
    setup_logging,
    configure_logging,
    get_logger,
    
    # Structured logging
    StructuredLogger,
    LogLevel,
    LogContext,
    
    # Log handlers
    JSONFormatter,
    DatabaseHandler,
    SlackHandler,
    
    # Log utilities
    log_execution_time,
    log_exception,
    log_metric,
    
    # Constants
    DEFAULT_LOG_FORMAT,
    DEFAULT_DATE_FORMAT,
)

from .validators import (
    # Input validation
    validate_airport_code,
    validate_price,
    validate_date_range,
    validate_email,
    validate_phone,
    validate_license_plate,
    
    # Business rule validation
    validate_price_floor,
    validate_price_cap,
    validate_cpc_protection,
    validate_booking_dates,
    validate_service_type,
    
    # Data validation
    validate_dict_structure,
    validate_json_schema,
    sanitize_input,
    
    # Validation utilities
    ValidationError,
    ValidationResult,
    ValidationRule,
    
    # Constants
    PRICE_VALIDATION_RULES,
    DATE_VALIDATION_RULES,
    EMAIL_REGEX,
    PHONE_REGEX,
)

from .safety_overrides import (
    # Safety system
    SafetySystem,
    SafetyLevel,
    SafetyOverride,
    
    # Override management
    enable_override,
    disable_override,
    check_override,
    get_active_overrides,
    
    # Emergency controls
    emergency_stop,
    emergency_resume,
    is_emergency_stopped,
    
    # Safety validation
    validate_safety_rules,
    check_safety_constraints,
    
    # Override utilities
    OverrideError,
    OverrideManager,
)

from .date_helpers import (
    # Date/time utilities
    now,
    today,
    parse_date,
    format_date,
    
    # Timezone handling
    get_airport_timezone,
    convert_to_utc,
    convert_from_utc,
    get_local_time,
    
    # Date calculations
    add_business_days,
    get_business_days_between,
    is_business_day,
    is_holiday,
    
    # Time utilities
    parse_time,
    format_time,
    calculate_duration,
    
    # Date ranges
    get_date_range,
    get_rolling_window,
    split_date_range,
    
    # Constants
    BUSINESS_HOURS,
    WEEKEND_DAYS,
)

# Import additional utilities
from .security import (
    encrypt_data,
    decrypt_data,
    hash_password,
    verify_password,
    generate_api_key,
    validate_api_key,
    sanitize_sql,
    prevent_sql_injection,
)

from .cache import (
    CacheManager,
    MemoryCache,
    RedisCache,
    get_cache,
    set_cache,
    delete_cache,
    clear_cache,
    cache_result,
)

from .metrics import (
    MetricsCollector,
    track_metric,
    increment_counter,
    record_gauge,
    measure_latency,
    get_metrics_summary,
)

from .serializers import (
    serialize_model,
    deserialize_model,
    to_json,
    from_json,
    to_csv,
    from_csv,
    compress_data,
    decompress_data,
)

from .file_utils import (
    read_json_file,
    write_json_file,
    read_yaml_file,
    write_yaml_file,
    ensure_directory,
    safe_delete,
    backup_file,
)

from .network import (
    make_api_request,
    retry_on_failure,
    exponential_backoff,
    validate_url,
    parse_query_params,
)

# Public API - expose only what should be used externally
__all__ = [
    # Logging
    "setup_logging",
    "configure_logging",
    "get_logger",
    "StructuredLogger",
    "LogLevel",
    "LogContext",
    "JSONFormatter",
    "DatabaseHandler",
    "SlackHandler",
    "log_execution_time",
    "log_exception",
    "log_metric",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    
    # Validation
    "validate_airport_code",
    "validate_price",
    "validate_date_range",
    "validate_email",
    "validate_phone",
    "validate_license_plate",
    "validate_price_floor",
    "validate_price_cap",
    "validate_cpc_protection",
    "validate_booking_dates",
    "validate_service_type",
    "validate_dict_structure",
    "validate_json_schema",
    "sanitize_input",
    "ValidationError",
    "ValidationResult",
    "ValidationRule",
    "PRICE_VALIDATION_RULES",
    "DATE_VALIDATION_RULES",
    "EMAIL_REGEX",
    "PHONE_REGEX",
    
    # Safety overrides
    "SafetySystem",
    "SafetyLevel",
    "SafetyOverride",
    "enable_override",
    "disable_override",
    "check_override",
    "get_active_overrides",
    "emergency_stop",
    "emergency_resume",
    "is_emergency_stopped",
    "validate_safety_rules",
    "check_safety_constraints",
    "OverrideError",
    "OverrideManager",
    
    # Date helpers
    "now",
    "today",
    "parse_date",
    "format_date",
    "get_airport_timezone",
    "convert_to_utc",
    "convert_from_utc",
    "get_local_time",
    "add_business_days",
    "get_business_days_between",
    "is_business_day",
    "is_holiday",
    "parse_time",
    "format_time",
    "calculate_duration",
    "get_date_range",
    "get_rolling_window",
    "split_date_range",
    "BUSINESS_HOURS",
    "WEEKEND_DAYS",
    
    # Security
    "encrypt_data",
    "decrypt_data",
    "hash_password",
    "verify_password",
    "generate_api_key",
    "validate_api_key",
    "sanitize_sql",
    "prevent_sql_injection",
    
    # Cache
    "CacheManager",
    "MemoryCache",
    "RedisCache",
    "get_cache",
    "set_cache",
    "delete_cache",
    "clear_cache",
    "cache_result",
    
    # Metrics
    "MetricsCollector",
    "track_metric",
    "increment_counter",
    "record_gauge",
    "measure_latency",
    "get_metrics_summary",
    
    # Serializers
    "serialize_model",
    "deserialize_model",
    "to_json",
    "from_json",
    "to_csv",
    "from_csv",
    "compress_data",
    "decompress_data",
    
    # File utilities
    "read_json_file",
    "write_json_file",
    "read_yaml_file",
    "write_yaml_file",
    "ensure_directory",
    "safe_delete",
    "backup_file",
    
    # Network
    "make_api_request",
    "retry_on_failure",
    "exponential_backoff",
    "validate_url",
    "parse_query_params",
    
    # Module info
    "__version__",
    "__description__",
]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _initialize_utils_module():
    """
    Initialize the utilities module.
    Sets up logging and validates all utility components.
    
    Reference: [Required Experience: Production engineering role]
    """
    import logging
    
    # Set up logging first (critical for debugging)
    from .logger import setup_logging
    setup_logging()
    
    logger = logging.getLogger("utils")
    
    try:
        # Validate all utility components
        validation_results = _validate_utility_components()
        
        if validation_results["all_valid"]:
            logger.info(
                "Utilities module initialized successfully",
                extra={
                    "module": "utils",
                    "version": __version__,
                    "components_validated": len(validation_results["components"]),
                    "validation_passed": True
                }
            )
        else:
            logger.warning(
                "Utilities module initialized with validation warnings",
                extra={
                    "module": "utils",
                    "version": __version__,
                    "validation_errors": validation_results["errors"],
                    "validation_passed": False
                }
            )
            
    except Exception as e:
        logger.error(
            f"Failed to initialize utilities module: {str(e)}",
            extra={"module": "utils", "error": str(e)}
        )
        raise

def _validate_utility_components() -> dict:
    """
    Validate all utility components for proper configuration.
    
    Returns:
        dict: Validation results
        
    Reference: [Absolutely Essential: Design with safety, logging, overrides]
    """
    validation_results = {
        "all_valid": True,
        "components": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        # Validate logger configuration
        try:
            from .logger import get_logger
            test_logger = get_logger("utils.validation")
            test_logger.debug("Logger test message")
            validation_results["components"]["logger"] = {
                "valid": True,
                "message": "Logger configured and working"
            }
        except Exception as e:
            validation_results["all_valid"] = False
            validation_results["errors"].append(f"Logger validation failed: {str(e)}")
            validation_results["components"]["logger"] = {
                "valid": False,
                "message": f"Logger error: {str(e)}"
            }
        
        # Validate safety system
        try:
            from .safety_overrides import SafetySystem
            safety_system = SafetySystem()
            validation_results["components"]["safety_system"] = {
                "valid": True,
                "message": "Safety system initialized",
                "overrides_count": len(safety_system.get_active_overrides())
            }
        except Exception as e:
            validation_results["warnings"].append(f"Safety system warning: {str(e)}")
            validation_results["components"]["safety_system"] = {
                "valid": False,
                "message": f"Safety system warning: {str(e)}"
            }
        
        # Validate date helpers
        try:
            from .date_helpers import now, today, parse_date
            current_time = now()
            current_date = today()
            parsed = parse_date("2024-01-01")
            
            validation_results["components"]["date_helpers"] = {
                "valid": True,
                "message": "Date helpers working",
                "current_time": current_time.isoformat(),
                "current_date": current_date.isoformat()
            }
        except Exception as e:
            validation_results["warnings"].append(f"Date helpers warning: {str(e)}")
            validation_results["components"]["date_helpers"] = {
                "valid": False,
                "message": f"Date helpers warning: {str(e)}"
            }
        
        # Validate validators
        try:
            from .validators import validate_airport_code, validate_price
            airport_valid = validate_airport_code("JFK")
            price_valid = validate_price(100.00)
            
            validation_results["components"]["validators"] = {
                "valid": True,
                "message": "Validators working",
                "airport_validation": airport_valid,
                "price_validation": price_valid
            }
        except Exception as e:
            validation_results["warnings"].append(f"Validators warning: {str(e)}")
            validation_results["components"]["validators"] = {
                "valid": False,
                "message": f"Validators warning: {str(e)}"
            }
        
        # Validate security utilities
        try:
            from .security import encrypt_data, decrypt_data
            test_data = "test_secret"
            encrypted = encrypt_data(test_data)
            decrypted = decrypt_data(encrypted)
            
            validation_results["components"]["security"] = {
                "valid": decrypted == test_data,
                "message": "Security utilities working",
                "encryption_test": decrypted == test_data
            }
            
            if decrypted != test_data:
                validation_results["warnings"].append("Encryption/decryption test failed")
        except Exception as e:
            validation_results["warnings"].append(f"Security utilities warning: {str(e)}")
            validation_results["components"]["security"] = {
                "valid": False,
                "message": f"Security utilities warning: {str(e)}"
            }
        
    except Exception as e:
        validation_results["all_valid"] = False
        validation_results["errors"].append(f"Utility validation error: {str(e)}")
    
    return validation_results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_utility_info() -> dict:
    """
    Get information about the utilities module.
    
    Returns:
        dict: Module information
        
    Reference: [Absolutely Essential: Must be able to explain AI decisions to management]
    """
    return {
        "module": "core.utils",
        "version": __version__,
        "description": __description__,
        "components": {
            "logging": "Structured logging with multiple handlers",
            "validation": "Input and business rule validation",
            "safety": "Safety overrides and emergency controls",
            "dates": "Timezone-aware date/time handling",
            "security": "Encryption and data protection",
            "cache": "Caching utilities",
            "metrics": "Performance metrics collection",
            "serialization": "Data serialization formats",
            "files": "File system utilities",
            "network": "HTTP and API utilities"
        },
        "initialized": True,
        "validation": _validate_utility_components()
    }

def health_check() -> dict:
    """
    Perform health check of utility components.
    
    Returns:
        dict: Health check results
        
    Reference: [Required Experience: Production engineering role]
    """
    import time
    
    health_result = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check logging
        try:
            from .logger import get_logger
            logger = get_logger("utils.health")
            logger.debug("Health check logging test")
            health_result["checks"]["logging"] = {
                "status": "healthy",
                "message": "Logging system operational"
            }
        except Exception as e:
            health_result["status"] = "degraded"
            health_result["checks"]["logging"] = {
                "status": "unhealthy",
                "message": f"Logging error: {str(e)}"
            }
            health_result["warnings"].append(f"Logging system issue: {str(e)}")
        
        # Check safety system
        try:
            from .safety_overrides import SafetySystem
            safety = SafetySystem()
            overrides = safety.get_active_overrides()
            health_result["checks"]["safety_system"] = {
                "status": "healthy",
                "message": f"Safety system operational with {len(overrides)} active overrides",
                "active_overrides": len(overrides)
            }
        except Exception as e:
            health_result["status"] = "degraded"
            health_result["checks"]["safety_system"] = {
                "status": "unhealthy",
                "message": f"Safety system error: {str(e)}"
            }
            health_result["errors"].append(f"Safety system error: {str(e)}")
        
        # Check date/time
        try:
            from .date_helpers import now, get_airport_timezone
            current_time = now()
            tz = get_airport_timezone("JFK")
            health_result["checks"]["date_utils"] = {
                "status": "healthy",
                "message": f"Date utilities operational, current time: {current_time}",
                "current_time": current_time.isoformat(),
                "timezone_test": tz is not None
            }
        except Exception as e:
            health_result["status"] = "degraded"
            health_result["checks"]["date_utils"] = {
                "status": "unhealthy",
                "message": f"Date utilities error: {str(e)}"
            }
            health_result["warnings"].append(f"Date utilities issue: {str(e)}")
        
        # Check validation
        try:
            from .validators import validate_price, ValidationError
            valid = validate_price(100.0)
            health_result["checks"]["validation"] = {
                "status": "healthy",
                "message": "Validation system operational",
                "price_validation_test": valid
            }
        except Exception as e:
            health_result["status"] = "degraded"
            health_result["checks"]["validation"] = {
                "status": "unhealthy",
                "message": f"Validation error: {str(e)}"
            }
            health_result["warnings"].append(f"Validation system issue: {str(e)}")
        
        # Check cache if Redis available
        try:
            from ..config.settings import get_settings
            settings = get_settings()
            if settings.REDIS_HOST:
                from .cache import CacheManager
                cache = CacheManager()
                test_key = "health_check_test"
                cache.set(test_key, "test_value", 10)
                value = cache.get(test_key)
                cache.delete(test_key)
                
                health_result["checks"]["cache"] = {
                    "status": "healthy" if value == "test_value" else "degraded",
                    "message": f"Cache system {'operational' if value == 'test_value' else 'partial'}",
                    "cache_test": value == "test_value"
                }
            else:
                health_result["checks"]["cache"] = {
                    "status": "healthy",
                    "message": "Cache system (memory only) operational"
                }
        except Exception as e:
            health_result["status"] = "degraded"
            health_result["checks"]["cache"] = {
                "status": "unhealthy",
                "message": f"Cache error: {str(e)}"
            }
            health_result["warnings"].append(f"Cache system issue: {str(e)}")
        
        # Calculate overall status
        healthy_checks = sum(1 for check in health_result["checks"].values() if check["status"] == "healthy")
        total_checks = len(health_result["checks"])
        
        health_result["summary"] = {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "unhealthy_checks": total_checks - healthy_checks,
            "success_rate": (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        }
        
        # Update overall status based on errors
        if health_result["errors"]:
            health_result["status"] = "unhealthy"
        elif health_result["warnings"] or healthy_checks < total_checks:
            health_result["status"] = "degraded"
        
        # Log health check result
        from .logger import get_logger
        logger = get_logger("utils.health")
        logger.info(
            "Utilities health check completed",
            extra={
                "status": health_result["status"],
                "healthy_checks": healthy_checks,
                "total_checks": total_checks,
                "errors": len(health_result["errors"]),
                "warnings": len(health_result["warnings"])
            }
        )
        
    except Exception as e:
        health_result["status"] = "error"
        health_result["errors"].append(f"Health check failed: {str(e)}")
        
        from .logger import get_logger
        logger = get_logger("utils.health")
        logger.error(f"Utilities health check failed: {str(e)}")
    
    return health_result

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_environment():
    """
    Set up the complete utilities environment.
    Call this at application startup.
    
    Reference: [Required Experience: Production engineering role]
    """
    from .logger import setup_logging, get_logger
    from .safety_overrides import SafetySystem
    
    logger = get_logger("utils.setup")
    
    try:
        # 1. Set up logging
        setup_logging()
        
        # 2. Initialize safety system
        safety_system = SafetySystem()
        
        # 3. Perform health check
        health = health_check()
        
        logger.info(
            "Utilities environment setup completed",
            extra={
                "logging_configured": True,
                "safety_system_initialized": True,
                "health_status": health["status"]
            }
        )
        
        return {
            "success": True,
            "logging_configured": True,
            "safety_system_initialized": True,
            "health_status": health["status"]
        }
        
    except Exception as e:
        logger.error(f"Utilities environment setup failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def emergency_protocol(action: str, **kwargs):
    """
    Execute emergency protocol for the utilities system.
    
    Args:
        action: Emergency action (stop, resume, validate, status)
        **kwargs: Action-specific parameters
        
    Returns:
        dict: Protocol execution result
        
    Reference: [Absolutely Essential: Design with safety, overrides]
    """
    from .logger import get_logger
    from .safety_overrides import emergency_stop, emergency_resume, is_emergency_stopped
    
    logger = get_logger("utils.emergency")
    
    try:
        if action == "stop":
            reason = kwargs.get("reason", "Emergency stop requested")
            result = emergency_stop(reason)
            logger.critical(
                "Emergency stop executed",
                extra={"reason": reason, "action": "stop"}
            )
            return result
            
        elif action == "resume":
            result = emergency_resume()
            logger.warning(
                "Emergency resume executed",
                extra={"action": "resume"}
            )
            return result
            
        elif action == "status":
            stopped = is_emergency_stopped()
            return {
                "emergency_stopped": stopped,
                "timestamp": now().isoformat()
            }
            
        elif action == "validate":
            from .validators import validate_safety_rules
            result = validate_safety_rules()
            logger.info(
                "Safety rules validation executed",
                extra={"action": "validate", "valid": result["valid"]}
            )
            return result
            
        else:
            error_msg = f"Unknown emergency action: {action}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
            
    except Exception as e:
        logger.error(f"Emergency protocol failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def timed_operation(operation_name: str, logger_name: str = "utils.timing"):
    """
    Context manager for timing operations.
    
    Usage:
        with timed_operation("price_calculation") as timer:
            result = calculate_prices()
            timer.add_metadata({"items_processed": len(result)})
    
    Reference: [Required Experience: Production engineering role]
    """
    import time
    from .logger import get_logger
    
    logger = get_logger(logger_name)
    start_time = time.time()
    
    class Timer:
        def __init__(self):
            self.metadata = {}
        
        def add_metadata(self, metadata: dict):
            self.metadata.update(metadata)
    
    timer = Timer()
    
    try:
        yield timer
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Operation '{operation_name}' failed after {duration:.3f}s",
            extra={
                "operation": operation_name,
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e),
                **timer.metadata
            }
        )
        raise
    else:
        duration = time.time() - start_time
        logger.info(
            f"Operation '{operation_name}' completed in {duration:.3f}s",
            extra={
                "operation": operation_name,
                "duration_seconds": duration,
                "status": "completed",
                **timer.metadata
            }
        )

@contextmanager
def safety_override(context: str, reason: str = "Manual override"):
    """
    Context manager for temporary safety overrides.
    
    Usage:
        with safety_override("price_calculation", "Market anomaly detected") as override:
            # Safety rules are temporarily disabled
            result = calculate_aggressive_prices()
    
    Reference: [Absolutely Essential: Design with safety, overrides]
    """
    from .safety_overrides import enable_override, disable_override, SafetyLevel
    from .logger import get_logger
    
    logger = get_logger("utils.safety")
    
    override_id = enable_override(
        context=context,
        reason=reason,
        level=SafetyLevel.MEDIUM
    )
    
    logger.warning(
        f"Safety override enabled for '{context}'",
        extra={
            "override_id": override_id,
            "context": context,
            "reason": reason,
            "action": "enable"
        }
    )
    
    try:
        yield override_id
    finally:
        disable_override(override_id)
        logger.info(
            f"Safety override disabled for '{context}'",
            extra={
                "override_id": override_id,
                "context": context,
                "action": "disable"
            }
        )

# ============================================================================
# DECORATORS
# ============================================================================

def validate_input(*validation_rules):
    """
    Decorator for input validation.
    
    Usage:
        @validate_input(
            ("airport_code", validate_airport_code),
            ("price", lambda x: validate_price(x, min=10, max=1000))
        )
        def calculate_price(airport_code: str, price: float):
            pass
    
    Reference: [Core Responsibilities: Safe dynamic pricing automation]
    """
    def decorator(func):
        from functools import wraps
        from .validators import ValidationError
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Apply validation rules
            for param_name, validator in validation_rules:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}' with value '{value}'"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_operation(operation_name: str = None, log_level: str = "INFO"):
    """
    Decorator for operation logging.
    
    Usage:
        @log_operation("price_update", log_level="INFO")
        def update_prices():
            pass
    """
    def decorator(func):
        from functools import wraps
        from .logger import get_logger, LogLevel
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = get_logger(f"operation.{op_name}")
            
            # Log start
            logger.log(
                LogLevel[log_level.upper()].value,
                f"Operation '{op_name}' started",
                extra={
                    "operation": op_name,
                    "status": "started",
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                logger.log(
                    LogLevel[log_level.upper()].value,
                    f"Operation '{op_name}' completed successfully",
                    extra={
                        "operation": op_name,
                        "status": "completed",
                        "has_result": result is not None
                    }
                )
                
                return result
                
            except Exception as e:
                # Log failure
                logger.error(
                    f"Operation '{op_name}' failed",
                    extra={
                        "operation": op_name,
                        "status": "failed",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                raise
        
        return wrapper
    
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retry logic on failures.
    
    Usage:
        @retry_on_failure(max_retries=3, delay=1.0)
        def call_external_api():
            pass
    """
    def decorator(func):
        from functools import wraps
        import time
        from .logger import get_logger
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("utils.retry")
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Operation '{func.__name__}' failed after {max_retries} retries",
                            extra={
                                "operation": func.__name__,
                                "attempts": attempt + 1,
                                "error": str(e)
                            }
                        )
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(
                        f"Operation '{func.__name__}' failed, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})",
                        extra={
                            "operation": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "wait_time": wait_time,
                            "error": str(e)
                        }
                    )
                    
                    time.sleep(wait_time)
            
            raise RuntimeError(f"Max retries ({max_retries}) exceeded for {func.__name__}")
        
        return wrapper
    
    return decorator

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize the module
_initialize_utils_module()

# Module metadata
module_metadata = {
    "name": "core.utils",
    "version": __version__,
    "description": __description__,
    "initialized": True,
    "health_status": health_check()["status"],
}

def get_module_info() -> dict:
    """
    Get information about the utilities module.
    
    Returns:
        dict: Module metadata
    """
    return module_metadata.copy()

# Export initialization status
INITIALIZATION_STATUS = {
    "module": "utils",
    "version": __version__,
    "initialized": True,
    "health": health_check(),
}

# ============================================================================
# GLOBAL UTILITY INSTANCES
# ============================================================================

# Create global instances for commonly used utilities
_logger_instance = None
_safety_system_instance = None
_cache_manager_instance = None

def get_global_logger() -> "StructuredLogger":
    """Get global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        from .logger import StructuredLogger
        _logger_instance = StructuredLogger("global")
    return _logger_instance

def get_global_safety_system() -> "SafetySystem":
    """Get global safety system instance."""
    global _safety_system_instance
    if _safety_system_instance is None:
        from .safety_overrides import SafetySystem
        _safety_system_instance = SafetySystem()
    return _safety_system_instance

def get_global_cache_manager() -> "CacheManager":
    """Get global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        from .cache import CacheManager
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance

# ============================================================================
# STARTUP HOOKS
# ============================================================================

def register_startup_hooks():
    """
    Register startup hooks for utilities.
    Called by main application during startup.
    
    Reference: [Required Experience: Production engineering role]
    """
    from .logger import get_logger
    logger = get_logger("utils.startup")
    
    try:
        # Perform initial health check
        health = health_check()
        
        # Set up periodic health checks in production
        from ..config.settings import get_settings
        settings = get_settings()
        
        if settings.is_production:
            import threading
            import time
            
            def periodic_health_check():
                """Background thread for periodic health checks."""
                while True:
                    time.sleep(300)  # Every 5 minutes
                    try:
                        current_health = health_check()
                        if current_health["status"] != "healthy":
                            logger.warning(
                                "Periodic health check发现问题",
                                extra={"health_status": current_health["status"]}
                            )
                    except Exception as e:
                        logger.error(f"Periodic health check failed: {str(e)}")
            
            health_thread = threading.Thread(
                target=periodic_health_check,
                name="utils_health_monitor",
                daemon=True
            )
            health_thread.start()
            
            logger.info("Started periodic health monitoring")
        
        logger.info(
            "Utilities startup hooks registered",
            extra={
                "health_status": health["status"],
                "periodic_monitoring": settings.is_production
            }
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register startup hooks: {str(e)}")
        return False

# Register startup hooks
register_startup_hooks()
