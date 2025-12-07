"""
Main Application Configuration Loader
Project: AI Commercial Platform for Airport Parking

This module handles loading, validation, and management of all application settings
from environment variables, configuration files, and defaults.

Reference: [Technical Stack: Full system integration requirements]
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import pydantic
from pydantic import Field, validator, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    username: str
    password: str
    database: str
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 5

@dataclass
class APIConfig:
    """External API configuration."""
    google_ads_api_key: Optional[str] = None
    meta_ads_api_key: Optional[str] = None
    bing_ads_api_key: Optional[str] = None
    mailchimp_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    stripe_api_key: Optional[str] = None
    opayo_api_key: Optional[str] = None
    
    # Rate limiting
    default_rate_limit: int = 100  # requests per minute
    burst_rate_limit: int = 150  # burst capacity

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    sentry_dsn: Optional[str] = None
    new_relic_license_key: Optional[str] = None
    log_retention_days: int = 30
    metrics_retention_days: int = 90

class Settings(BaseSettings):
    """
    Main application settings class.
    Loads configuration from environment variables with validation.
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in environment
    )
    
    # ========== Core Application Settings ==========
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    APP_NAME: str = "AI Commercial Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO
    SECRET_KEY: str = Field(default="", min_length=32)
    API_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # ========== Database Settings ==========
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 3306
    DATABASE_USER: str = "root"
    DATABASE_PASSWORD: str = ""
    DATABASE_NAME: str = "airport_parking_ai"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # ========== Redis/Cache Settings ==========
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    
    # ========== External Service Settings ==========
    GOOGLE_ADS_API_KEY: Optional[str] = None
    META_ADS_API_KEY: Optional[str] = None
    BING_ADS_API_KEY: Optional[str] = None
    MAILCHIMP_API_KEY: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    STRIPE_API_KEY: Optional[str] = None
    OPAYO_API_KEY: Optional[str] = None
    
    # ========== AI/ML Model Settings ==========
    MODEL_CACHE_DIR: str = "./data/models"
    EMBEDDINGS_CACHE_DIR: str = "./data/embeddings"
    TRAINING_DATA_DIR: str = "./data/processed/training_sets"
    PREDICTION_BATCH_SIZE: int = 1000
    MODEL_RETRAIN_DAYS: int = 7
    
    # ========== Pricing AI Settings ==========
    PRICING_UPDATE_INTERVAL_MINUTES: int = 15
    MAX_PRICE_CHANGE_PERCENT: float = 20.0  # Max 20% change in one update
    MINIMUM_PRICE_CHANGE: float = 0.50  # Minimum $0.50 change
    SAFETY_OVERRIDE_ENABLED: bool = True
    
    # ========== Marketing AI Settings ==========
    MARKETING_BUDGET_UPDATE_HOURS: int = 1
    MIN_CHANNEL_BUDGET: float = 50.0  # Minimum $50 per channel per day
    MAX_ROAS_TARGET: float = 10.0  # Target 10x return on ad spend
    MIN_ROAS_THRESHOLD: float = 3.0  # Stop spending if ROAS < 3x
    
    # ========== Competitor Intel Settings ==========
    COMPETITOR_SCRAPE_INTERVAL_MINUTES: int = 30
    MAX_CONCURRENT_SCRAPERS: int = 5
    PROXY_ROTATION_ENABLED: bool = True
    
    # ========== Franchisee AI Settings ==========
    FPS_UPDATE_INTERVAL_HOURS: int = 24
    FRAUD_DETECTION_CONFIDENCE_THRESHOLD: float = 0.85
    ALERT_COOLDOWN_MINUTES: int = 30
    
    # ========== Website AI Settings ==========
    PERSONALIZATION_UPDATE_SECONDS: int = 30
    ABANDONMENT_PREDICTION_THRESHOLD: float = 0.7
    MAX_UX_VARIANTS: int = 5
    
    # ========== Call Center AI Settings ==========
    CALL_CENTER_WORKER_COUNT: int = 10
    VOICE_AI_TIMEOUT_SECONDS: int = 30
    MINIMUM_CALL_RESOLUTION_RATE: float = 0.7  # 70% target
    
    # ========== Monitoring & Alerting ==========
    PROMETHEUS_ENABLED: bool = True
    GRAFANA_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None
    ALERT_WEBHOOK_URL: Optional[str] = None
    LOG_RETENTION_DAYS: int = 30
    
    # ========== PHP Integration Settings ==========
    PHP_API_BASE_URL: str = "http://localhost:8000"
    PHP_API_TIMEOUT_SECONDS: int = 10
    PHP_CACHE_ENABLED: bool = True
    PHP_FALLBACK_ENABLED: bool = True
    
    # ========== Security Settings ==========
    API_RATE_LIMIT: int = 100  # requests per minute
    JWT_SECRET: str = Field(default="", min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    
    # Validators
    @validator("SECRET_KEY", "JWT_SECRET")
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret keys are set in production."""
        if os.getenv("ENVIRONMENT") == Environment.PRODUCTION and len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters in production")
        return v
    
    @validator("ALLOWED_HOSTS", "CORS_ORIGINS")
    def validate_hosts(cls, v: List[str]) -> List[str]:
        """Ensure at least one host is specified."""
        if not v:
            raise ValueError("At least one host must be specified")
        return v
    
    @validator("MAX_PRICE_CHANGE_PERCENT")
    def validate_price_change(cls, v: float) -> float:
        """Ensure price change percentage is reasonable."""
        if v < 0 or v > 100:
            raise ValueError("Price change percentage must be between 0 and 100")
        return v
    
    @root_validator
    def validate_production_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation for production environment."""
        env = values.get("ENVIRONMENT")
        
        if env == Environment.PRODUCTION:
            # In production, ensure critical settings are configured
            if not values.get("SECRET_KEY") or len(values.get("SECRET_KEY", "")) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
            
            if not values.get("DATABASE_PASSWORD"):
                raise ValueError("DATABASE_PASSWORD must be set in production")
            
            # Check external API keys for production
            if not values.get("STRIPE_API_KEY"):
                raise ValueError("STRIPE_API_KEY must be set in production")
            
            # Ensure monitoring is enabled in production
            if not values.get("PROMETHEUS_ENABLED"):
                raise ValueError("Prometheus monitoring must be enabled in production")
        
        return values
    
    # Property methods for convenience
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT == Environment.STAGING
    
    @property
    def database_config(self) -> DatabaseConfig:
        """Get database configuration as a structured object."""
        return DatabaseConfig(
            host=self.DATABASE_HOST,
            port=self.DATABASE_PORT,
            username=self.DATABASE_USER,
            password=self.DATABASE_PASSWORD,
            database=self.DATABASE_NAME,
            pool_size=self.DATABASE_POOL_SIZE,
            max_overflow=self.DATABASE_MAX_OVERFLOW
        )
    
    @property
    def redis_config(self) -> RedisConfig:
        """Get Redis configuration as a structured object."""
        return RedisConfig(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            password=self.REDIS_PASSWORD,
            db=self.REDIS_DB
        )
    
    @property
    def api_config(self) -> APIConfig:
        """Get API configuration as a structured object."""
        return APIConfig(
            google_ads_api_key=self.GOOGLE_ADS_API_KEY,
            meta_ads_api_key=self.META_ADS_API_KEY,
            bing_ads_api_key=self.BING_ADS_API_KEY,
            mailchimp_api_key=self.MAILCHIMP_API_KEY,
            twilio_account_sid=self.TWILIO_ACCOUNT_SID,
            twilio_auth_token=self.TWILIO_AUTH_TOKEN,
            stripe_api_key=self.STRIPE_API_KEY,
            opayo_api_key=self.OPAYO_API_KEY
        )
    
    @property
    def monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration as a structured object."""
        return MonitoringConfig(
            prometheus_enabled=self.PROMETHEUS_ENABLED,
            grafana_enabled=self.GRAFANA_ENABLED,
            sentry_dsn=self.SENTRY_DSN,
            log_retention_days=self.LOG_RETENTION_DAYS
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary, excluding sensitive data."""
        data = self.model_dump()
        
        # Remove sensitive information
        sensitive_fields = [
            "SECRET_KEY", "DATABASE_PASSWORD", "REDIS_PASSWORD",
            "GOOGLE_ADS_API_KEY", "META_ADS_API_KEY", "BING_ADS_API_KEY",
            "MAILCHIMP_API_KEY", "TWILIO_AUTH_TOKEN", "STRIPE_API_KEY",
            "OPAYO_API_KEY", "JWT_SECRET", "SENTRY_DSN"
        ]
        
        for field in sensitive_fields:
            if field in data and data[field]:
                data[field] = "***REDACTED***"
        
        return data
    
    def get_safe_summary(self) -> Dict[str, Any]:
        """Get a safe summary of settings for logging and debugging."""
        return {
            "app_name": self.APP_NAME,
            "version": self.APP_VERSION,
            "environment": self.ENVIRONMENT.value,
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL.value,
            "database": {
                "host": self.DATABASE_HOST,
                "port": self.DATABASE_PORT,
                "database": self.DATABASE_NAME,
                "pool_size": self.DATABASE_POOL_SIZE
            },
            "redis": {
                "host": self.REDIS_HOST,
                "port": self.REDIS_PORT,
                "db": self.REDIS_DB
            },
            "ai_settings": {
                "pricing_update_minutes": self.PRICING_UPDATE_INTERVAL_MINUTES,
                "marketing_update_hours": self.MARKETING_BUDGET_UPDATE_HOURS,
                "competitor_scrape_minutes": self.COMPETITOR_SCRAPE_INTERVAL_MINUTES,
                "model_retrain_days": self.MODEL_RETRAIN_DAYS
            },
            "php_integration": {
                "base_url": self.PHP_API_BASE_URL,
                "cache_enabled": self.PHP_CACHE_ENABLED,
                "fallback_enabled": self.PHP_FALLBACK_ENABLED
            },
            "monitoring": {
                "prometheus_enabled": self.PROMETHEUS_ENABLED,
                "grafana_enabled": self.GRAFANA_ENABLED,
                "log_retention_days": self.LOG_RETENTION_DAYS
            }
        }

# Global settings instance
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Get the global settings instance.
    Creates it if it doesn't exist.
    
    Returns:
        Settings: The global settings instance
        
    Reference: [Required Experience: Production engineering role]
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = Settings()
    
    return _settings_instance

def initialize_settings(env_file: Optional[str] = None) -> Settings:
    """
    Initialize settings from environment file.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        Settings: Initialized settings instance
        
    Reference: [Technical Stack: Environment configuration management]
    """
    global _settings_instance
    
    if env_file:
        # Load specific environment file
        load_dotenv(env_file)
    
    # Create new settings instance
    _settings_instance = Settings()
    
    # Log initialization
    import logging
    logger = logging.getLogger("settings")
    logger.info(
        f"Settings initialized for {_settings_instance.ENVIRONMENT.value} environment",
        extra=_settings_instance.get_safe_summary()
    )
    
    return _settings_instance

def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Returns:
        Settings: Reloaded settings instance
        
    Reference: [Absolutely Essential: Design with safety, logging, overrides]
    """
    global _settings_instance
    
    # Clear the instance to force reload
    _settings_instance = None
    
    # Reload environment variables
    load_dotenv(override=True)
    
    # Get new instance
    return get_settings()

def validate_settings() -> bool:
    """
    Validate that all required settings are properly configured.
    
    Returns:
        bool: True if settings are valid, False otherwise
        
    Reference: [Required Experience: Production AI/data engineering]
    """
    try:
        settings = get_settings()
        
        # Basic validation
        if not settings.APP_NAME:
            return False
        
        if not settings.DATABASE_HOST:
            return False
        
        if not settings.DATABASE_NAME:
            return False
        
        # Environment-specific validation
        if settings.is_production:
            if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
                return False
            
            if not settings.DATABASE_PASSWORD:
                return False
        
        # Validate directory paths exist or can be created
        directories = [
            settings.MODEL_CACHE_DIR,
            settings.EMBEDDINGS_CACHE_DIR,
            settings.TRAINING_DATA_DIR
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception:
                return False
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger("settings")
        logger.error(f"Settings validation failed: {str(e)}")
        return False

def export_settings_to_file(filepath: str, format: str = "json") -> bool:
    """
    Export current settings to a file.
    
    Args:
        filepath: Path to output file
        format: Output format (json, yaml)
        
    Returns:
        bool: True if export successful, False otherwise
        
    Reference: [Absolutely Essential: Must be able to explain AI decisions to management]
    """
    try:
        settings = get_settings()
        data = settings.to_dict()
        
        with open(filepath, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False)
            else:  # default to JSON
                json.dump(data, f, indent=2)
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger("settings")
        logger.error(f"Failed to export settings: {str(e)}")
        return False

def load_settings_from_file(filepath: str, format: str = "json") -> bool:
    """
    Load settings from a configuration file.
    
    Args:
        filepath: Path to configuration file
        format: File format (json, yaml)
        
    Returns:
        bool: True if load successful, False otherwise
        
    Reference: [Technical Stack: Configuration management]
    """
    try:
        with open(filepath, 'r') as f:
            if format.lower() == "yaml":
                data = yaml.safe_load(f)
            else:  # default to JSON
                data = json.load(f)
        
        # Set environment variables from loaded data
        for key, value in data.items():
            if value is not None:
                os.environ[key] = str(value)
        
        # Reload settings
        reload_settings()
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger("settings")
        logger.error(f"Failed to load settings from file: {str(e)}")
        return False

# Initialize settings on module import
initialize_settings()
