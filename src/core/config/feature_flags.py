"""
Feature Flag Management System
Project: AI Commercial Platform for Airport Parking

This module manages feature toggles for safe deployment, gradual rollouts,
A/B testing, and emergency overrides of AI systems.

Reference: [Absolutely Essential: Design with safety, logging, overrides]
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import redis
import hashlib

from pydantic import BaseModel, Field, validator

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

class FeatureFlag(str, Enum):
    """Feature flag identifiers for all AI modules."""
    
    # ===== CORE INFRASTRUCTURE =====
    SYSTEM_MONITORING = "system_monitoring"
    REAL_TIME_LOGGING = "real_time_logging"
    PERFORMANCE_METRICS = "performance_metrics"
    
    # ===== PRICING AI FEATURES =====
    # Reference: [Core Responsibilities: 1. Commercial & Pricing AI]
    DYNAMIC_PRICING = "dynamic_pricing"
    DEMAND_FORECASTING = "demand_forecasting"
    COMPETITOR_RESPONSE = "competitor_response"
    PRICE_ELASTICITY_MODELS = "price_elasticity_models"
    PRICE_AB_TESTING = "price_ab_testing"
    SAFETY_OVERRIDES = "pricing_safety_overrides"
    
    # ===== MARKETING AI FEATURES =====
    # Reference: [Core Responsibilities: 2. Marketing & Multi-Channel Budget Optimisation]
    MARKETING_BUDGET_AUTOMATION = "marketing_budget_automation"
    ROAS_PREDICTION = "roas_prediction"
    MULTI_TOUCH_ATTRIBUTION = "multi_touch_attribution"
    GOOGLE_ADS_AUTOMATION = "google_ads_automation"
    META_ADS_AUTOMATION = "meta_ads_automation"
    EMAIL_MARKETING_AUTOMATION = "email_marketing_automation"
    
    # ===== COMPETITOR INTELLIGENCE FEATURES =====
    # Reference: [Core Responsibilities: 3. Competitor & Market Intelligence]
    COMPETITOR_PRICE_MONITORING = "competitor_price_monitoring"
    AUTOMATED_SCRAPING = "automated_scraping"
    PRICE_SHOCK_DETECTION = "price_shock_detection"
    PROMOTION_TRACKING = "promotion_tracking"
    
    # ===== FRANCHISEE AI FEATURES =====
    # Reference: [Core Responsibilities: 4. Franchisee Performance, Risk & Fraud Detection]
    FRANCHISEE_PERFORMANCE_SCORING = "franchisee_performance_scoring"
    DRIVER_RISK_PREDICTION = "driver_risk_prediction"
    FRAUD_DETECTION = "fraud_detection"
    REAL_TIME_ALERTS = "real_time_alerts"
    AUTOMATED_INTERVENTIONS = "automated_interventions"
    
    # ===== WEBSITE AI FEATURES =====
    # Reference: [Core Responsibilities: 5. Website Behaviour, Conversion & Personalisation AI]
    PRICE_SENSITIVITY_DETECTION = "price_sensitivity_detection"
    ABANDONMENT_PREDICTION = "abandonment_prediction"
    DYNAMIC_UX_PERSONALIZATION = "dynamic_ux_personalization"
    REAL_TIME_FUNNEL_OPTIMIZATION = "real_time_funnel_optimization"
    CTA_OPTIMIZATION = "cta_optimization"
    
    # ===== CALL CENTER AI FEATURES =====
    # Reference: [Core Responsibilities: 6. AI Call Centre (DRT & CRS Automation)]
    VOICE_AI_PROCESSING = "voice_ai_processing"
    AUTOMATED_CALL_HANDLING = "automated_call_handling"
    LATE_FEE_AUTOMATION = "late_fee_automation"
    BOOKING_MODIFICATION_AI = "booking_modification_ai"
    REFUND_PROCESSING_AI = "refund_processing_ai"
    
    # ===== PHP INTEGRATION FEATURES =====
    PHP_AI_INTEGRATION = "php_ai_integration"
    REAL_TIME_PRICE_UPDATES = "real_time_price_updates"
    CACHING_LAYER = "caching_layer"
    FALLBACK_MECHANISMS = "fallback_mechanisms"

class FeatureGroup(str, Enum):
    """Groups of related features."""
    CORE_INFRASTRUCTURE = "core_infrastructure"
    PRICING_AI = "pricing_ai"
    MARKETING_AI = "marketing_ai"
    COMPETITOR_INTEL = "competitor_intel"
    FRANCHISEE_AI = "franchisee_ai"
    WEBSITE_AI = "website_ai"
    CALL_CENTER_AI = "call_center_ai"
    PHP_INTEGRATION = "php_integration"
    SAFETY_FEATURES = "safety_features"

class RolloutStrategy(str, Enum):
    """Feature rollout strategies."""
    DISABLED = "disabled"           # Feature completely disabled
    ENABLED = "enabled"             # Feature fully enabled
    PERCENTAGE = "percentage"       # Percentage-based rollout
    WHITELIST = "whitelist"         # Enabled for specific airports/users
    AB_TEST = "ab_test"             # A/B testing
    CANARY = "canary"               # Gradual rollout with monitoring

class FeatureStatus(str, Enum):
    """Feature status for reporting."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ROLLING_OUT = "rolling_out"
    ROLLED_BACK = "rolled_back"
    EXPERIMENTAL = "experimental"

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FeatureConfiguration:
    """Configuration for a single feature flag."""
    feature: FeatureFlag
    enabled: bool = False
    rollout_strategy: RolloutStrategy = RolloutStrategy.DISABLED
    rollout_percentage: float = 0.0  # 0.0 to 1.0
    whitelist: Set[str] = field(default_factory=set)  # airport codes or user IDs
    airports: Set[str] = field(default_factory=set)   # specific airports
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active_for(self, identifier: str = None) -> bool:
        """Check if feature is active for a specific identifier."""
        if not self.enabled:
            return False
        
        # Check date range
        now = datetime.now()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        
        # Check based on rollout strategy
        if self.rollout_strategy == RolloutStrategy.DISABLED:
            return False
        elif self.rollout_strategy == RolloutStrategy.ENABLED:
            return True
        elif self.rollout_strategy == RolloutStrategy.PERCENTAGE:
            if not identifier:
                return False
            # Deterministic based on identifier hash
            hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
            return (hash_value % 100) / 100.0 < self.rollout_percentage
        elif self.rollout_strategy == RolloutStrategy.WHITELIST:
            return identifier in self.whitelist if identifier else False
        elif self.rollout_strategy == RolloutStrategy.CANARY:
            # Canary rollout for airports
            return identifier in self.airports if identifier else False
        elif self.rollout_strategy == RolloutStrategy.AB_TEST:
            # A/B test - check if in test group
            return identifier in self.whitelist if identifier else False
        
        return False

class FeatureFlagModel(BaseModel):
    """Pydantic model for feature flag validation."""
    feature: FeatureFlag
    enabled: bool = False
    rollout_strategy: RolloutStrategy = RolloutStrategy.DISABLED
    rollout_percentage: float = Field(0.0, ge=0.0, le=1.0)
    whitelist: List[str] = []
    airports: List[str] = []
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    @validator('airports')
    def validate_airport_codes(cls, v):
        """Validate airport codes are valid."""
        from .constants import AIRPORT_CODES
        for airport in v:
            if airport.upper() not in AIRPORT_CODES:
                raise ValueError(f"Invalid airport code: {airport}")
        return [code.upper() for code in v]
    
    @validator('rollout_percentage')
    def validate_percentage(cls, v, values):
        """Validate percentage based on strategy."""
        if values.get('rollout_strategy') == RolloutStrategy.PERCENTAGE and v <= 0:
            raise ValueError("Percentage must be > 0 for percentage rollout")
        return v

# ============================================================================
# FEATURE GROUPS
# ============================================================================

FEATURE_GROUPS: Dict[FeatureGroup, List[FeatureFlag]] = {
    FeatureGroup.CORE_INFRASTRUCTURE: [
        FeatureFlag.SYSTEM_MONITORING,
        FeatureFlag.REAL_TIME_LOGGING,
        FeatureFlag.PERFORMANCE_METRICS,
    ],
    FeatureGroup.PRICING_AI: [
        FeatureFlag.DYNAMIC_PRICING,
        FeatureFlag.DEMAND_FORECASTING,
        FeatureFlag.COMPETITOR_RESPONSE,
        FeatureFlag.PRICE_ELASTICITY_MODELS,
        FeatureFlag.PRICE_AB_TESTING,
        FeatureFlag.SAFETY_OVERRIDES,
    ],
    FeatureGroup.MARKETING_AI: [
        FeatureFlag.MARKETING_BUDGET_AUTOMATION,
        FeatureFlag.ROAS_PREDICTION,
        FeatureFlag.MULTI_TOUCH_ATTRIBUTION,
        FeatureFlag.GOOGLE_ADS_AUTOMATION,
        FeatureFlag.META_ADS_AUTOMATION,
        FeatureFlag.EMAIL_MARKETING_AUTOMATION,
    ],
    FeatureGroup.COMPETITOR_INTEL: [
        FeatureFlag.COMPETITOR_PRICE_MONITORING,
        FeatureFlag.AUTOMATED_SCRAPING,
        FeatureFlag.PRICE_SHOCK_DETECTION,
        FeatureFlag.PROMOTION_TRACKING,
    ],
    FeatureGroup.FRANCHISEE_AI: [
        FeatureFlag.FRANCHISEE_PERFORMANCE_SCORING,
        FeatureFlag.DRIVER_RISK_PREDICTION,
        FeatureFlag.FRAUD_DETECTION,
        FeatureFlag.REAL_TIME_ALERTS,
        FeatureFlag.AUTOMATED_INTERVENTIONS,
    ],
    FeatureGroup.WEBSITE_AI: [
        FeatureFlag.PRICE_SENSITIVITY_DETECTION,
        FeatureFlag.ABANDONMENT_PREDICTION,
        FeatureFlag.DYNAMIC_UX_PERSONALIZATION,
        FeatureFlag.REAL_TIME_FUNNEL_OPTIMIZATION,
        FeatureFlag.CTA_OPTIMIZATION,
    ],
    FeatureGroup.CALL_CENTER_AI: [
        FeatureFlag.VOICE_AI_PROCESSING,
        FeatureFlag.AUTOMATED_CALL_HANDLING,
        FeatureFlag.LATE_FEE_AUTOMATION,
        FeatureFlag.BOOKING_MODIFICATION_AI,
        FeatureFlag.REFUND_PROCESSING_AI,
    ],
    FeatureGroup.PHP_INTEGRATION: [
        FeatureFlag.PHP_AI_INTEGRATION,
        FeatureFlag.REAL_TIME_PRICE_UPDATES,
        FeatureFlag.CACHING_LAYER,
        FeatureFlag.FALLBACK_MECHANISMS,
    ],
    FeatureGroup.SAFETY_FEATURES: [
        FeatureFlag.SAFETY_OVERRIDES,
        FeatureFlag.FALLBACK_MECHANISMS,
    ],
}

# Feature dependencies (some features require others)
FEATURE_DEPENDENCIES: Dict[FeatureFlag, List[FeatureFlag]] = {
    FeatureFlag.DYNAMIC_PRICING: [
        FeatureFlag.DEMAND_FORECASTING,
        FeatureFlag.PRICE_ELASTICITY_MODELS,
    ],
    FeatureFlag.MARKETING_BUDGET_AUTOMATION: [
        FeatureFlag.ROAS_PREDICTION,
        FeatureFlag.MULTI_TOUCH_ATTRIBUTION,
    ],
    FeatureFlag.FRAUD_DETECTION: [
        FeatureFlag.REAL_TIME_ALERTS,
    ],
    FeatureFlag.AUTOMATED_CALL_HANDLING: [
        FeatureFlag.VOICE_AI_PROCESSING,
    ],
}

# ============================================================================
# FEATURE FLAG MANAGER
# ============================================================================

class FeatureFlagManager:
    """
    Manages feature flags with support for gradual rollouts, A/B testing,
    and emergency overrides.
    
    Reference: [Required Experience: Production engineering role]
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("feature_flags")
        self.features: Dict[FeatureFlag, FeatureConfiguration] = {}
        self.redis_client = None
        self.config_path = config_path
        
        # Initialize with default configuration
        self._initialize_defaults()
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Try to connect to Redis for distributed feature flag management
        self._init_redis()
        
        self.logger.info(
            f"Feature Flag Manager initialized with {len(self.features)} features",
            extra={"feature_count": len(self.features)}
        )
    
    def _initialize_defaults(self) -> None:
        """Initialize with default feature configurations."""
        
        # Core infrastructure - always enabled in production
        self.features[FeatureFlag.SYSTEM_MONITORING] = FeatureConfiguration(
            feature=FeatureFlag.SYSTEM_MONITORING,
            enabled=True,
            rollout_strategy=RolloutStrategy.ENABLED,
            metadata={"critical": True, "requires_restart": False}
        )
        
        # Pricing AI features - start disabled for safe rollout
        # Reference: [Core Responsibilities: 1. Commercial & Pricing AI]
        self.features[FeatureFlag.DYNAMIC_PRICING] = FeatureConfiguration(
            feature=FeatureFlag.DYNAMIC_PRICING,
            enabled=False,  # Start disabled
            rollout_strategy=RolloutStrategy.CANARY,
            airports={"JFK", "LAX"},  # Start with major airports
            metadata={
                "business_impact": "high",
                "risk_level": "high",
                "owner": "pricing_ai_team",
                "description": "Dynamic pricing algorithm that adjusts prices based on demand and competition"
            }
        )
        
        # Safety overrides - always enabled
        # Reference: [Absolutely Essential: Design with safety, overrides]
        self.features[FeatureFlag.SAFETY_OVERRIDES] = FeatureConfiguration(
            feature=FeatureFlag.SAFETY_OVERRIDES,
            enabled=True,
            rollout_strategy=RolloutStrategy.ENABLED,
            metadata={"critical": True, "emergency_override": True}
        )
        
        # Marketing AI features
        # Reference: [Core Responsibilities: 2. Marketing & Multi-Channel Budget Optimisation]
        self.features[FeatureFlag.MARKETING_BUDGET_AUTOMATION] = FeatureConfiguration(
            feature=FeatureFlag.MARKETING_BUDGET_AUTOMATION,
            enabled=False,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=0.1,  # 10% rollout initially
            metadata={
                "business_impact": "medium",
                "risk_level": "medium",
                "owner": "marketing_ai_team",
                "channels": ["google_ads", "meta_ads"]
            }
        )
        
        # Competitor intelligence
        # Reference: [Core Responsibilities: 3. Competitor & Market Intelligence]
        self.features[FeatureFlag.COMPETITOR_PRICE_MONITORING] = FeatureConfiguration(
            feature=FeatureFlag.COMPETITOR_PRICE_MONITORING,
            enabled=True,
            rollout_strategy=RolloutStrategy.ENABLED,
            metadata={
                "monitoring_interval": "15min",
                "alert_threshold": 0.10,  # 10% price change
            }
        )
        
        # Franchisee AI features
        # Reference: [Core Responsibilities: 4. Franchisee Performance, Risk & Fraud Detection]
        self.features[FeatureFlag.FRAUD_DETECTION] = FeatureConfiguration(
            feature=FeatureFlag.FRAUD_DETECTION,
            enabled=True,
            rollout_strategy=RolloutStrategy.ENABLED,
            metadata={
                "confidence_threshold": 0.85,
                "alert_channels": ["email", "slack"],
            }
        )
        
        # Website AI features
        # Reference: [Core Responsibilities: 5. Website Behaviour, Conversion & Personalisation AI]
        self.features[FeatureFlag.PRICE_SENSITIVITY_DETECTION] = FeatureConfiguration(
            feature=FeatureFlag.PRICE_SENSITIVITY_DETECTION,
            enabled=False,
            rollout_strategy=RolloutStrategy.AB_TEST,
            rollout_percentage=0.5,
            metadata={
                "test_name": "price_sensitivity_v1",
                "sample_size": 10000,
            }
        )
        
        # Call Center AI features
        # Reference: [Core Responsibilities: 6. AI Call Centre (DRT & CRS Automation)]
        self.features[FeatureFlag.AUTOMATED_CALL_HANDLING] = FeatureConfiguration(
            feature=FeatureFlag.AUTOMATED_CALL_HANDLING,
            enabled=False,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=0.3,  # 30% of calls
            metadata={
                "target_automation_rate": 0.7,  # 70% target
                "call_types": ["arrival", "return"],
            }
        )
        
        # PHP Integration
        # Reference: [Technical Stack: PHP (embedded AI integration)]
        self.features[FeatureFlag.PHP_AI_INTEGRATION] = FeatureConfiguration(
            feature=FeatureFlag.PHP_AI_INTEGRATION,
            enabled=True,
            rollout_strategy=RolloutStrategy.ENABLED,
            metadata={"api_version": "v1", "cache_ttl": 300}
        )
        
        # Initialize all other features as disabled
        for feature in FeatureFlag:
            if feature not in self.features:
                self.features[feature] = FeatureConfiguration(
                    feature=feature,
                    enabled=False,
                    rollout_strategy=RolloutStrategy.DISABLED
                )
    
    def _init_redis(self) -> None:
        """Initialize Redis connection for distributed feature flag management."""
        try:
            from ..settings import get_settings
            settings = get_settings()
            
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()  # Test connection
            self.logger.info("Redis connection established for feature flags")
        except Exception as e:
            self.logger.warning(f"Redis connection failed, using in-memory only: {e}")
            self.redis_client = None
    
    def is_enabled(self, feature: FeatureFlag, identifier: str = None) -> bool:
        """
        Check if a feature is enabled for a specific identifier.
        
        Args:
            feature: Feature flag to check
            identifier: Optional identifier (airport code, user ID, etc.)
            
        Returns:
            bool: True if feature is enabled
            
        Reference: [Required Experience: Ability to design safe AI automation]
        """
        if feature not in self.features:
            self.logger.warning(f"Unknown feature flag: {feature}")
            return False
        
        config = self.features[feature]
        
        # Check Redis cache first if available
        if self.redis_client:
            try:
                cache_key = f"feature_flag:{feature}:{identifier or 'global'}"
                cached = self.redis_client.get(cache_key)
                if cached is not None:
                    return cached == "1"
            except Exception as e:
                self.logger.error(f"Redis cache error: {e}")
        
        # Check dependencies first
        for dependency in FEATURE_DEPENDENCIES.get(feature, []):
            if not self.is_enabled(dependency, identifier):
                self.logger.debug(
                    f"Feature {feature} disabled due to missing dependency: {dependency}",
                    extra={"feature": feature, "dependency": dependency}
                )
                return False
        
        # Check if feature is active
        is_active = config.is_active_for(identifier)
        
        # Cache result in Redis
        if self.redis_client:
            try:
                cache_key = f"feature_flag:{feature}:{identifier or 'global'}"
                self.redis_client.setex(
                    cache_key,
                    60,  # 1 minute TTL
                    "1" if is_active else "0"
                )
            except Exception as e:
                self.logger.error(f"Redis cache set error: {e}")
        
        # Log feature checks for high-risk features
        if config.metadata.get("risk_level") == "high" and is_active:
            self.logger.info(
                f"High-risk feature {feature} checked for {identifier}",
                extra={
                    "feature": feature,
                    "identifier": identifier,
                    "enabled": is_active,
                    "risk_level": "high"
                }
            )
        
        return is_active
    
    def enable_feature(self, feature: FeatureFlag, **kwargs) -> bool:
        """
        Enable a feature flag with optional configuration.
        
        Args:
            feature: Feature flag to enable
            **kwargs: Additional configuration options
            
        Returns:
            bool: True if successful
            
        Reference: [Absolutely Essential: Design with overrides]
        """
        if feature not in self.features:
            self.logger.error(f"Cannot enable unknown feature: {feature}")
            return False
        
        config = self.features[feature]
        config.enabled = True
        
        # Update configuration based on kwargs
        if 'rollout_strategy' in kwargs:
            config.rollout_strategy = kwargs['rollout_strategy']
        if 'rollout_percentage' in kwargs:
            config.rollout_percentage = kwargs['rollout_percentage']
        if 'airports' in kwargs:
            config.airports = set(kwargs['airports'])
        if 'whitelist' in kwargs:
            config.whitelist = set(kwargs['whitelist'])
        
        # Log the change
        self.logger.info(
            f"Feature {feature} enabled",
            extra={
                "feature": feature,
                "rollout_strategy": config.rollout_strategy,
                "rollout_percentage": config.rollout_percentage,
                "airports": list(config.airports),
                "action": "enable"
            }
        )
        
        # Clear Redis cache for this feature
        self._clear_feature_cache(feature)
        
        return True
    
    def disable_feature(self, feature: FeatureFlag, reason: str = "") -> bool:
        """
        Disable a feature flag.
        
        Args:
            feature: Feature flag to disable
            reason: Reason for disabling
            
        Returns:
            bool: True if successful
        """
        if feature not in self.features:
            self.logger.error(f"Cannot disable unknown feature: {feature}")
            return False
        
        config = self.features[feature]
        was_enabled = config.enabled
        config.enabled = False
        
        # Log the change
        self.logger.warning(
            f"Feature {feature} disabled",
            extra={
                "feature": feature,
                "reason": reason,
                "previous_state": "enabled" if was_enabled else "disabled",
                "action": "disable"
            }
        )
        
        # Clear Redis cache for this feature
        self._clear_feature_cache(feature)
        
        # If this is a high-risk feature, send alert
        if config.metadata.get("risk_level") == "high":
            self._send_feature_disabled_alert(feature, reason)
        
        return True
    
    def set_feature_override(self, feature: FeatureFlag, airport: str, enabled: bool) -> bool:
        """
        Set an airport-specific override for a feature.
        
        Args:
            feature: Feature flag
            airport: Airport code
            enabled: Whether feature should be enabled for this airport
            
        Returns:
            bool: True if successful
        """
        from .constants import validate_airport_code
        if not validate_airport_code(airport):
            self.logger.error(f"Invalid airport code: {airport}")
            return False
        
        if feature not in self.features:
            self.logger.error(f"Unknown feature: {feature}")
            return False
        
        airport = airport.upper()
        config = self.features[feature]
        
        if enabled:
            config.airports.add(airport)
            if airport in config.whitelist:
                config.whitelist.remove(airport)
        else:
            config.airports.discard(airport)
            config.whitelist.discard(airport)
        
        self.logger.info(
            f"Feature override set: {feature} for {airport} = {enabled}",
            extra={
                "feature": feature,
                "airport": airport,
                "enabled": enabled,
                "action": "override"
            }
        )
        
        # Clear cache for this feature and airport
        self._clear_feature_cache(feature, airport)
        
        return True
    
    def _clear_feature_cache(self, feature: FeatureFlag, identifier: str = None) -> None:
        """Clear Redis cache for a feature."""
        if self.redis_client:
            try:
                if identifier:
                    # Clear specific identifier cache
                    cache_key = f"feature_flag:{feature}:{identifier}"
                    self.redis_client.delete(cache_key)
                else:
                    # Clear all caches for this feature
                    pattern = f"feature_flag:{feature}:*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.error(f"Error clearing feature cache: {e}")
    
    def _send_feature_disabled_alert(self, feature: FeatureFlag, reason: str) -> None:
        """Send alert when high-risk feature is disabled."""
        # In production, this would integrate with alerting system (PagerDuty, Slack, etc.)
        alert_message = (
            f"🚨 HIGH-RISK FEATURE DISABLED\n"
            f"Feature: {feature.value}\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().isoformat()}\n"
            f"Action Required: Investigate immediately"
        )
        
        self.logger.critical(alert_message, extra={"alert": True, "feature": feature})
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load feature flag configuration from YAML or JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            bool: True if successful
            
        Reference: [Required Experience: Production engineering role]
        """
        try:
            path = Path(filepath)
            if not path.exists():
                self.logger.error(f"Configuration file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Validate and update features
            for feature_data in data.get('features', []):
                try:
                    model = FeatureFlagModel(**feature_data)
                    config = FeatureConfiguration(
                        feature=model.feature,
                        enabled=model.enabled,
                        rollout_strategy=model.rollout_strategy,
                        rollout_percentage=model.rollout_percentage,
                        whitelist=set(model.whitelist),
                        airports=set(model.airports),
                        start_date=model.start_date,
                        end_date=model.end_date,
                        metadata=model.metadata
                    )
                    self.features[model.feature] = config
                except Exception as e:
                    self.logger.error(f"Error loading feature {feature_data.get('feature')}: {e}")
                    continue
            
            self.logger.info(
                f"Loaded feature flags from {filepath}",
                extra={"file": filepath, "features_loaded": len(data.get('features', []))}
            )
            
            # Clear all caches since configuration changed
            if self.redis_client:
                self._clear_all_caches()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading feature flags from file: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save current feature flag configuration to file.
        
        Args:
            filepath: Path to output file
            
        Returns:
            bool: True if successful
        """
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "features": []
            }
            
            for feature, config in self.features.items():
                feature_data = {
                    "feature": feature.value,
                    "enabled": config.enabled,
                    "rollout_strategy": config.rollout_strategy.value,
                    "rollout_percentage": config.rollout_percentage,
                    "whitelist": list(config.whitelist),
                    "airports": list(config.airports),
                    "start_date": config.start_date.isoformat() if config.start_date else None,
                    "end_date": config.end_date.isoformat() if config.end_date else None,
                    "metadata": config.metadata
                }
                export_data["features"].append(feature_data)
            
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(export_data, f, default_flow_style=False)
                else:
                    json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved feature flags to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving feature flags: {e}")
            return False
    
    def _clear_all_caches(self) -> None:
        """Clear all feature flag caches from Redis."""
        if self.redis_client:
            try:
                pattern = "feature_flag:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.error(f"Error clearing all caches: {e}")
    
    def get_feature_status(self, feature: FeatureFlag) -> Dict[str, Any]:
        """
        Get detailed status of a feature flag.
        
        Args:
            feature: Feature flag
            
        Returns:
            Dict with feature status
        """
        if feature not in self.features:
            return {"error": "Feature not found"}
        
        config = self.features[feature]
        
        # Determine feature status
        if not config.enabled:
            status = FeatureStatus.INACTIVE
        elif config.rollout_strategy == RolloutStrategy.DISABLED:
            status = FeatureStatus.INACTIVE
        elif config.rollout_strategy == RolloutStrategy.ENABLED:
            status = FeatureStatus.ACTIVE
        elif config.rollout_strategy in [RolloutStrategy.PERCENTAGE, RolloutStrategy.CANARY]:
            status = FeatureStatus.ROLLING_OUT
        elif config.rollout_strategy == RolloutStrategy.AB_TEST:
            status = FeatureStatus.EXPERIMENTAL
        else:
            status = FeatureStatus.INACTIVE
        
        return {
            "feature": feature.value,
            "status": status.value,
            "enabled": config.enabled,
            "rollout_strategy": config.rollout_strategy.value,
            "rollout_percentage": config.rollout_percentage,
            "airports_enabled": list(config.airports),
            "whitelist_size": len(config.whitelist),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "dependencies": [dep.value for dep in FEATURE_DEPENDENCIES.get(feature, [])],
            "metadata": config.metadata
        }
    
    def get_all_features_status(self) -> Dict[str, Any]:
        """
        Get status of all features.
        
        Returns:
            Dict with all features status
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "summary": {
                "total": len(self.features),
                "enabled": 0,
                "disabled": 0,
                "rolling_out": 0,
                "experimental": 0
            }
        }
        
        for feature in self.features:
            status = self.get_feature_status(feature)
            result["features"][feature.value] = status
            
            # Update summary
            if status["enabled"]:
                result["summary"]["enabled"] += 1
                if status["status"] == FeatureStatus.ROLLING_OUT.value:
                    result["summary"]["rolling_out"] += 1
                elif status["status"] == FeatureStatus.EXPERIMENTAL.value:
                    result["summary"]["experimental"] += 1
            else:
                result["summary"]["disabled"] += 1
        
        return result
    
    def validate_feature_config(self) -> Dict[str, Any]:
        """
        Validate feature flag configuration for consistency and safety.
        
        Returns:
            Dict with validation results
            
        Reference: [Absolutely Essential: Design with safety, logging, overrides]
        """
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "high_risk_features": []
        }
        
        # Check each feature
        for feature, config in self.features.items():
            # Check for high-risk features without safety overrides
            if (config.metadata.get("risk_level") == "high" and 
                not self.is_enabled(FeatureFlag.SAFETY_OVERRIDES)):
                validation["valid"] = False
                validation["issues"].append(
                    f"High-risk feature {feature} enabled without safety overrides"
                )
                validation["high_risk_features"].append(feature.value)
            
            # Check dependencies
            for dependency in FEATURE_DEPENDENCIES.get(feature, []):
                if config.enabled and not self.is_enabled(dependency):
                    validation["valid"] = False
                    validation["issues"].append(
                        f"Feature {feature} enabled but dependency {dependency} is disabled"
                    )
            
            # Check rollout percentage for percentage strategy
            if (config.rollout_strategy == RolloutStrategy.PERCENTAGE and 
                config.rollout_percentage <= 0):
                validation["warnings"].append(
                    f"Feature {feature} has percentage rollout with 0%"
                )
        
        # Check for circular dependencies (simplified check)
        # In production, would implement full dependency graph validation
        
        self.logger.info(
            "Feature configuration validation completed",
            extra=validation
        )
        
        return validation

# ============================================================================
# GLOBAL FEATURE FLAG MANAGER INSTANCE
# ============================================================================

_feature_flag_manager: Optional[FeatureFlagManager] = None

def get_feature_flags() -> FeatureFlagManager:
    """
    Get the global feature flag manager instance.
    
    Returns:
        FeatureFlagManager: Global instance
        
    Reference: [Required Experience: Production engineering role]
    """
    global _feature_flag_manager
    
    if _feature_flag_manager is None:
        # Load from config file if exists
        config_path = Path("config/feature_flags.yaml")
        if config_path.exists():
            _feature_flag_manager = FeatureFlagManager(str(config_path))
        else:
            _feature_flag_manager = FeatureFlagManager()
    
    return _feature_flag_manager

def is_feature_enabled(feature: FeatureFlag, identifier: str = None) -> bool:
    """
    Convenience function to check if a feature is enabled.
    
    Args:
        feature: Feature flag
        identifier: Optional identifier
        
    Returns:
        bool: True if enabled
    """
    return get_feature_flags().is_enabled(feature, identifier)

def enable_feature(feature: FeatureFlag, **kwargs) -> bool:
    """Enable a feature flag."""
    return get_feature_flags().enable_feature(feature, **kwargs)

def disable_feature(feature: FeatureFlag, reason: str = "") -> bool:
    """Disable a feature flag."""
    return get_feature_flags().disable_feature(feature, reason)

def set_feature_override(feature: FeatureFlag, airport: str, enabled: bool) -> bool:
    """Set airport-specific feature override."""
    return get_feature_flags().set_feature_override(feature, airport, enabled)

def get_enabled_features(identifier: str = None) -> List[str]:
    """Get list of enabled features for an identifier."""
    manager = get_feature_flags()
    enabled = []
    for feature in FeatureFlag:
        if manager.is_enabled(feature, identifier):
            enabled.append(feature.value)
    return enabled

def get_disabled_features(identifier: str = None) -> List[str]:
    """Get list of disabled features for an identifier."""
    manager = get_feature_flags()
    disabled = []
    for feature in FeatureFlag:
        if not manager.is_enabled(feature, identifier):
            disabled.append(feature.value)
    return disabled

def validate_feature_config() -> Dict[str, Any]:
    """Validate feature configuration."""
    return get_feature_flags().validate_feature_config()

def get_feature_status(feature: FeatureFlag) -> Dict[str, Any]:
    """Get feature status."""
    return get_feature_flags().get_feature_status(feature)

def get_all_features_status() -> Dict[str, Any]:
    """Get status of all features."""
    return get_feature_flags().get_all_features_status()

# Initialize on module import
get_feature_flags()

# ============================================================================
# FEATURE FLAG DECORATORS
# ============================================================================

def feature_required(feature: FeatureFlag):
    """
    Decorator to require a feature flag for function execution.
    
    Usage:
        @feature_required(FeatureFlag.DYNAMIC_PRICING)
        def update_prices():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature):
                raise FeatureDisabledError(
                    f"Feature {feature.value} is required but disabled",
                    feature=feature
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def feature_fallback(feature: FeatureFlag, fallback_func):
    """
    Decorator to provide fallback when feature is disabled.
    
    Usage:
        @feature_fallback(FeatureFlag.DYNAMIC_PRICING, manual_pricing)
        def update_prices():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature):
                logging.getLogger("feature_flags").info(
                    f"Using fallback for {feature.value}",
                    extra={"feature": feature.value, "using_fallback": True}
                )
                return fallback_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# EXCEPTIONS
# ============================================================================

class FeatureFlagError(Exception):
    """Base exception for feature flag errors."""
    pass

class FeatureDisabledError(FeatureFlagError):
    """Raised when a required feature is disabled."""
    def __init__(self, message: str, feature: FeatureFlag):
        self.feature = feature
        super().__init__(f"{message} (feature: {feature.value})")

class FeatureDependencyError(FeatureFlagError):
    """Raised when feature dependencies are not met."""
    pass

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def initialize_feature_flags() -> FeatureFlagManager:
    """
    Initialize feature flags system.
    
    Returns:
        FeatureFlagManager: Initialized manager
        
    Reference: [Required Experience: Production engineering role]
    """
    manager = get_feature_flags()
    
    # Validate configuration
    validation = manager.validate_feature_config()
    
    if not validation["valid"]:
        error_msg = f"Feature flag validation failed: {validation['issues']}"
        manager.logger.critical(error_msg, extra={"validation": validation})
        raise FeatureFlagError(error_msg)
    
    # Log initialization
    summary = manager.get_all_features_status()["summary"]
    manager.logger.info(
        "Feature flags system initialized",
        extra={
            "total_features": summary["total"],
            "enabled_features": summary["enabled"],
            "rolling_out": summary["rolling_out"],
            "high_risk_count": len(validation["high_risk_features"])
        }
    )
    
    return manager

# Auto-initialize on module import
initialize_feature_flags()
