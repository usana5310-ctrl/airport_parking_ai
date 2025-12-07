"""
Business Constants Module
Project: AI Commercial Platform for Airport Parking

This module defines all business constants, airport codes, operational limits,
and fixed values used throughout the AI Commercial Platform.

Reference: [Core Responsibilities: 1. Commercial & Pricing AI - Airport-level models]
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from enum import Enum
from datetime import time, timedelta
import json
from pathlib import Path

# ============================================================================
# AIRPORT CONSTANTS
# ============================================================================

# IATA Airport Codes with metadata
AIRPORT_CODES: Dict[str, Dict[str, Any]] = {
    # Major US Airports
    "JFK": {
        "name": "John F. Kennedy International Airport",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "timezone": "America/New_York",
        "region": "Northeast",
        "capacity": 1500,  # Parking spots
        "peak_season": ["Jun", "Jul", "Aug", "Dec"],
        "base_demand_factor": 1.2,
        "min_operational_hours": 24,
    },
    "LAX": {
        "name": "Los Angeles International Airport",
        "city": "Los Angeles",
        "state": "CA",
        "country": "USA",
        "timezone": "America/Los_Angeles",
        "region": "West",
        "capacity": 2000,
        "peak_season": ["Mar", "Jul", "Aug", "Dec"],
        "base_demand_factor": 1.3,
        "min_operational_hours": 24,
    },
    "ORD": {
        "name": "O'Hare International Airport",
        "city": "Chicago",
        "state": "IL",
        "country": "USA",
        "timezone": "America/Chicago",
        "region": "Midwest",
        "capacity": 1800,
        "peak_season": ["Jun", "Jul", "Aug", "Dec"],
        "base_demand_factor": 1.1,
        "min_operational_hours": 24,
    },
    "DFW": {
        "name": "Dallas/Fort Worth International Airport",
        "city": "Dallas",
        "state": "TX",
        "country": "USA",
        "timezone": "America/Chicago",
        "region": "South",
        "capacity": 1600,
        "peak_season": ["Mar", "Jul", "Nov", "Dec"],
        "base_demand_factor": 1.0,
        "min_operational_hours": 24,
    },
    "MIA": {
        "name": "Miami International Airport",
        "city": "Miami",
        "state": "FL",
        "country": "USA",
        "timezone": "America/New_York",
        "region": "Southeast",
        "capacity": 1200,
        "peak_season": ["Jan", "Feb", "Mar", "Dec"],  # Winter season
        "base_demand_factor": 1.4,
        "min_operational_hours": 24,
    },
    "SFO": {
        "name": "San Francisco International Airport",
        "city": "San Francisco",
        "state": "CA",
        "country": "USA",
        "timezone": "America/Los_Angeles",
        "region": "West",
        "capacity": 1400,
        "peak_season": ["Jun", "Jul", "Aug", "Sep"],
        "base_demand_factor": 1.2,
        "min_operational_hours": 24,
    },
    "SEA": {
        "name": "Seattle-Tacoma International Airport",
        "city": "Seattle",
        "state": "WA",
        "country": "USA",
        "timezone": "America/Los_Angeles",
        "region": "West",
        "capacity": 1100,
        "peak_season": ["Jun", "Jul", "Aug", "Dec"],
        "base_demand_factor": 1.1,
        "min_operational_hours": 24,
    },
    "ATL": {
        "name": "Hartsfield-Jackson Atlanta International Airport",
        "city": "Atlanta",
        "state": "GA",
        "country": "USA",
        "timezone": "America/New_York",
        "region": "Southeast",
        "capacity": 2200,
        "peak_season": ["Mar", "Jun", "Jul", "Nov"],
        "base_demand_factor": 1.5,  # World's busiest airport
        "min_operational_hours": 24,
    },
    # Add more airports as needed
}

# Airport groupings by region
AIRPORT_REGIONS: Dict[str, List[str]] = {
    "Northeast": ["JFK", "BOS", "EWR", "PHL", "BWI"],
    "Midwest": ["ORD", "MDW", "DTW", "MSP", "STL"],
    "South": ["DFW", "IAH", "HOU", "AUS", "SAT"],
    "Southeast": ["MIA", "ATL", "MCO", "TPA", "FLL"],
    "West": ["LAX", "SFO", "SEA", "SAN", "LAS", "PHX"],
}

# Airport timezone mappings (simplified from full data)
AIRPORT_TIMEZONES: Dict[str, str] = {
    airport: data["timezone"]
    for airport, data in AIRPORT_CODES.items()
}

# Airport names (simplified mapping)
AIRPORT_NAMES: Dict[str, str] = {
    airport: data["name"]
    for airport, data in AIRPORT_CODES.items()
}

# ============================================================================
# PRICING & REVENUE CONSTANTS
# ============================================================================

# Minimum price floors per airport (break-even costs)
# Reference: [Core Responsibilities: Seasonal break-even floors]
MINIMUM_PRICE_FLOORS: Dict[str, float] = {
    "JFK": 25.00,   # High operational costs in NYC
    "LAX": 22.00,   # LA operational costs
    "ORD": 20.00,   # Chicago
    "DFW": 18.00,   # Dallas
    "MIA": 21.00,   # Miami
    "SFO": 26.00,   # San Francisco (highest due to costs)
    "SEA": 20.00,   # Seattle
    "ATL": 19.00,   # Atlanta (efficient operations)
}

# Maximum price caps (to avoid customer complaints)
MAXIMUM_PRICE_CAPS: Dict[str, float] = {
    "JFK": 75.00,
    "LAX": 70.00,
    "ORD": 65.00,
    "DFW": 60.00,
    "MIA": 68.00,
    "SFO": 80.00,
    "SEA": 65.00,
    "ATL": 62.00,
}

# CPC-based price protection minimums
# Reference: [Core Responsibilities: CPC-based price protection]
DEFAULT_CPC_LIMITS: Dict[str, float] = {
    "JFK": 8.00,    # High competition, high CPC
    "LAX": 7.50,
    "ORD": 6.50,
    "DFW": 5.50,
    "MIA": 7.00,
    "SFO": 9.00,    # Highest CPC due to tech competition
    "SEA": 6.00,
    "ATL": 5.00,    # Lower CPC due to volume
}

# Price elasticity coefficients by airport (-1.5 means 1% price increase = 1.5% demand decrease)
# Reference: [Core Responsibilities: Design price elasticity models per airport & service]
PRICE_ELASTICITY: Dict[str, float] = {
    "JFK": -1.2,    # Business travelers less price sensitive
    "LAX": -1.5,    # Mix of business and leisure
    "ORD": -1.3,    # Hub airport, moderate sensitivity
    "DFW": -1.6,    # Price sensitive market
    "MIA": -1.4,    # Leisure destination
    "SFO": -1.1,    # Business/tech, least sensitive
    "SEA": -1.3,
    "ATL": -1.7,    # Very price sensitive due to competition
}

# Service type multipliers
SERVICE_TYPE_MULTIPLIERS: Dict[str, float] = {
    "economy": 1.0,
    "covered": 1.3,
    "valet": 1.8,
    "premium": 2.2,
    "ev_charging": 1.4,
}

# Duration-based discounts (percentage discount for longer stays)
DURATION_DISCOUNTS: Dict[int, float] = {
    1: 0.0,     # 0% discount for 1 day
    3: 0.05,    # 5% discount for 3+ days
    7: 0.10,    # 10% discount for 7+ days
    14: 0.15,   # 15% discount for 14+ days
    30: 0.20,   # 20% discount for 30+ days
}

# ============================================================================
# TIME & OPERATIONAL CONSTANTS
# ============================================================================

# Default booking window in days
DEFAULT_BOOKING_WINDOW_DAYS: int = 365  # Can book up to 1 year in advance

# Minimum and maximum stay durations
MINIMUM_STAY_HOURS: int = 2    # Minimum 2-hour parking
MAXIMUM_STAY_DAYS: int = 90    # Maximum 90-day parking

# Shuttle schedule intervals (minutes)
SHUTTLE_INTERVAL_MINUTES: int = 10
SHUTTLE_OPERATING_HOURS: Tuple[time, time] = (time(4, 0), time(1, 0))  # 4 AM to 1 AM next day

# Check-in/Check-out buffer times (minutes)
CHECK_IN_BUFFER_MINUTES: int = 30
CHECK_OUT_BUFFER_MINUTES: int = 60
GRACE_PERIOD_MINUTES: int = 15

# ============================================================================
# MARKETING & ADVERTISING CONSTANTS
# ============================================================================

# Reference: [Core Responsibilities: 2. Marketing & Multi-Channel Budget Optimisation]

# Marketing channel identifiers
MARKETING_CHANNELS: Dict[str, str] = {
    "google_ads": "Google Ads",
    "meta_ads": "Meta (Facebook/Instagram)",
    "bing_ads": "Bing & Yahoo Ads",
    "email": "Email (Mailchimp)",
    "sms": "SMS",
    "affiliates": "Affiliates",
    "direct": "Direct Traffic",
    "organic": "Organic Search",
}

# Default daily budget allocation by channel (percentages)
DEFAULT_CHANNEL_BUDGET_ALLOCATION: Dict[str, float] = {
    "google_ads": 0.40,    # 40%
    "meta_ads": 0.25,      # 25%
    "bing_ads": 0.10,      # 10%
    "email": 0.15,         # 15%
    "sms": 0.05,           # 5%
    "affiliates": 0.05,    # 5%
}

# Target ROAS (Return on Ad Spend) by channel
TARGET_ROAS_BY_CHANNEL: Dict[str, float] = {
    "google_ads": 8.0,     # 8x return
    "meta_ads": 6.0,       # 6x return
    "bing_ads": 5.0,       # 5x return
    "email": 25.0,         # 25x return (high ROI for email)
    "sms": 15.0,           # 15x return
    "affiliates": 10.0,    # 10x return
}

# Attribution window days (how long to credit a marketing touch)
ATTRIBUTION_WINDOW_DAYS: Dict[str, int] = {
    "direct": 1,           # 1 day for direct
    "google_ads": 30,      # 30 days for Google Ads
    "meta_ads": 28,        # 28 days for Meta
    "email": 7,            # 7 days for email
    "sms": 3,              # 3 days for SMS
}

# ============================================================================
# CURRENCY & FINANCIAL CONSTANTS
# ============================================================================

CURRENCY_CODE: str = "USD"
CURRENCY_SYMBOL: str = "$"
DECIMAL_PLACES: int = 2

# Tax rates by state (percentage)
TAX_RATES_BY_STATE: Dict[str, float] = {
    "NY": 8.875,    # New York
    "CA": 7.25,     # California
    "IL": 6.25,     # Illinois
    "TX": 6.25,     # Texas
    "FL": 6.00,     # Florida
    "WA": 6.50,     # Washington
    "GA": 4.00,     # Georgia
}

# Commission rates for franchisees (percentage of revenue)
FRANCHISEE_COMMISSION_RATES: Dict[str, float] = {
    "standard": 0.70,   # 70% to franchisee
    "premium": 0.65,    # 65% to franchisee (premium locations)
    "new": 0.75,        # 75% to franchisee (first 6 months)
}

# Late fee structure (per hour after grace period)
LATE_FEE_STRUCTURE: List[Tuple[int, float]] = [
    (1, 10.00),     # First hour: $10
    (3, 25.00),     # 1-3 hours: $25 total
    (6, 50.00),     # 3-6 hours: $50 total
    (24, 100.00),   # 6-24 hours: $100 total
    (999, 150.00),  # 24+ hours: $150 (max)
]

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Price validation rules
PRICE_VALIDATION_RULES: Dict[str, Any] = {
    "min_price": 0.01,
    "max_price": 9999.99,
    "price_increment": 0.01,
    "max_percentage_change_daily": 0.50,  # Max 50% change in one day
    "min_percentage_change": 0.01,        # Min 1% change to trigger update
}

# Date validation rules
DATE_VALIDATION_RULES: Dict[str, Any] = {
    "min_booking_advance_hours": 1,       # Can book minimum 1 hour in advance
    "max_booking_advance_days": 365,      # Can book max 1 year in advance
    "min_checkin_advance_minutes": 30,    # Must check in at least 30 min before
    "max_checkout_extension_hours": 2,    # Can extend checkout by max 2 hours
}

# Input validation regex patterns
VALIDATION_PATTERNS: Dict[str, str] = {
    "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "phone": r'^\+?1?\d{9,15}$',
    "license_plate": r'^[A-Z0-9]{1,10}$',
    "booking_reference": r'^[A-Z0-9]{6,12}$',
    "airport_code": r'^[A-Z]{3}$',
}

# ============================================================================
# AI MODEL CONSTANTS
# ============================================================================

# Reference: [Technical Stack: AI / ML: Python, forecasting models, optimisation systems]

# Demand forecasting model parameters
DEMAND_FORECASTING_PARAMS: Dict[str, Any] = {
    "forecast_horizon_days": 30,
    "seasonality_periods": [7, 30, 365],  # Weekly, monthly, yearly seasonality
    "confidence_interval": 0.95,
    "min_training_samples": 30,
}

# Machine learning model versions and thresholds
MODEL_THRESHOLDS: Dict[str, float] = {
    "demand_forecast_confidence": 0.70,
    "price_elasticity_confidence": 0.80,
    "fraud_detection_confidence": 0.85,
    "abandonment_prediction_confidence": 0.75,
    "roas_prediction_confidence": 0.65,
}

# Batch processing sizes
BATCH_SIZES: Dict[str, int] = {
    "pricing_updates": 1000,
    "marketing_optimization": 500,
    "competitor_scraping": 100,
    "fraud_detection": 200,
    "call_center_processing": 50,
}

# ============================================================================
# FRANCHISEE & OPERATIONAL CONSTANTS
# ============================================================================

# Reference: [Core Responsibilities: 4. Franchisee Performance, Risk & Fraud Detection]

# Franchisee Performance Scoring (FPS) weights
FPS_WEIGHTS: Dict[str, float] = {
    "customer_rating": 0.25,
    "on_time_performance": 0.20,
    "vehicle_cleanliness": 0.15,
    "sla_compliance": 0.20,
    "revenue_growth": 0.10,
    "complaint_rate": -0.10,  # Negative weight for complaints
}

# SLA (Service Level Agreement) thresholds
SLA_THRESHOLDS: Dict[str, float] = {
    "shuttle_wait_time_minutes": 15,
    "checkin_time_minutes": 5,
    "checkout_time_minutes": 10,
    "vehicle_ready_time_minutes": 20,
    "customer_response_hours": 2,
}

# Fraud detection thresholds
FRAUD_THRESHOLDS: Dict[str, float] = {
    "refund_rate": 0.10,      # >10% refund rate triggers alert
    "no_show_rate": 0.15,     # >15% no-show rate
    "late_cancel_rate": 0.20, # >20% late cancellation
    "discount_abuse": 0.25,   # >25% bookings with discounts
}

# ============================================================================
# WEBSITE & CONVERSION CONSTANTS
# ============================================================================

# Reference: [Core Responsibilities: 5. Website Behaviour, Conversion & Personalisation AI]

# User behavior thresholds
USER_BEHAVIOR_THRESHOLDS: Dict[str, int] = {
    "abandonment_time_seconds": 300,      # 5 minutes of inactivity
    "price_view_threshold": 3,            # Viewed 3+ prices
    "session_duration_threshold": 180,    # 3+ minute session
    "page_view_threshold": 5,             # 5+ page views
}

# Personalization variants
PERSONALIZATION_VARIANTS: Dict[str, List[str]] = {
    "cta_button_text": [
        "Book Now & Save",
        "Reserve Your Spot",
        "Secure Parking",
        "Get Instant Quote",
        "Check Availability",
    ],
    "color_scheme": ["blue", "green", "orange", "purple"],
    "layout_type": ["compact", "detailed", "image_focused", "price_focused"],
}

# A/B test configuration
AB_TEST_CONFIG: Dict[str, Any] = {
    "min_sample_size": 1000,
    "test_duration_days": 7,
    "confidence_level": 0.95,
    "min_detection_effect": 0.05,  # Minimum 5% improvement to be significant
}

# ============================================================================
# CALL CENTER AUTOMATION CONSTANTS
# ============================================================================

# Reference: [Core Responsibilities: 6. AI Call Centre (DRT & CRS Automation)]

# Call types and handling priorities
CALL_TYPES: Dict[str, Dict[str, Any]] = {
    "arrival": {
        "priority": 1,
        "avg_handle_time": 180,  # 3 minutes
        "automation_rate": 0.85,  # 85% target automation
    },
    "return": {
        "priority": 1,
        "avg_handle_time": 150,  # 2.5 minutes
        "automation_rate": 0.80,
    },
    "modification": {
        "priority": 2,
        "avg_handle_time": 240,  # 4 minutes
        "automation_rate": 0.70,
    },
    "refund": {
        "priority": 3,
        "avg_handle_time": 300,  # 5 minutes
        "automation_rate": 0.60,
    },
    "complaint": {
        "priority": 3,
        "avg_handle_time": 420,  # 7 minutes
        "automation_rate": 0.50,
    },
}

# Payment method codes
PAYMENT_METHODS: Dict[str, str] = {
    "VISA": "visa",
    "MASTERCARD": "mastercard",
    "AMEX": "amex",
    "DISCOVER": "discover",
    "PAYPAL": "paypal",
    "APPLE_PAY": "apple_pay",
    "GOOGLE_PAY": "google_pay",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_airport_info(airport_code: str) -> Dict[str, Any]:
    """
    Get complete information for a specific airport.
    
    Args:
        airport_code: IATA airport code (e.g., "JFK")
        
    Returns:
        Dict with airport information
        
    Raises:
        ValueError: If airport code not found
        
    Reference: [Core Responsibilities: Airport-level models]
    """
    airport_code = airport_code.upper()
    if airport_code not in AIRPORT_CODES:
        raise ValueError(f"Airport code '{airport_code}' not found in configuration")
    
    return AIRPORT_CODES[airport_code].copy()

def validate_airport_code(airport_code: str) -> bool:
    """
    Validate if an airport code is supported by the system.
    
    Args:
        airport_code: IATA airport code
        
    Returns:
        bool: True if valid, False otherwise
    """
    return airport_code.upper() in AIRPORT_CODES

def get_business_day_range(start_date, end_date):
    """
    Calculate number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Number of business days
    """
    # Implementation would go here
    # For now, return placeholder
    return (end_date - start_date).days

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convert amount between currencies.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code
        
    Returns:
        float: Converted amount
        
    Note: For now only supports USD, would integrate with forex API in production
    """
    if from_currency != "USD" or to_currency != "USD":
        # In production, integrate with currency conversion API
        raise NotImplementedError("Currency conversion only supports USD in this version")
    
    return amount

def get_minimum_price(airport_code: str, service_type: str = "economy") -> float:
    """
    Calculate minimum price for an airport and service type.
    
    Args:
        airport_code: IATA airport code
        service_type: Type of parking service
        
    Returns:
        float: Minimum price
        
    Reference: [Core Responsibilities: Seasonal break-even floors]
    """
    if not validate_airport_code(airport_code):
        raise ValueError(f"Invalid airport code: {airport_code}")
    
    airport_code = airport_code.upper()
    base_price = MINIMUM_PRICE_FLOORS.get(airport_code, 20.00)
    multiplier = SERVICE_TYPE_MULTIPLIERS.get(service_type.lower(), 1.0)
    
    return round(base_price * multiplier, 2)

def get_price_elasticity(airport_code: str) -> float:
    """
    Get price elasticity coefficient for an airport.
    
    Args:
        airport_code: IATA airport code
        
    Returns:
        float: Price elasticity coefficient
        
    Reference: [Core Responsibilities: Design price elasticity models per airport & service]
    """
    if not validate_airport_code(airport_code):
        raise ValueError(f"Invalid airport code: {airport_code}")
    
    return PRICE_ELASTICITY.get(airport_code.upper(), -1.5)

def get_tax_rate(state_code: str) -> float:
    """
    Get tax rate for a state.
    
    Args:
        state_code: Two-letter state code
        
    Returns:
        float: Tax rate as percentage
    """
    return TAX_RATES_BY_STATE.get(state_code.upper(), 0.0)

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def validate_constants() -> Dict[str, bool]:
    """
    Validate that all constants are properly configured.
    
    Returns:
        Dict with validation results
        
    Reference: [Absolutely Essential: Design with safety, logging, overrides]
    """
    validation_results = {}
    
    try:
        # Validate airport constants
        validation_results["airports_configured"] = len(AIRPORT_CODES) > 0
        validation_results["airport_price_floors"] = all(
            airport in MINIMUM_PRICE_FLOORS 
            for airport in AIRPORT_CODES.keys()
        )
        validation_results["airport_price_caps"] = all(
            airport in MAXIMUM_PRICE_CAPS 
            for airport in AIRPORT_CODES.keys()
        )
        
        # Validate financial constants
        validation_results["tax_rates_configured"] = len(TAX_RATES_BY_STATE) > 0
        validation_results["service_multipliers"] = len(SERVICE_TYPE_MULTIPLIERS) > 0
        
        # Validate all price floors are less than price caps
        price_validation = True
        for airport in AIRPORT_CODES.keys():
            if (MINIMUM_PRICE_FLOORS.get(airport, 0) >= 
                MAXIMUM_PRICE_CAPS.get(airport, float('inf'))):
                price_validation = False
                break
        validation_results["price_ranges_valid"] = price_validation
        
        # Overall validation
        validation_results["all_valid"] = all(validation_results.values())
        
    except Exception as e:
        validation_results["validation_error"] = str(e)
        validation_results["all_valid"] = False
    
    return validation_results

def export_constants_to_json(filepath: str) -> bool:
    """
    Export all constants to a JSON file for documentation.
    
    Args:
        filepath: Path to output JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        export_data = {
            "airports": AIRPORT_CODES,
            "pricing": {
                "minimum_price_floors": MINIMUM_PRICE_FLOORS,
                "maximum_price_caps": MAXIMUM_PRICE_CAPS,
                "price_elasticity": PRICE_ELASTICITY,
            },
            "marketing": {
                "channels": MARKETING_CHANNELS,
                "budget_allocation": DEFAULT_CHANNEL_BUDGET_ALLOCATION,
            },
            "validation": validate_constants(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"Error exporting constants: {e}")
        return False

# Run validation on module import
constants_validation = validate_constants()
if not constants_validation.get("all_valid", False):
    import warnings
    warnings.warn(
        f"Constants validation failed: {constants_validation}",
        RuntimeWarning
    )
