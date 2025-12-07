"""
SQLAlchemy ORM Models
Project: AI Commercial Platform for Airport Parking

This module defines all database models using SQLAlchemy ORM.
Includes models for bookings, pricing, marketing, franchisees, website, call center, and AI.

Reference: [About The Role: 10+ years of live booking, pricing, marketing, behavioural, and operational data]
"""

import uuid
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date, Time,
    Text, JSON, ForeignKey, BigInteger, Numeric, Enum as SQLEnum,
    UniqueConstraint, CheckConstraint, Index, event, DDL
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.orm import relationship, validates, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func, expression

# Base class for all models
Base = declarative_base()

# ============================================================================
# ENUMS
# ============================================================================

class BookingStatus(str, Enum):
    """Status of a parking booking."""
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    MODIFIED = "modified"

class ServiceType(str, Enum):
    """Types of parking services."""
    ECONOMY = "economy"
    COVERED = "covered"
    VALET = "valet"
    PREMIUM = "premium"
    EV_CHARGING = "ev_charging"

class PaymentStatus(str, Enum):
    """Payment status."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

class FranchiseeRating(str, Enum):
    """Franchisee performance ratings."""
    EXCELLENT = "excellent"    # 4.5-5.0
    GOOD = "good"            # 4.0-4.49
    FAIR = "fair"            # 3.5-3.99
    POOR = "poor"            # 3.0-3.49
    CRITICAL = "critical"    # < 3.0

class CallType(str, Enum):
    """Types of call center calls."""
    ARRIVAL = "arrival"
    RETURN = "return"
    MODIFICATION = "modification"
    REFUND = "refund"
    COMPLAINT = "complaint"
    GENERAL = "general"

class CallResolution(str, Enum):
    """Call resolution status."""
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    PENDING = "pending"
    FAILED = "failed"

class ModelType(str, Enum):
    """Types of AI models."""
    DEMAND_FORECAST = "demand_forecast"
    PRICE_ELASTICITY = "price_elasticity"
    ROAS_PREDICTION = "roas_prediction"
    FRAUD_DETECTION = "fraud_detection"
    ABANDONMENT_PREDICTION = "abandonment_prediction"
    FRANCHISEE_SCORING = "franchisee_scoring"

# ============================================================================
# CORE MODELS
# ============================================================================

class Customer(Base):
    """
    Customer information model.
    
    Reference: [About The Role: Customer data for personalization]
    """
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Marketing preferences
    marketing_opt_in = Column(Boolean, default=True)
    email_opt_in = Column(Boolean, default=True)
    sms_opt_in = Column(Boolean, default=False)
    
    # Customer classification
    customer_type = Column(String(50), default="regular")  # regular, corporate, vip
    loyalty_points = Column(Integer, default=0)
    customer_segment = Column(String(100), nullable=True)  # AI-calculated segment
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_active = Column(DateTime, nullable=True)
    
    # Relationships
    bookings = relationship("Booking", back_populates="customer", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="customer", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_customer_email', 'email'),
        Index('idx_customer_created', 'created_at'),
        Index('idx_customer_segment', 'customer_segment'),
    )
    
    @hybrid_property
    def full_name(self) -> str:
        """Get customer's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @hybrid_property
    def booking_count(self) -> int:
        """Get total number of bookings."""
        return len(self.bookings) if self.bookings else 0
    
    @hybrid_property
    def total_spent(self) -> Decimal:
        """Get total amount spent by customer."""
        if not self.bookings:
            return Decimal('0.00')
        return sum(booking.total_price for booking in self.bookings if booking.total_price)
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        if '@' not in email:
            raise ValueError("Invalid email address")
        return email.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'email': self.email,
            'phone': self.phone,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'customer_type': self.customer_type,
            'loyalty_points': self.loyalty_points,
            'booking_count': self.booking_count,
            'total_spent': float(self.total_spent) if self.total_spent else 0.0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None,
        }


class Booking(Base):
    """
    Parking booking model - core transactional entity.
    
    Reference: [About The Role: Live booking data]
    """
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    reference_code = Column(String(20), unique=True, nullable=False, index=True)
    
    # Customer
    customer_id = Column(Integer, ForeignKey('customers.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Airport and timing
    airport_code = Column(String(3), nullable=False, index=True)
    terminal = Column(String(10), nullable=True)
    
    check_in = Column(DateTime, nullable=False, index=True)
    check_out = Column(DateTime, nullable=False, index=True)
    actual_check_in = Column(DateTime, nullable=True)
    actual_check_out = Column(DateTime, nullable=True)
    
    # Booking details
    status = Column(SQLEnum(BookingStatus), default=BookingStatus.CONFIRMED, nullable=False)
    vehicle_license = Column(String(20), nullable=False)
    vehicle_make = Column(String(50), nullable=True)
    vehicle_model = Column(String(50), nullable=True)
    vehicle_color = Column(String(30), nullable=True)
    
    # Pricing
    base_price = Column(Numeric(10, 2), nullable=False)
    service_fee = Column(Numeric(10, 2), default=0.00)
    tax_amount = Column(Numeric(10, 2), default=0.00)
    discount_amount = Column(Numeric(10, 2), default=0.00)
    late_fee = Column(Numeric(10, 2), default=0.00)
    total_price = Column(Numeric(10, 2), nullable=False)
    
    # Payment
    payment_status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    payment_method = Column(String(50), nullable=True)
    payment_transaction_id = Column(String(100), nullable=True)
    
    # Franchisee assignment
    franchisee_id = Column(Integer, ForeignKey('franchisees.id'), nullable=True, index=True)
    driver_id = Column(Integer, ForeignKey('drivers.id'), nullable=True)
    
    # Marketing attribution
    utm_source = Column(String(100), nullable=True)
    utm_medium = Column(String(100), nullable=True)
    utm_campaign = Column(String(100), nullable=True)
    utm_content = Column(String(100), nullable=True)
    utm_term = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="bookings")
    franchisee = relationship("Franchisee", back_populates="bookings")
    driver = relationship("Driver", back_populates="bookings")
    services = relationship("BookingService", back_populates="booking", cascade="all, delete-orphan")
    modifications = relationship("BookingModification", back_populates="booking", cascade="all, delete-orphan")
    price_history = relationship("PriceHistory", back_populates="booking", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_booking_airport_date', 'airport_code', 'check_in'),
        Index('idx_booking_status_date', 'status', 'created_at'),
        Index('idx_booking_customer_date', 'customer_id', 'created_at'),
        Index('idx_booking_franchisee_date', 'franchisee_id', 'check_in'),
        CheckConstraint('check_out > check_in', name='check_checkout_after_checkin'),
        CheckConstraint('total_price >= 0', name='check_positive_price'),
    )
    
    @hybrid_property
    def duration_days(self) -> float:
        """Calculate booking duration in days."""
        if self.check_in and self.check_out:
            duration = self.check_out - self.check_in
            return duration.total_seconds() / (24 * 3600)
        return 0.0
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if booking is currently active."""
        now = datetime.utcnow()
        return (
            self.status in [BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN] and
            self.check_in <= now <= self.check_out
        )
    
    @hybrid_property
    def is_late_checkout(self) -> bool:
        """Check if customer checked out late."""
        if self.actual_check_out and self.check_out:
            return self.actual_check_out > self.check_out
        return False
    
    @hybrid_property
    def late_minutes(self) -> int:
        """Calculate late checkout minutes."""
        if self.is_late_checkout and self.actual_check_out and self.check_out:
            late_duration = self.actual_check_out - self.check_out
            return int(late_duration.total_seconds() / 60)
        return 0
    
    def calculate_late_fee(self) -> Decimal:
        """Calculate late fee based on late minutes."""
        from ..config.constants import LATE_FEE_STRUCTURE
        
        late_minutes = self.late_minutes
        if late_minutes <= 0:
            return Decimal('0.00')
        
        # Find applicable fee tier
        for max_minutes, fee in LATE_FEE_STRUCTURE:
            if late_minutes <= max_minutes:
                return Decimal(str(fee))
        
        # If exceeds all tiers, use max fee
        return Decimal(str(LATE_FEE_STRUCTURE[-1][1]))
    
    @validates('airport_code')
    def validate_airport_code(self, key, airport_code):
        """Validate airport code format."""
        from ..config.constants import validate_airport_code as validate_code
        if not validate_code(airport_code):
            raise ValueError(f"Invalid airport code: {airport_code}")
        return airport_code.upper()
    
    @validates('reference_code')
    def validate_reference_code(self, key, reference_code):
        """Ensure reference code is uppercase."""
        return reference_code.upper()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'reference_code': self.reference_code,
            'customer_id': self.customer_id,
            'airport_code': self.airport_code,
            'check_in': self.check_in.isoformat() if self.check_in else None,
            'check_out': self.check_out.isoformat() if self.check_out else None,
            'status': self.status.value,
            'duration_days': self.duration_days,
            'total_price': float(self.total_price) if self.total_price else 0.0,
            'payment_status': self.payment_status.value,
            'franchisee_id': self.franchisee_id,
            'is_active': self.is_active,
            'is_late_checkout': self.is_late_checkout,
            'late_minutes': self.late_minutes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class BookingService(Base):
    """
    Additional services for a booking.
    """
    __tablename__ = "booking_services"
    
    id = Column(Integer, primary_key=True)
    booking_id = Column(Integer, ForeignKey('bookings.id', ondelete='CASCADE'), nullable=False, index=True)
    service_type = Column(SQLEnum(ServiceType), nullable=False)
    service_name = Column(String(100), nullable=False)
    unit_price = Column(Numeric(10, 2), nullable=False)
    quantity = Column(Integer, default=1)
    total_price = Column(Numeric(10, 2), nullable=False)
    
    # Service-specific data
    metadata = Column(JSON, nullable=True)  # EV charging details, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="services")
    
    __table_args__ = (
        Index('idx_booking_service_type', 'booking_id', 'service_type'),
        CheckConstraint('unit_price >= 0', name='check_positive_unit_price'),
        CheckConstraint('quantity > 0', name='check_positive_quantity'),
    )


class BookingModification(Base):
    """
    Track modifications to bookings.
    """
    __tablename__ = "booking_modifications"
    
    id = Column(Integer, primary_key=True)
    booking_id = Column(Integer, ForeignKey('bookings.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Modification details
    modification_type = Column(String(50), nullable=False)  # date_change, service_change, cancellation
    old_values = Column(JSON, nullable=False)
    new_values = Column(JSON, nullable=False)
    price_change = Column(Numeric(10, 2), nullable=True)
    
    # Who made the change
    modified_by = Column(String(100), nullable=False)  # customer, franchisee, system, call_center
    reason = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="modifications")
    
    __table_args__ = (
        Index('idx_modification_booking_date', 'booking_id', 'created_at'),
    )


# ============================================================================
# PRICING MODELS
# ============================================================================

class PriceHistory(Base):
    """
    Historical price data for demand forecasting and analysis.
    
    Reference: [Core Responsibilities: 1. Commercial & Pricing AI]
    """
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True)
    
    # Context
    airport_code = Column(String(3), nullable=False, index=True)
    service_type = Column(SQLEnum(ServiceType), nullable=False)
    date = Column(Date, nullable=False, index=True)
    hour = Column(Integer, nullable=True)  # For intra-day pricing
    
    # Price points
    base_price = Column(Numeric(10, 2), nullable=False)
    final_price = Column(Numeric(10, 2), nullable=False)
    competitor_avg_price = Column(Numeric(10, 2), nullable=True)
    competitor_min_price = Column(Numeric(10, 2), nullable=True)
    competitor_max_price = Column(Numeric(10, 2), nullable=True)
    
    # Demand metrics
    bookings_count = Column(Integer, default=0)
    capacity_utilization = Column(Float, nullable=True)  # 0.0 to 1.0
    demand_factor = Column(Float, nullable=True)
    
    # External factors
    day_of_week = Column(Integer, nullable=True)  # 0=Monday, 6=Sunday
    is_holiday = Column(Boolean, default=False)
    weather_score = Column(Float, nullable=True)  # 0.0 to 1.0 (bad to good)
    flight_count = Column(Integer, nullable=True)
    
    # Booking reference if applicable
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="price_history")
    
    # Indexes
    __table_args__ = (
        Index('idx_price_history_airport_date', 'airport_code', 'date'),
        Index('idx_price_history_service_date', 'service_type', 'date'),
        UniqueConstraint('airport_code', 'service_type', 'date', 'hour', name='uq_price_history'),
        CheckConstraint('base_price >= 0', name='check_positive_base_price'),
        CheckConstraint('final_price >= 0', name='check_positive_final_price'),
        CheckConstraint('hour >= 0 AND hour < 24', name='check_valid_hour'),
    )
    
    @hybrid_property
    def price_premium(self) -> float:
        """Calculate price premium over base."""
        if self.base_price and self.base_price > 0:
            return float((self.final_price - self.base_price) / self.base_price)
        return 0.0
    
    @hybrid_property
    def competitor_spread(self) -> float:
        """Calculate competitor price spread."""
        if self.competitor_min_price and self.competitor_max_price:
            return float(self.competitor_max_price - self.competitor_min_price)
        return 0.0


class PriceExperiment(Base):
    """
    A/B testing for pricing strategies.
    
    Reference: [Core Responsibilities: Create price A/B test engines]
    """
    __tablename__ = "price_experiments"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Experiment details
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    airport_code = Column(String(3), nullable=False, index=True)
    service_type = Column(SQLEnum(ServiceType), nullable=False)
    
    # Experiment variants
    control_price = Column(Numeric(10, 2), nullable=False)
    variant_a_price = Column(Numeric(10, 2), nullable=True)
    variant_b_price = Column(Numeric(10, 2), nullable=True)
    
    # Timing
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False)
    actual_end_date = Column(DateTime, nullable=True)
    
    # Sample sizes
    target_sample_size = Column(Integer, default=1000)
    actual_sample_size = Column(Integer, default=0)
    
    # Randomization
    randomization_seed = Column(String(100), nullable=True)
    
    # Results
    control_conversion_rate = Column(Float, nullable=True)
    variant_a_conversion_rate = Column(Float, nullable=True)
    variant_b_conversion_rate = Column(Float, nullable=True)
    
    control_revenue_per_visitor = Column(Numeric(10, 2), nullable=True)
    variant_a_revenue_per_visitor = Column(Numeric(10, 2), nullable=True)
    variant_b_revenue_per_visitor = Column(Numeric(10, 2), nullable=True)
    
    statistical_significance = Column(Float, nullable=True)  # p-value
    winning_variant = Column(String(10), nullable=True)  # control, variant_a, variant_b
    
    # Status
    status = Column(String(50), default="draft")  # draft, active, completed, cancelled
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_experiment_status_date', 'status', 'start_date'),
        UniqueConstraint('airport_code', 'service_type', 'name', name='uq_experiment_name'),
        CheckConstraint('end_date > start_date', name='check_experiment_dates'),
        CheckConstraint('control_price > 0', name='check_positive_control_price'),
    )
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        now = datetime.utcnow()
        return (
            self.status == "active" and
            self.start_date <= now <= (self.actual_end_date or self.end_date)
        )
    
    @hybrid_property
    def duration_days(self) -> float:
        """Calculate experiment duration in days."""
        if self.start_date and (self.actual_end_date or self.end_date):
            end_date = self.actual_end_date or self.end_date
            duration = end_date - self.start_date
            return duration.total_seconds() / (24 * 3600)
        return 0.0


class CompetitorPrice(Base):
    """
    Competitor price monitoring data.
    
    Reference: [Core Responsibilities: 3. Competitor & Market Intelligence]
    """
    __tablename__ = "competitor_prices"
    
    id = Column(Integer, primary_key=True)
    
    # Competitor identification
    competitor_name = Column(String(100), nullable=False, index=True)
    competitor_location = Column(String(200), nullable=True)
    airport_code = Column(String(3), nullable=False, index=True)
    
    # Price data
    service_type = Column(SQLEnum(ServiceType), nullable=False)
    date = Column(Date, nullable=False, index=True)
    scraped_at = Column(DateTime, nullable=False, index=True)
    
    # Price points
    base_price = Column(Numeric(10, 2), nullable=False)
    final_price = Column(Numeric(10, 2), nullable=False)
    
    # Promotions and discounts
    has_promotion = Column(Boolean, default=False)
    promotion_text = Column(Text, nullable=True)
    discount_percentage = Column(Float, nullable=True)
    
    # Availability
    is_available = Column(Boolean, default=True)
    estimated_availability = Column(String(50), nullable=True)  # low, medium, high
    
    # Metadata
    source_url = Column(Text, nullable=True)
    scrape_method = Column(String(50), nullable=False)  # api, web_scrape, manual
    confidence_score = Column(Float, default=1.0)  # 0.0 to 1.0
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_competitor_airport_date', 'competitor_name', 'airport_code', 'date'),
        Index('idx_competitor_service_date', 'service_type', 'date'),
        UniqueConstraint('competitor_name', 'airport_code', 'service_type', 'date', 
                        name='uq_competitor_price_daily'),
        CheckConstraint('base_price >= 0', name='check_positive_competitor_price'),
        CheckConstraint('final_price >= 0', name='check_positive_competitor_final_price'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', 
                       name='check_valid_confidence'),
    )
    
    @hybrid_property
    def price_difference(self, our_price: Decimal) -> Decimal:
        """Calculate price difference from our price."""
        return our_price - self.final_price
    
    @hybrid_property
    def is_price_shock(self) -> bool:
        """Check if this represents a significant price change."""
        # Would compare with historical prices
        return False


# ============================================================================
# MARKETING MODELS
# ============================================================================

class MarketingCampaign(Base):
    """
    Marketing campaign tracking.
    
    Reference: [Core Responsibilities: 2. Marketing & Multi-Channel Budget Optimisation]
    """
    __tablename__ = "marketing_campaigns"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Campaign details
    name = Column(String(200), nullable=False)
    channel = Column(String(50), nullable=False, index=True)  # google_ads, meta_ads, email, etc.
    campaign_type = Column(String(50), nullable=False)  # acquisition, retention, reactivation
    
    # Targeting
    target_airports = Column(JSON, nullable=True)  # List of airport codes
    target_audience = Column(String(100), nullable=True)
    target_dates = Column(JSON, nullable=True)  # Date range or specific dates
    
    # Budget and spend
    total_budget = Column(Numeric(12, 2), nullable=False)
    daily_budget = Column(Numeric(12, 2), nullable=True)
    spent_to_date = Column(Numeric(12, 2), default=0.00)
    
    # Goals
    target_roas = Column(Float, nullable=True)  # Return on Ad Spend
    target_cpa = Column(Numeric(10, 2), nullable=True)  # Cost Per Acquisition
    target_clicks = Column(Integer, nullable=True)
    target_conversions = Column(Integer, nullable=True)
    
    # Timing
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False, index=True)
    
    # Status
    status = Column(String(50), default="draft")  # draft, active, paused, completed, cancelled
    
    # Performance (updated daily)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    revenue = Column(Numeric(12, 2), default=0.00)
    
    # Attribution
    attribution_window_days = Column(Integer, default=30)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ad_performances = relationship("AdPerformance", back_populates="campaign", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_campaign_channel_status', 'channel', 'status'),
        Index('idx_campaign_dates', 'start_date', 'end_date'),
        CheckConstraint('end_date > start_date', name='check_campaign_dates'),
        CheckConstraint('total_budget >= 0', name='check_positive_budget'),
        CheckConstraint('spent_to_date >= 0', name='check_positive_spend'),
    )
    
    @hybrid_property
    def days_remaining(self) -> int:
        """Calculate days remaining in campaign."""
        today = date.today()
        if today > self.end_date:
            return 0
        return (self.end_date - today).days
    
    @hybrid_property
    def roas(self) -> float:
        """Calculate Return on Ad Spend."""
        if self.spent_to_date and self.spent_to_date > 0:
            return float(self.revenue / self.spent_to_date)
        return 0.0
    
    @hybrid_property
    def cpa(self) -> float:
        """Calculate Cost Per Acquisition."""
        if self.conversions and self.conversions > 0:
            return float(self.spent_to_date / self.conversions)
        return 0.0
    
    @hybrid_property
    def ctr(self) -> float:
        """Calculate Click-Through Rate."""
        if self.impressions and self.impressions > 0:
            return float(self.clicks / self.impressions)
        return 0.0
    
    @hybrid_property
    def conversion_rate(self) -> float:
        """Calculate Conversion Rate."""
        if self.clicks and self.clicks > 0:
            return float(self.conversions / self.clicks)
        return 0.0


class AdPerformance(Base):
    """
    Daily performance data for ads/campaigns.
    """
    __tablename__ = "ad_performance"
    
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey('marketing_campaigns.id', ondelete='CASCADE'), 
                        nullable=False, index=True)
    
    # Date and airport
    date = Column(Date, nullable=False, index=True)
    airport_code = Column(String(3), nullable=True, index=True)
    
    # Performance metrics
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    cost = Column(Numeric(12, 2), default=0.00)
    revenue = Column(Numeric(12, 2), default=0.00)
    
    # Quality metrics
    average_position = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    campaign = relationship("MarketingCampaign", back_populates="ad_performances")
    
    __table_args__ = (
        Index('idx_ad_performance_campaign_date', 'campaign_id', 'date'),
        UniqueConstraint('campaign_id', 'date', 'airport_code', name='uq_ad_performance_daily'),
        CheckConstraint('impressions >= 0', name='check_non_negative_impressions'),
        CheckConstraint('clicks >= 0', name='check_non_negative_clicks'),
        CheckConstraint('cost >= 0', name='check_non_negative_cost'),
    )


class AttributionEvent(Base):
    """
    Multi-touch attribution events.
    
    Reference: [Core Responsibilities: Marketing attribution models]
    """
    __tablename__ = "attribution_events"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # User identification
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    device_id = Column(String(100), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # ad_click, email_open, website_visit, conversion
    event_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Channel attribution
    channel = Column(String(50), nullable=False, index=True)
    campaign_id = Column(Integer, ForeignKey('marketing_campaigns.id'), nullable=True, index=True)
    ad_id = Column(String(100), nullable=True)
    
    # UTM parameters
    utm_source = Column(String(100), nullable=True)
    utm_medium = Column(String(100), nullable=True)
    utm_campaign = Column(String(100), nullable=True)
    utm_content = Column(String(100), nullable=True)
    utm_term = Column(String(100), nullable=True)
    
    # Conversion details (if applicable)
    conversion_value = Column(Numeric(10, 2), nullable=True)
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True, index=True)
    
    # Metadata
    page_url = Column(Text, nullable=True)
    referrer_url = Column(Text, nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer")
    booking = relationship("Booking")
    campaign = relationship("MarketingCampaign")
    
    __table_args__ = (
        Index('idx_attribution_customer_channel', 'customer_id', 'channel', 'event_timestamp'),
        Index('idx_attribution_conversion_events', 'event_type', 'event_timestamp'),
        Index('idx_attribution_booking', 'booking_id'),
    )


# ============================================================================
# FRANCHISEE MODELS
# ============================================================================

class Franchisee(Base):
    """
    Franchisee information and contract details.
    
    Reference: [Core Responsibilities: 4. Franchisee Performance, Risk & Fraud Detection]
    """
    __tablename__ = "franchisees"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Basic information
    name = Column(String(200), nullable=False)
    contact_name = Column(String(200), nullable=False)
    contact_email = Column(String(255), nullable=False)
    contact_phone = Column(String(20), nullable=False)
    
    # Location
    airport_code = Column(String(3), nullable=False, index=True)
    location_address = Column(Text, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Contract details
    contract_start = Column(Date, nullable=False)
    contract_end = Column(Date, nullable=True)
    contract_type = Column(String(50), nullable=False)  # standard, premium, exclusive
    commission_rate = Column(Float, nullable=False)  # Percentage (e.g., 0.70 for 70%)
    
    # Capacity
    total_spots = Column(Integer, nullable=False)
    covered_spots = Column(Integer, default=0)
    ev_charging_spots = Column(Integer, default=0)
    
    # Fleet
    shuttle_count = Column(Integer, default=2)
    driver_count = Column(Integer, default=4)
    
    # Status
    status = Column(String(50), default="active")  # active, suspended, terminated
    onboarding_date = Column(Date, nullable=True)
    termination_date = Column(Date, nullable=True)
    termination_reason = Column(Text, nullable=True)
    
    # Financial
    total_revenue = Column(Numeric(12, 2), default=0.00)
    total_commission = Column(Numeric(12, 2), default=0.00)
    last_payout_date = Column(Date, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bookings = relationship("Booking", back_populates="franchisee", cascade="all, delete-orphan")
    drivers = relationship("Driver", back_populates="franchisee", cascade="all, delete-orphan")
    performances = relationship("FranchiseePerformance", back_populates="franchisee", 
                               cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_franchisee_airport_status', 'airport_code', 'status'),
        Index('idx_franchisee_contract_dates', 'contract_start', 'contract_end'),
        CheckConstraint('commission_rate >= 0 AND commission_rate <= 1', name='check_valid_commission'),
        CheckConstraint('total_spots > 0', name='check_positive_spots'),
    )
    
    @hybrid_property
    def contract_days_remaining(self) -> int:
        """Calculate days remaining in contract."""
        if not self.contract_end:
            return None
        today = date.today()
        if today > self.contract_end:
            return 0
        return (self.contract_end - today).days
    
    @hybrid_property
    def capacity_utilization(self, date_filter: date = None) -> float:
        """Calculate capacity utilization."""
        # Would calculate based on bookings
        return 0.0
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if franchisee is currently active."""
        return self.status == "active"


class FranchiseePerformance(Base):
    """
    Daily performance metrics for franchisees.
    """
    __tablename__ = "franchisee_performances"
    
    id = Column(Integer, primary_key=True)
    franchisee_id = Column(Integer, ForeignKey('franchisees.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Booking metrics
    total_bookings = Column(Integer, default=0)
    completed_bookings = Column(Integer, default=0)
    cancelled_bookings = Column(Integer, default=0)
    no_show_bookings = Column(Integer, default=0)
    
    # Revenue metrics
    total_revenue = Column(Numeric(12, 2), default=0.00)
    average_booking_value = Column(Numeric(10, 2), nullable=True)
    
    # Service metrics
    on_time_performance = Column(Float, nullable=True)  # 0.0 to 1.0
    average_shuttle_wait_time = Column(Float, nullable=True)  # minutes
    customer_rating = Column(Float, nullable=True)  # 1.0 to 5.0
    
    # Operational metrics
    capacity_utilization = Column(Float, nullable=True)  # 0.0 to 1.0
    spot_turnover_rate = Column(Float, nullable=True)  # times per day
    
    # Risk metrics
    complaint_count = Column(Integer, default=0)
    incident_count = Column(Integer, default=0)
    refund_amount = Column(Numeric(10, 2), default=0.00)
    late_fee_collected = Column(Numeric(10, 2), default=0.00)
    
    # FPS Score
    fps_score = Column(Float, nullable=True)  # 0.0 to 100.0
    fps_rating = Column(SQLEnum(FranchiseeRating), nullable=True)
    
    # Flags
    has_anomaly = Column(Boolean, default=False)
    requires_intervention = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    franchisee = relationship("Franchisee", back_populates="performances")
    
    __table_args__ = (
        Index('idx_franchisee_performance_date', 'franchisee_id', 'date'),
        UniqueConstraint('franchisee_id', 'date', name='uq_franchisee_daily_performance'),
        CheckConstraint('customer_rating >= 1 AND customer_rating <= 5', name='check_valid_rating'),
        CheckConstraint('fps_score >= 0 AND fps_score <= 100', name='check_valid_fps_score'),
    )
    
    @hybrid_property
    def cancellation_rate(self) -> float:
        """Calculate cancellation rate."""
        if self.total_bookings > 0:
            return float(self.cancelled_bookings / self.total_bookings)
        return 0.0
    
    @hybrid_property
    def no_show_rate(self) -> float:
        """Calculate no-show rate."""
        if self.total_bookings > 0:
            return float(self.no_show_bookings / self.total_bookings)
        return 0.0


class Driver(Base):
    """
    Driver information for franchisees.
    """
    __tablename__ = "drivers"
    
    id = Column(Integer, primary_key=True)
    franchisee_id = Column(Integer, ForeignKey('franchisees.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    
    # Personal information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=False)
    email = Column(String(255), nullable=True)
    
    # Employment
    hire_date = Column(Date, nullable=False)
    termination_date = Column(Date, nullable=True)
    status = Column(String(50), default="active")  # active, on_leave, terminated
    
    # License and certification
    driver_license_number = Column(String(50), nullable=False)
    license_expiry = Column(Date, nullable=False)
    background_check_date = Column(Date, nullable=True)
    training_completion_date = Column(Date, nullable=True)
    
    # Vehicle assignment
    assigned_vehicle = Column(String(50), nullable=True)
    vehicle_type = Column(String(50), nullable=True)  # sedan, van, shuttle
    
    # Performance
    total_trips = Column(Integer, default=0)
    average_rating = Column(Float, nullable=True)
    last_performance_review = Column(Date, nullable=True)
    
    # Risk factors
    incident_count = Column(Integer, default=0)
    complaint_count = Column(Integer, default=0)
    late_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    franchisee = relationship("Franchisee", back_populates="drivers")
    bookings = relationship("Booking", back_populates="driver", cascade="all, delete-orphan")
    performances = relationship("DriverPerformance", back_populates="driver", 
                               cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_driver_franchisee_status', 'franchisee_id', 'status'),
        CheckConstraint('average_rating >= 1 AND average_rating <= 5', name='check_valid_driver_rating'),
    )
    
    @hybrid_property
    def full_name(self) -> str:
        """Get driver's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if driver is currently active."""
        return self.status == "active"


class DriverPerformance(Base):
    """
    Daily performance metrics for drivers.
    """
    __tablename__ = "driver_performances"
    
    id = Column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey('drivers.id', ondelete='CASCADE'), 
                      nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Trip metrics
    total_trips = Column(Integer, default=0)
    completed_trips = Column(Integer, default=0)
    on_time_trips = Column(Integer, default=0)
    late_trips = Column(Integer, default=0)
    
    # Time metrics
    total_driving_hours = Column(Float, default=0.0)
    average_trip_duration = Column(Float, nullable=True)  # minutes
    average_wait_time = Column(Float, nullable=True)  # minutes
    
    # Customer feedback
    customer_ratings = Column(JSON, nullable=True)  # Array of ratings
    average_rating = Column(Float, nullable=True)
    complaint_count = Column(Integer, default=0)
    
    # Operational metrics
    distance_covered = Column(Float, nullable=True)  # kilometers/miles
    fuel_consumption = Column(Float, nullable=True)  # liters/gallons
    
    # Risk score
    risk_score = Column(Float, nullable=True)  # 0.0 to 1.0
    risk_factors = Column(JSON, nullable=True)  # List of risk factors
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    driver = relationship("Driver", back_populates="performances")
    
    __table_args__ = (
        Index('idx_driver_performance_date', 'driver_id', 'date'),
        UniqueConstraint('driver_id', 'date', name='uq_driver_daily_performance'),
        CheckConstraint('average_rating >= 1 AND average_rating <= 5', name='check_valid_performance_rating'),
    )
    
    @hybrid_property
    def on_time_performance(self) -> float:
        """Calculate on-time performance rate."""
        if self.total_trips > 0:
            return float(self.on_time_trips / self.total_trips)
        return 0.0
    
    @hybrid_property
    def late_performance(self) -> float:
        """Calculate late performance rate."""
        if self.total_trips > 0:
            return float(self.late_trips / self.total_trips)
        return 0.0


# ============================================================================
# WEBSITE MODELS
# ============================================================================

class UserSession(Base):
    """
    Website user session tracking.
    
    Reference: [Core Responsibilities: 5. Website Behaviour, Conversion & Personalisation AI]
    """
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # User identification
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    device_id = Column(String(100), nullable=True)
    
    # Session details
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    
    # Technical details
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)
    device_type = Column(String(50), nullable=True)  # desktop, mobile, tablet
    
    # Referral
    referrer_url = Column(Text, nullable=True)
    landing_page = Column(Text, nullable=False)
    
    # Marketing attribution
    utm_source = Column(String(100), nullable=True)
    utm_medium = Column(String(100), nullable=True)
    utm_campaign = Column(String(100), nullable=True)
    utm_content = Column(String(100), nullable=True)
    utm_term = Column(String(100), nullable=True)
    
    # Session attributes
    is_bounce = Column(Boolean, default=True)
    is_converted = Column(Boolean, default=False)
    conversion_value = Column(Numeric(10, 2), nullable=True)
    
    # AI personalization
    price_sensitivity_score = Column(Float, nullable=True)  # 0.0 to 1.0
    abandonment_probability = Column(Float, nullable=True)  # 0.0 to 1.0
    user_segment = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="sessions")
    page_views = relationship("PageView", back_populates="session", cascade="all, delete-orphan")
    conversion_events = relationship("ConversionEvent", back_populates="session", 
                                    cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_session_customer_time', 'customer_id', 'start_time'),
        Index('idx_session_conversion', 'is_converted', 'start_time'),
        Index('idx_session_marketing', 'utm_source', 'utm_campaign'),
    )
    
    @hybrid_property
    def duration_seconds(self) -> int:
        """Calculate session duration in seconds."""
        if self.end_time and self.start_time:
            duration = self.end_time - self.start_time
            return int(duration.total_seconds())
        return 0
    
    @hybrid_property
    def page_view_count(self) -> int:
        """Get total page views in session."""
        return len(self.page_views) if self.page_views else 0
    
    @hybrid_method
    def is_abandoned(self, threshold_minutes: int = 5) -> bool:
        """Check if session is abandoned based on inactivity threshold."""
        if self.end_time:
            return False  # Session has ended
        # Check if last activity was more than threshold ago
        last_page_view = max((pv.timestamp for pv in self.page_views), default=self.start_time)
        inactive_time = datetime.utcnow() - last_page_view
        return inactive_time.total_seconds() > (threshold_minutes * 60)


class PageView(Base):
    """
    Individual page view tracking.
    """
    __tablename__ = "page_views"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('user_sessions.id', ondelete='CASCADE'), 
                       nullable=False, index=True)
    
    # Page details
    page_url = Column(Text, nullable=False)
    page_title = Column(String(500), nullable=True)
    page_category = Column(String(100), nullable=True)  # homepage, search, booking, etc.
    
    # Timing
    timestamp = Column(DateTime, nullable=False, index=True)
    time_on_page = Column(Float, nullable=True)  # seconds
    
    # User actions
    scroll_depth = Column(Float, nullable=True)  # 0.0 to 1.0
    clicks = Column(Integer, default=0)
    form_interactions = Column(Integer, default=0)
    
    # Technical
    load_time = Column(Float, nullable=True)  # milliseconds
    is_exit = Column(Boolean, default=False)
    
    # Personalization
    shown_price = Column(Numeric(10, 2), nullable=True)
    shown_discount = Column(Numeric(10, 2), nullable=True)
    cta_variant = Column(String(50), nullable=True)  # A/B test variant
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("UserSession", back_populates="page_views")
    
    __table_args__ = (
        Index('idx_pageview_session_sequence', 'session_id', 'timestamp'),
        Index('idx_pageview_category_time', 'page_category', 'timestamp'),
    )


class ConversionEvent(Base):
    """
    Conversion events (booking steps, form submissions, etc.).
    """
    __tablename__ = "conversion_events"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('user_sessions.id', ondelete='CASCADE'), 
                       nullable=False, index=True)
    
    # Event details
    event_type = Column(String(100), nullable=False)  # search_initiated, price_viewed, booking_started, etc.
    event_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Event data
    event_data = Column(JSON, nullable=True)  # Search parameters, price details, etc.
    event_value = Column(Numeric(10, 2), nullable=True)
    
    # Booking reference if applicable
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("UserSession", back_populates="conversion_events")
    booking = relationship("Booking")
    
    __table_args__ = (
        Index('idx_conversion_event_type_time', 'event_type', 'event_timestamp'),
        Index('idx_conversion_booking', 'booking_id'),
    )


# ============================================================================
# CALL CENTER MODELS
# ============================================================================

class CallLog(Base):
    """
    Call center call logs.
    
    Reference: [Core Responsibilities: 6. AI Call Centre (DRT & CRS Automation)]
    """
    __tablename__ = "call_logs"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Call details
    call_sid = Column(String(100), nullable=False, unique=True)  # Twilio SID or similar
    call_type = Column(SQLEnum(CallType), nullable=False, index=True)
    call_direction = Column(String(20), nullable=False)  # inbound, outbound
    
    # Participants
    from_number = Column(String(20), nullable=False)
    to_number = Column(String(20), nullable=False)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    agent_id = Column(String(100), nullable=True)  # Human agent or AI agent
    
    # Timing
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    queue_time = Column(Integer, nullable=True)  # seconds in queue
    hold_time = Column(Integer, nullable=True)  # seconds on hold
    
    # Call content
    transcription = Column(Text, nullable=True)
    intent = Column(String(100), nullable=True)  # AI-detected intent
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    
    # Resolution
    resolution = Column(SQLEnum(CallResolution), nullable=True)
    resolution_details = Column(Text, nullable=True)
    automated_resolution = Column(Boolean, default=False)
    
    # Booking reference if applicable
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True, index=True)
    
    # Metadata
    recording_url = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer")
    booking = relationship("Booking")
    resolutions = relationship("CallResolution", back_populates="call", cascade="all, delete-orphan")
    payments = relationship("PaymentTransaction", back_populates="call", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_call_customer_time', 'customer_id', 'start_time'),
        Index('idx_call_type_time', 'call_type', 'start_time'),
        Index('idx_call_automated_resolution', 'automated_resolution', 'start_time'),
        CheckConstraint('queue_time >= 0', name='check_non_negative_queue_time'),
    )
    
    @hybrid_property
    def duration_seconds(self) -> int:
        """Calculate call duration in seconds."""
        if self.end_time and self.start_time:
            duration = self.end_time - self.start_time
            return int(duration.total_seconds())
        return 0
    
    @hybrid_property
    def talk_time_seconds(self) -> int:
        """Calculate actual talk time (excluding queue and hold)."""
        if self.duration_seconds > 0:
            return self.duration_seconds - (self.queue_time or 0) - (self.hold_time or 0)
        return 0


class CallResolution(Base):
    """
    Detailed resolution tracking for calls.
    """
    __tablename__ = "call_resolutions"
    
    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey('call_logs.id', ondelete='CASCADE'), 
                    nullable=False, index=True)
    
    # Resolution details
    resolution_type = Column(String(100), nullable=False)  # booking_created, modification, refund, etc.
    resolution_status = Column(String(50), nullable=False)  # completed, pending, failed
    resolution_timestamp = Column(DateTime, nullable=False)
    
    # Action details
    action_taken = Column(Text, nullable=False)
    action_result = Column(Text, nullable=True)
    
    # Booking modifications
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True)
    modification_details = Column(JSON, nullable=True)
    
    # AI assistance
    ai_confidence = Column(Float, nullable=True)
    ai_suggestion = Column(Text, nullable=True)
    human_override = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    call = relationship("CallLog", back_populates="resolutions")
    booking = relationship("Booking")
    
    __table_args__ = (
        Index('idx_resolution_call_type', 'call_id', 'resolution_type'),
        Index('idx_resolution_booking', 'booking_id'),
    )


class PaymentTransaction(Base):
    """
    Payment transactions from call center.
    """
    __tablename__ = "payment_transactions"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Transaction details
    transaction_id = Column(String(100), nullable=False, unique=True)  # Stripe/PayPal ID
    payment_method = Column(String(50), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD")
    
    # Context
    call_id = Column(Integer, ForeignKey('call_logs.id'), nullable=True, index=True)
    booking_id = Column(Integer, ForeignKey('bookings.id'), nullable=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    
    # Payment details
    description = Column(String(500), nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False)  # succeeded, pending, failed, refunded
    error_message = Column(Text, nullable=True)
    
    # Security
    fraud_score = Column(Float, nullable=True)
    risk_level = Column(String(50), nullable=True)
    
    # Timestamps
    initiated_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    call = relationship("CallLog", back_populates="payments")
    booking = relationship("Booking")
    customer = relationship("Customer")
    
    __table_args__ = (
        Index('idx_payment_customer_date', 'customer_id', 'initiated_at'),
        Index('idx_payment_status_date', 'status', 'initiated_at'),
        CheckConstraint('amount > 0', name='check_positive_payment_amount'),
    )


# ============================================================================
# AI MODELS
# ============================================================================

class AIModelVersion(Base):
    """
    AI model version tracking.
    
    Reference: [Technical Stack: AI / ML: Python, forecasting models, optimisation systems]
    """
    __tablename__ = "ai_model_versions"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Model identification
    model_type = Column(SQLEnum(ModelType), nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Training details
    training_data_start = Column(Date, nullable=True)
    training_data_end = Column(Date, nullable=True)
    training_samples = Column(Integer, nullable=True)
    training_duration = Column(Float, nullable=True)  # seconds
    
    # Model parameters
    hyperparameters = Column(JSON, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    model_size = Column(Float, nullable=True)  # MB
    
    # Performance metrics
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    other_metrics = Column(JSON, nullable=True)
    
    # Deployment
    is_production = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime, nullable=True)
    deployed_by = Column(String(100), nullable=True)
    
    # File storage
    model_path = Column(Text, nullable=False)
    preprocessing_path = Column(Text, nullable=True)
    
    # Status
    status = Column(String(50), default="trained")  # trained, validated, deployed, archived
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("ModelPrediction", back_populates="model_version", 
                              cascade="all, delete-orphan")
    performances = relationship("ModelPerformance", back_populates="model_version", 
                               cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_model_type_version', 'model_type', 'version'),
        Index('idx_model_production', 'is_production', 'model_type'),
        UniqueConstraint('model_type', 'version', name='uq_model_version'),
    )
    
    @hybrid_property
    def full_name(self) -> str:
        """Get full model name with version."""
        return f"{self.model_name}_v{self.version}"


class ModelPrediction(Base):
    """
    Individual model predictions for audit and analysis.
    """
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey('ai_model_versions.id', ondelete='CASCADE'), 
                             nullable=False, index=True)
    
    # Prediction context
    prediction_type = Column(String(100), nullable=False)  # demand, price, fraud, etc.
    entity_id = Column(String(100), nullable=True)  # booking_id, customer_id, etc.
    entity_type = Column(String(100), nullable=True)
    
    # Prediction data
    input_features = Column(JSON, nullable=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Prediction results
    prediction_value = Column(JSON, nullable=False)  # Could be scalar, array, or object
    confidence_score = Column(Float, nullable=True)
    prediction_metadata = Column(JSON, nullable=True)
    
    # Ground truth (if available later)
    actual_value = Column(JSON, nullable=True)
    prediction_error = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model_version = relationship("AIModelVersion", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_type_time', 'prediction_type', 'prediction_timestamp'),
        Index('idx_prediction_entity', 'entity_type', 'entity_id'),
    )


class ModelPerformance(Base):
    """
    Model performance tracking over time.
    """
    __tablename__ = "model_performances"
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey('ai_model_versions.id', ondelete='CASCADE'), 
                             nullable=False, index=True)
    
    # Performance period
    evaluation_date = Column(Date, nullable=False, index=True)
    evaluation_window_days = Column(Integer, nullable=True)  # Rolling window size
    
    # Performance metrics
    sample_size = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)  # Mean Absolute Error
    rmse = Column(Float, nullable=True)  # Root Mean Square Error
    other_metrics = Column(JSON, nullable=True)
    
    # Drift detection
    data_drift_score = Column(Float, nullable=True)  # 0.0 to 1.0
    concept_drift_score = Column(Float, nullable=True)  # 0.0 to 1.0
    has_drift = Column(Boolean, default=False)
    
    # Business impact
    business_value = Column(Numeric(12, 2), nullable=True)
    error_cost = Column(Numeric(12, 2), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model_version = relationship("AIModelVersion", back_populates="performances")
    
    __table_args__ = (
        Index('idx_model_performance_date', 'model_version_id', 'evaluation_date'),
        UniqueConstraint('model_version_id', 'evaluation_date', name='uq_model_daily_performance'),
        CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='check_valid_accuracy'),
    )


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def create_all_tables(engine):
    """
    Create all database tables.
    
    Args:
        engine: SQLAlchemy engine
        
    Reference: [Required Experience: Production AI/data engineering]
    """
    Base.metadata.create_all(engine)
    
    # Create additional indexes and constraints
    _create_additional_indexes(engine)
    
    return True

def drop_all_tables(engine):
    """
    Drop all database tables (USE WITH CAUTION!).
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.drop_all(engine)
    return True

def _create_additional_indexes(engine):
    """Create additional performance indexes."""
    # This would contain additional composite indexes for common queries
    pass

def get_model_class(model_name: str):
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model class
        
    Returns:
        SQLAlchemy model class
        
    Raises:
        ValueError: If model not found
    """
    models_dict = {
        'Customer': Customer,
        'Booking': Booking,
        'BookingService': BookingService,
        'BookingModification': BookingModification,
        'PriceHistory': PriceHistory,
        'PriceExperiment': PriceExperiment,
        'CompetitorPrice': CompetitorPrice,
        'MarketingCampaign': MarketingCampaign,
        'AdPerformance': AdPerformance,
        'AttributionEvent': AttributionEvent,
        'Franchisee': Franchisee,
        'FranchiseePerformance': FranchiseePerformance,
        'Driver': Driver,
        'DriverPerformance': DriverPerformance,
        'UserSession': UserSession,
        'PageView': PageView,
        'ConversionEvent': ConversionEvent,
        'CallLog': CallLog,
        'CallResolution': CallResolution,
        'PaymentTransaction': PaymentTransaction,
        'AIModelVersion': AIModelVersion,
        'ModelPrediction': ModelPrediction,
        'ModelPerformance': ModelPerformance,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model not found: {model_name}")
    
    return models_dict[model_name]

# ============================================================================
# MODEL VALIDATION
# ============================================================================

def validate_all_models() -> Dict[str, List[str]]:
    """
    Validate all model definitions for consistency.
    
    Returns:
        Dict with validation results
        
    Reference: [Absolutely Essential: Design with safety, logging]
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "models_checked": 0
    }
    
    try:
        # Check all models have primary keys
        for table_name, table in Base.metadata.tables.items():
            validation_results["models_checked"] += 1
            
            if not table.primary_key.columns:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Table {table_name} has no primary key")
            
            # Check for indexes on foreign keys
            for column in table.columns:
                if column.foreign_keys:
                    has_index = any(
                        index for index in table.indexes 
                        if column in index.columns
                    )
                    if not has_index:
                        validation_results["warnings"].append(
                            f"Foreign key {table_name}.{column.name} is not indexed"
                        )
        
        # Check model relationships
        required_models = [
            'Customer', 'Booking', 'PriceHistory', 'MarketingCampaign',
            'Franchisee', 'UserSession', 'CallLog', 'AIModelVersion'
        ]
        
        for model_name in required_models:
            if model_name not in globals():
                validation_results["valid"] = False
                validation_results["errors"].append(f"Required model missing: {model_name}")
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation error: {str(e)}")
    
    return validation_results
