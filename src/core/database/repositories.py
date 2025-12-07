"""
Data Access Layer & Query Abstractions
Project: AI Commercial Platform for Airport Parking

This module implements the Repository pattern for clean data access,
providing abstracted query interfaces for all business entities.

Reference: [About The Role: Turning 10+ years of data into AI decision system]
"""

from typing import (
    Any, Dict, List, Optional, Tuple, Union, Type, TypeVar, Generic,
    Iterator, Sequence, Callable
)
from datetime import datetime, date, timedelta
from decimal import Decimal
from contextlib import contextmanager
import logging
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json

from sqlalchemy import (
    and_, or_, not_, func, desc, asc, text, between, extract,
    select, update, delete, insert, case
)
from sqlalchemy.orm import (
    Session, Query, joinedload, contains_eager, Load,
    aliased, subqueryload, selectinload
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.sql.expression import literal_column

# Import models
from .models import (
    Base, Customer, Booking, BookingService, BookingModification,
    PriceHistory, PriceExperiment, CompetitorPrice,
    MarketingCampaign, AdPerformance, AttributionEvent,
    Franchisee, FranchiseePerformance, Driver, DriverPerformance,
    UserSession, PageView, ConversionEvent,
    CallLog, CallResolution, PaymentTransaction,
    AIModelVersion, ModelPrediction, ModelPerformance,
    BookingStatus, ServiceType, PaymentStatus, FranchiseeRating,
    CallType, CallResolution as CallResolutionEnum
)
from .connections import session_scope, execute_with_retry

# ============================================================================
# TYPES AND CONSTANTS
# ============================================================================

T = TypeVar('T', bound=Base)
ModelType = Type[T]

class QueryOperator(str, Enum):
    """Query filter operators."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"

@dataclass
class QueryFilter:
    """Query filter specification."""
    field: str
    operator: QueryOperator
    value: Any
    value_type: Optional[Type] = None

@dataclass
class Pagination:
    """Pagination specification."""
    page: int = 1
    per_page: int = 50
    max_per_page: int = 1000

@dataclass
class QueryOptions:
    """Query execution options."""
    filters: List[QueryFilter] = field(default_factory=list)
    sort_by: List[Tuple[str, SortOrder]] = field(default_factory=list)
    pagination: Optional[Pagination] = None
    eager_load: List[str] = field(default_factory=list)
    distinct: bool = False
    for_update: bool = False
    cache_key: Optional[str] = None
    cache_ttl: int = 300  # seconds

# ============================================================================
# BASE REPOSITORY
# ============================================================================

class BaseRepository(Generic[T]):
    """
    Base repository implementing common CRUD operations.
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    
    def __init__(self, model_class: ModelType, session: Optional[Session] = None):
        self.model_class = model_class
        self.logger = logging.getLogger(f"repository.{model_class.__name__.lower()}")
        self._session = session
    
    @property
    def session(self) -> Session:
        """Get database session."""
        if self._session is None:
            from .connections import get_session
            return get_session()
        return self._session
    
    # ==================== CRUD OPERATIONS ====================
    
    def get(self, id: Any, **kwargs) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            id: Entity ID
            **kwargs: Additional filter criteria
            
        Returns:
            Entity or None if not found
        """
        try:
            query = self.session.query(self.model_class).filter_by(id=id)
            
            # Apply additional filters
            for key, value in kwargs.items():
                query = query.filter(getattr(self.model_class, key) == value)
            
            return query.first()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID: {str(e)}")
            raise
    
    def get_by(self, **kwargs) -> Optional[T]:
        """
        Get single entity by criteria.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            Entity or None if not found
        """
        try:
            return self.session.query(self.model_class).filter_by(**kwargs).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by criteria: {str(e)}")
            raise
    
    def get_many(self, ids: List[Any]) -> List[T]:
        """
        Get multiple entities by IDs.
        
        Args:
            ids: List of entity IDs
            
        Returns:
            List of entities
        """
        try:
            if not ids:
                return []
            return self.session.query(self.model_class).filter(
                self.model_class.id.in_(ids)
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting multiple {self.model_class.__name__}: {str(e)}")
            raise
    
    def find(self, options: Optional[QueryOptions] = None) -> List[T]:
        """
        Find entities with query options.
        
        Args:
            options: Query options
            
        Returns:
            List of entities
        """
        query = self._build_query(options)
        return self._execute_query(query, options)
    
    def find_one(self, options: Optional[QueryOptions] = None) -> Optional[T]:
        """
        Find single entity with query options.
        
        Args:
            options: Query options
            
        Returns:
            Entity or None
        """
        if options:
            options.pagination = Pagination(page=1, per_page=1)
        else:
            options = QueryOptions(pagination=Pagination(page=1, per_page=1))
        
        results = self.find(options)
        return results[0] if results else None
    
    def count(self, options: Optional[QueryOptions] = None) -> int:
        """
        Count entities matching criteria.
        
        Args:
            options: Query options
            
        Returns:
            Count of entities
        """
        try:
            query = self._build_query(options)
            
            # Remove eager loading for count
            query = query.with_entities(func.count()).select_from(self.model_class)
            
            # Apply filters
            query = self._apply_filters(query, options.filters if options else [])
            
            return query.scalar() or 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting {self.model_class.__name__}: {str(e)}")
            raise
    
    def exists(self, **kwargs) -> bool:
        """
        Check if entity exists.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            True if exists, False otherwise
        """
        try:
            return self.session.query(
                self.session.query(self.model_class).filter_by(**kwargs).exists()
            ).scalar()
        except SQLAlchemyError as e:
            self.logger.error(f"Error checking existence of {self.model_class.__name__}: {str(e)}")
            raise
    
    def create(self, entity: T, flush: bool = False) -> T:
        """
        Create new entity.
        
        Args:
            entity: Entity to create
            flush: If True, flush session
            
        Returns:
            Created entity
        """
        try:
            self.session.add(entity)
            if flush:
                self.session.flush()
            
            self.logger.debug(
                f"Created {self.model_class.__name__}",
                extra={"entity_id": getattr(entity, 'id', None)}
            )
            
            return entity
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error creating {self.model_class.__name__}: {str(e)}")
            self.session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating {self.model_class.__name__}: {str(e)}")
            raise
    
    def create_many(self, entities: List[T], flush: bool = False) -> List[T]:
        """
        Create multiple entities.
        
        Args:
            entities: List of entities to create
            flush: If True, flush session
            
        Returns:
            Created entities
        """
        try:
            self.session.add_all(entities)
            if flush:
                self.session.flush()
            
            self.logger.debug(
                f"Created {len(entities)} {self.model_class.__name__} entities"
            )
            
            return entities
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error creating multiple {self.model_class.__name__}: {str(e)}")
            self.session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating multiple {self.model_class.__name__}: {str(e)}")
            raise
    
    def update(self, entity: T, data: Dict[str, Any], flush: bool = False) -> T:
        """
        Update entity with data.
        
        Args:
            entity: Entity to update
            data: Data to update
            flush: If True, flush session
            
        Returns:
            Updated entity
        """
        try:
            for key, value in data.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            if flush:
                self.session.flush()
            
            self.logger.debug(
                f"Updated {self.model_class.__name__}",
                extra={"entity_id": getattr(entity, 'id', None)}
            )
            
            return entity
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error updating {self.model_class.__name__}: {str(e)}")
            self.session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating {self.model_class.__name__}: {str(e)}")
            raise
    
    def update_by_id(self, id: Any, data: Dict[str, Any], flush: bool = False) -> Optional[T]:
        """
        Update entity by ID.
        
        Args:
            id: Entity ID
            data: Data to update
            flush: If True, flush session
            
        Returns:
            Updated entity or None if not found
        """
        entity = self.get(id)
        if entity:
            return self.update(entity, data, flush)
        return None
    
    def update_many(self, filter_criteria: Dict[str, Any], data: Dict[str, Any]) -> int:
        """
        Update multiple entities.
        
        Args:
            filter_criteria: Filter criteria
            data: Data to update
            
        Returns:
            Number of entities updated
        """
        try:
            result = self.session.query(self.model_class).filter_by(**filter_criteria).update(
                data,
                synchronize_session='fetch'
            )
            
            self.logger.debug(
                f"Updated {result} {self.model_class.__name__} entities"
            )
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating multiple {self.model_class.__name__}: {str(e)}")
            raise
    
    def delete(self, entity: T, flush: bool = False) -> bool:
        """
        Delete entity.
        
        Args:
            entity: Entity to delete
            flush: If True, flush session
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            self.session.delete(entity)
            if flush:
                self.session.flush()
            
            self.logger.debug(
                f"Deleted {self.model_class.__name__}",
                extra={"entity_id": getattr(entity, 'id', None)}
            )
            
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting {self.model_class.__name__}: {str(e)}")
            raise
    
    def delete_by_id(self, id: Any, flush: bool = False) -> bool:
        """
        Delete entity by ID.
        
        Args:
            id: Entity ID
            flush: If True, flush session
            
        Returns:
            True if deleted, False otherwise
        """
        entity = self.get(id)
        if entity:
            return self.delete(entity, flush)
        return False
    
    def delete_many(self, filter_criteria: Dict[str, Any]) -> int:
        """
        Delete multiple entities.
        
        Args:
            filter_criteria: Filter criteria
            
        Returns:
            Number of entities deleted
        """
        try:
            result = self.session.query(self.model_class).filter_by(**filter_criteria).delete(
                synchronize_session='fetch'
            )
            
            self.logger.debug(
                f"Deleted {result} {self.model_class.__name__} entities"
            )
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting multiple {self.model_class.__name__}: {str(e)}")
            raise
    
    # ==================== QUERY BUILDING ====================
    
    def _build_query(self, options: Optional[QueryOptions] = None) -> Query:
        """
        Build SQLAlchemy query from options.
        
        Args:
            options: Query options
            
        Returns:
            SQLAlchemy query
        """
        query = self.session.query(self.model_class)
        
        if not options:
            return query
        
        # Apply eager loading
        query = self._apply_eager_loading(query, options.eager_load)
        
        # Apply filters
        query = self._apply_filters(query, options.filters)
        
        # Apply sorting
        query = self._apply_sorting(query, options.sort_by)
        
        # Apply distinct
        if options.distinct:
            query = query.distinct()
        
        # Apply for update
        if options.for_update:
            query = query.with_for_update()
        
        return query
    
    def _apply_filters(self, query: Query, filters: List[QueryFilter]) -> Query:
        """
        Apply filters to query.
        
        Args:
            query: SQLAlchemy query
            filters: List of filters
            
        Returns:
            Modified query
        """
        if not filters:
            return query
        
        filter_conditions = []
        
        for filter_spec in filters:
            condition = self._build_filter_condition(filter_spec)
            if condition is not None:
                filter_conditions.append(condition)
        
        if filter_conditions:
            query = query.filter(and_(*filter_conditions))
        
        return query
    
    def _build_filter_condition(self, filter_spec: QueryFilter):
        """
        Build SQLAlchemy filter condition.
        
        Args:
            filter_spec: Filter specification
            
        Returns:
            SQLAlchemy filter condition or None
        """
        field = getattr(self.model_class, filter_spec.field, None)
        if field is None:
            self.logger.warning(f"Invalid filter field: {filter_spec.field}")
            return None
        
        operator = filter_spec.operator
        value = filter_spec.value
        
        # Handle special operators
        if operator == QueryOperator.EQUALS:
            return field == value
        elif operator == QueryOperator.NOT_EQUALS:
            return field != value
        elif operator == QueryOperator.GREATER_THAN:
            return field > value
        elif operator == QueryOperator.GREATER_EQUAL:
            return field >= value
        elif operator == QueryOperator.LESS_THAN:
            return field < value
        elif operator == QueryOperator.LESS_EQUAL:
            return field <= value
        elif operator == QueryOperator.IN:
            return field.in_(value if isinstance(value, list) else [value])
        elif operator == QueryOperator.NOT_IN:
            return field.notin_(value if isinstance(value, list) else [value])
        elif operator == QueryOperator.LIKE:
            return field.like(value)
        elif operator == QueryOperator.ILIKE:
            return field.ilike(value)
        elif operator == QueryOperator.BETWEEN:
            if isinstance(value, tuple) and len(value) == 2:
                return field.between(value[0], value[1])
            return None
        elif operator == QueryOperator.IS_NULL:
            return field.is_(None)
        elif operator == QueryOperator.IS_NOT_NULL:
            return field.is_not(None)
        
        return None
    
    def _apply_eager_loading(self, query: Query, eager_load: List[str]) -> Query:
        """
        Apply eager loading to query.
        
        Args:
            query: SQLAlchemy query
            eager_load: List of relationships to eager load
            
        Returns:
            Modified query
        """
        if not eager_load:
            return query
        
        for relationship in eager_load:
            if hasattr(self.model_class, relationship):
                query = query.options(joinedload(getattr(self.model_class, relationship)))
            else:
                self.logger.warning(f"Invalid eager load relationship: {relationship}")
        
        return query
    
    def _apply_sorting(self, query: Query, sort_by: List[Tuple[str, SortOrder]]) -> Query:
        """
        Apply sorting to query.
        
        Args:
            query: SQLAlchemy query
            sort_by: List of (field, order) tuples
            
        Returns:
            Modified query
        """
        if not sort_by:
            return query
        
        for field, order in sort_by:
            model_field = getattr(self.model_class, field, None)
            if model_field is None:
                self.logger.warning(f"Invalid sort field: {field}")
                continue
            
            if order == SortOrder.ASC:
                query = query.order_by(asc(model_field))
            elif order == SortOrder.DESC:
                query = query.order_by(desc(model_field))
        
        return query
    
    def _execute_query(self, query: Query, options: Optional[QueryOptions] = None) -> List[T]:
        """
        Execute query with pagination.
        
        Args:
            query: SQLAlchemy query
            options: Query options
            
        Returns:
            List of entities
        """
        try:
            # Apply pagination
            if options and options.pagination:
                offset = (options.pagination.page - 1) * options.pagination.per_page
                limit = min(options.pagination.per_page, options.pagination.max_per_page)
                query = query.offset(offset).limit(limit)
            
            # Execute query
            return query.all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    # ==================== UTILITY METHODS ====================
    
    def paginate_query(self, query: Query, page: int = 1, per_page: int = 50) -> Query:
        """
        Apply pagination to query.
        
        Args:
            query: SQLAlchemy query
            page: Page number (1-indexed)
            per_page: Items per page
            
        Returns:
            Paginated query
        """
        offset = (page - 1) * per_page
        return query.offset(offset).limit(per_page)
    
    def filter_by_date_range(
        self,
        query: Query,
        date_field: str,
        start_date: date,
        end_date: date
    ) -> Query:
        """
        Filter query by date range.
        
        Args:
            query: SQLAlchemy query
            date_field: Name of date field
            start_date: Start date
            end_date: End date
            
        Returns:
            Filtered query
        """
        field = getattr(self.model_class, date_field, None)
        if field is None:
            self.logger.warning(f"Invalid date field: {date_field}")
            return query
        
        return query.filter(between(field, start_date, end_date))
    
    def apply_filters(self, query: Query, **filters) -> Query:
        """
        Apply multiple filters to query.
        
        Args:
            query: SQLAlchemy query
            **filters: Filter criteria
            
        Returns:
            Filtered query
        """
        filter_conditions = []
        
        for field, value in filters.items():
            if value is None:
                continue
            
            model_field = getattr(self.model_class, field, None)
            if model_field is None:
                self.logger.warning(f"Invalid filter field: {field}")
                continue
            
            if isinstance(value, (list, tuple)):
                filter_conditions.append(model_field.in_(value))
            else:
                filter_conditions.append(model_field == value)
        
        if filter_conditions:
            query = query.filter(and_(*filter_conditions))
        
        return query
    
    def bulk_upsert(
        self,
        data: List[Dict[str, Any]],
        conflict_fields: List[str],
        update_fields: List[str]
    ) -> int:
        """
        Bulk upsert (insert or update) records.
        
        Args:
            data: List of data dictionaries
            conflict_fields: Fields to detect conflicts
            update_fields: Fields to update on conflict
            
        Returns:
            Number of affected rows
        """
        if not data:
            return 0
        
        try:
            # Build insert statement
            stmt = insert(self.model_class).values(data)
            
            # Build update on conflict
            update_dict = {
                field: getattr(stmt.excluded, field)
                for field in update_fields
                if hasattr(self.model_class, field)
            }
            
            # Execute upsert
            result = self.session.execute(
                stmt.on_duplicate_key_update(**update_dict)
            )
            
            self.logger.debug(
                f"Bulk upserted {len(data)} {self.model_class.__name__} records"
            )
            
            return result.rowcount
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error in bulk upsert: {str(e)}")
            raise

# ============================================================================
# SPECIALIZED REPOSITORIES
# ============================================================================

class BookingRepository(BaseRepository[Booking]):
    """
    Booking repository with business-specific queries.
    
    Reference: [About The Role: Live booking data]
    """
    
    def get_active_bookings(self, airport_code: Optional[str] = None) -> List[Booking]:
        """
        Get currently active bookings.
        
        Args:
            airport_code: Optional airport filter
            
        Returns:
            List of active bookings
        """
        now = datetime.now()
        query = self.session.query(Booking).filter(
            Booking.status.in_([BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN]),
            Booking.check_in <= now,
            Booking.check_out >= now
        )
        
        if airport_code:
            query = query.filter(Booking.airport_code == airport_code)
        
        return query.all()
    
    def get_bookings_by_date_range(
        self,
        airport_code: str,
        start_date: date,
        end_date: date,
        statuses: Optional[List[BookingStatus]] = None
    ) -> List[Booking]:
        """
        Get bookings for an airport within date range.
        
        Args:
            airport_code: Airport code
            start_date: Start date
            end_date: End date
            statuses: Optional status filter
            
        Returns:
            List of bookings
        """
        query = self.session.query(Booking).filter(
            Booking.airport_code == airport_code,
            func.date(Booking.check_in) >= start_date,
            func.date(Booking.check_out) <= end_date
        )
        
        if statuses:
            query = query.filter(Booking.status.in_(statuses))
        
        return query.order_by(Booking.check_in).all()
    
    def get_customer_bookings(
        self,
        customer_id: int,
        days: int = 90,
        include_cancelled: bool = False
    ) -> List[Booking]:
        """
        Get customer's recent bookings.
        
        Args:
            customer_id: Customer ID
            days: Number of days to look back
            include_cancelled: Include cancelled bookings
            
        Returns:
            List of bookings
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = self.session.query(Booking).filter(
            Booking.customer_id == customer_id,
            Booking.created_at >= cutoff_date
        )
        
        if not include_cancelled:
            query = query.filter(Booking.status != BookingStatus.CANCELLED)
        
        return query.order_by(desc(Booking.check_in)).all()
    
    def get_booking_stats_by_airport(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Get booking statistics by airport.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of statistics by airport
        """
        query = self.session.query(
            Booking.airport_code,
            func.count(Booking.id).label('total_bookings'),
            func.sum(Booking.total_price).label('total_revenue'),
            func.avg(Booking.total_price).label('avg_booking_value'),
            func.sum(case(
                (Booking.status == BookingStatus.CANCELLED, 1),
                else_=0
            )).label('cancelled_count')
        ).filter(
            func.date(Booking.check_in) >= start_date,
            func.date(Booking.check_in) <= end_date
        ).group_by(
            Booking.airport_code
        ).order_by(
            desc('total_revenue')
        )
        
        results = []
        for row in query.all():
            results.append({
                'airport_code': row.airport_code,
                'total_bookings': row.total_bookings or 0,
                'total_revenue': float(row.total_revenue or 0),
                'avg_booking_value': float(row.avg_booking_value or 0),
                'cancellation_rate': (
                    (row.cancelled_count or 0) / (row.total_bookings or 1)
                ) * 100
            })
        
        return results
    
    def find_late_checkouts(self, threshold_minutes: int = 30) -> List[Booking]:
        """
        Find bookings with late checkouts.
        
        Args:
            threshold_minutes: Late threshold in minutes
            
        Returns:
            List of late checkout bookings
        """
        query = self.session.query(Booking).filter(
            Booking.actual_check_out.is_not(None),
            Booking.check_out.is_not(None),
            func.timestampdiff(
                text('MINUTE'),
                Booking.check_out,
                Booking.actual_check_out
            ) > threshold_minutes,
            Booking.late_fee == 0  # Only those not charged yet
        )
        
        return query.all()


class PricingRepository(BaseRepository[PriceHistory]):
    """
    Pricing data repository for AI models.
    
    Reference: [Core Responsibilities: 1. Commercial & Pricing AI]
    """
    
    def get_price_history(
        self,
        airport_code: str,
        service_type: ServiceType,
        start_date: date,
        end_date: date
    ) -> List[PriceHistory]:
        """
        Get price history for forecasting.
        
        Args:
            airport_code: Airport code
            service_type: Service type
            start_date: Start date
            end_date: End date
            
        Returns:
            List of price history records
        """
        return self.session.query(PriceHistory).filter(
            PriceHistory.airport_code == airport_code,
            PriceHistory.service_type == service_type,
            PriceHistory.date >= start_date,
            PriceHistory.date <= end_date
        ).order_by(
            PriceHistory.date,
            PriceHistory.hour
        ).all()
    
    def get_competitor_prices(
        self,
        airport_code: str,
        service_type: ServiceType,
        days: int = 7
    ) -> List[CompetitorPrice]:
        """
        Get recent competitor prices.
        
        Args:
            airport_code: Airport code
            service_type: Service type
            days: Number of days to look back
            
        Returns:
            List of competitor prices
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        return self.session.query(CompetitorPrice).filter(
            CompetitorPrice.airport_code == airport_code,
            CompetitorPrice.service_type == service_type,
            CompetitorPrice.date >= cutoff_date
        ).order_by(
            desc(CompetitorPrice.date),
            CompetitorPrice.competitor_name
        ).all()
    
    def get_price_elasticity_data(
        self,
        airport_code: str,
        days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Get data for price elasticity analysis.
        
        Args:
            airport_code: Airport code
            days: Number of days to analyze
            
        Returns:
            List of price elasticity data points
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        query = self.session.query(
            PriceHistory.date,
            PriceHistory.base_price,
            PriceHistory.final_price,
            PriceHistory.bookings_count,
            PriceHistory.competitor_avg_price,
            PriceHistory.demand_factor
        ).filter(
            PriceHistory.airport_code == airport_code,
            PriceHistory.date >= cutoff_date,
            PriceHistory.bookings_count > 0
        ).order_by(
            PriceHistory.date
        )
        
        results = []
        for row in query.all():
            results.append({
                'date': row.date,
                'base_price': float(row.base_price),
                'final_price': float(row.final_price),
                'bookings_count': row.bookings_count,
                'competitor_avg_price': float(row.competitor_avg_price) if row.competitor_avg_price else None,
                'demand_factor': row.demand_factor,
                'price_change_pct': (
                    (float(row.final_price) - float(row.base_price)) / float(row.base_price)
                ) * 100 if row.base_price > 0 else 0
            })
        
        return results


class MarketingRepository(BaseRepository[MarketingCampaign]):
    """
    Marketing data repository for attribution and optimization.
    
    Reference: [Core Responsibilities: 2. Marketing & Multi-Channel Budget Optimisation]
    """
    
    def get_campaign_performance(
        self,
        start_date: date,
        end_date: date,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get campaign performance metrics.
        
        Args:
            start_date: Start date
            end_date: End date
            channel: Optional channel filter
            
        Returns:
            List of campaign performance data
        """
        query = self.session.query(
            MarketingCampaign.id,
            MarketingCampaign.name,
            MarketingCampaign.channel,
            MarketingCampaign.total_budget,
            MarketingCampaign.spent_to_date,
            MarketingCampaign.revenue,
            MarketingCampaign.conversions,
            func.coalesce(
                MarketingCampaign.spent_to_date / nullif(MarketingCampaign.conversions, 0),
                0
            ).label('cpa'),
            func.coalesce(
                MarketingCampaign.revenue / nullif(MarketingCampaign.spent_to_date, 0),
                0
            ).label('roas')
        ).filter(
            MarketingCampaign.start_date <= end_date,
            MarketingCampaign.end_date >= start_date,
            MarketingCampaign.status == 'active'
        )
        
        if channel:
            query = query.filter(MarketingCampaign.channel == channel)
        
        results = []
        for row in query.all():
            results.append({
                'campaign_id': row.id,
                'campaign_name': row.name,
                'channel': row.channel,
                'total_budget': float(row.total_budget),
                'spent_to_date': float(row.spent_to_date),
                'revenue': float(row.revenue),
                'conversions': row.conversions,
                'cpa': float(row.cpa),
                'roas': float(row.roas),
                'remaining_budget': float(row.total_budget - row.spent_to_date),
                'budget_utilization': (
                    float(row.spent_to_date) / float(row.total_budget) * 100
                ) if row.total_budget > 0 else 0
            })
        
        return results
    
    def get_attribution_data(
        self,
        customer_id: int,
        days: int = 90
    ) -> List[AttributionEvent]:
        """
        Get attribution events for a customer.
        
        Args:
            customer_id: Customer ID
            days: Number of days to look back
            
        Returns:
            List of attribution events
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return self.session.query(AttributionEvent).filter(
            AttributionEvent.customer_id == customer_id,
            AttributionEvent.event_timestamp >= cutoff_date
        ).order_by(
            desc(AttributionEvent.event_timestamp)
        ).all()
    
    def get_multi_touch_attribution(
        self,
        booking_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get multi-touch attribution for a booking.
        
        Args:
            booking_id: Booking ID
            
        Returns:
            List of attribution touches
        """
        query = self.session.query(
            AttributionEvent.event_type,
            AttributionEvent.channel,
            AttributionEvent.event_timestamp,
            AttributionEvent.utm_source,
            AttributionEvent.utm_medium,
            AttributionEvent.utm_campaign,
            MarketingCampaign.name.label('campaign_name')
        ).outerjoin(
            MarketingCampaign,
            AttributionEvent.campaign_id == MarketingCampaign.id
        ).filter(
            AttributionEvent.booking_id == booking_id
        ).order_by(
            AttributionEvent.event_timestamp
        )
        
        results = []
        for row in query.all():
            results.append({
                'event_type': row.event_type,
                'channel': row.channel,
                'event_timestamp': row.event_timestamp,
                'utm_source': row.utm_source,
                'utm_medium': row.utm_medium,
                'utm_campaign': row.utm_campaign,
                'campaign_name': row.campaign_name,
                'touch_number': len(results) + 1
            })
        
        return results


class FranchiseeRepository(BaseRepository[Franchisee]):
    """
    Franchisee repository for performance scoring and risk detection.
    
    Reference: [Core Responsibilities: 4. Franchisee Performance, Risk & Fraud Detection]
    """
    
    def get_franchisee_performance(
        self,
        franchisee_id: int,
        days: int = 30
    ) -> List[FranchiseePerformance]:
        """
        Get franchisee performance history.
        
        Args:
            franchisee_id: Franchisee ID
            days: Number of days to look back
            
        Returns:
            List of performance records
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        return self.session.query(FranchiseePerformance).filter(
            FranchiseePerformance.franchisee_id == franchisee_id,
            FranchiseePerformance.date >= cutoff_date
        ).order_by(
            FranchiseePerformance.date
        ).all()
    
    def calculate_fps_score(
        self,
        franchisee_id: int,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Calculate Franchisee Performance Score.
        
        Args:
            franchisee_id: Franchisee ID
            start_date: Start date
            end_date: End date
            
        Returns:
            FPS score and components
        """
        # Get performance data
        performances = self.session.query(FranchiseePerformance).filter(
            FranchiseePerformance.franchisee_id == franchisee_id,
            FranchiseePerformance.date >= start_date,
            FranchiseePerformance.date <= end_date
        ).all()
        
        if not performances:
            return {
                'fps_score': 0,
                'data_points': 0,
                'components': {}
            }
        
        # Calculate weighted scores
        total_weight = 0
        weighted_score = 0
        components = {}
        
        for perf in performances:
            if perf.fps_score is not None:
                # Use daily FPS if available
                weighted_score += perf.fps_score
                total_weight += 1
            else:
                # Calculate from components
                # In production, this would use actual weights from constants
                pass
        
        fps_score = weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            'fps_score': round(fps_score, 2),
            'data_points': len(performances),
            'components': components
        }
    
    def detect_fraud_anomalies(
        self,
        franchisee_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Detect fraud anomalies for a franchisee.
        
        Args:
            franchisee_id: Franchisee ID
            days: Number of days to analyze
            
        Returns:
            List of detected anomalies
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        # Get recent performance
        performances = self.session.query(FranchiseePerformance).filter(
            FranchiseePerformance.franchisee_id == franchisee_id,
            FranchiseePerformance.date >= cutoff_date
        ).order_by(
            FranchiseePerformance.date
        ).all()
        
        anomalies = []
        
        if len(performances) < 2:
            return anomalies
        
        # Analyze trends
        refund_rates = [p.refund_amount / max(p.total_revenue, 1) for p in performances if p.total_revenue > 0]
        cancellation_rates = [p.cancellation_rate for p in performances]
        
        # Check for sudden spikes
        if len(refund_rates) >= 3:
            recent_avg = sum(refund_rates[-3:]) / 3
            historical_avg = sum(refund_rates[:-3]) / max(len(refund_rates) - 3, 1)
            
            if recent_avg > historical_avg * 2:  # 100% increase
                anomalies.append({
                    'type': 'refund_spike',
                    'message': f'Refund rate increased from {historical_avg:.1%} to {recent_avg:.1%}',
                    'severity': 'high'
                })
        
        if len(cancellation_rates) >= 3:
            recent_avg = sum(cancellation_rates[-3:]) / 3
            historical_avg = sum(cancellation_rates[:-3]) / max(len(cancellation_rates) - 3, 1)
            
            if recent_avg > historical_avg * 1.5:  # 50% increase
                anomalies.append({
                    'type': 'cancellation_spike',
                    'message': f'Cancellation rate increased from {historical_avg:.1%} to {recent_avg:.1%}',
                    'severity': 'medium'
                })
        
        return anomalies


class CustomerRepository(BaseRepository[Customer]):
    """
    Customer repository for segmentation and personalization.
    
    Reference: [Core Responsibilities: 5. Website Behaviour, Conversion & Personalisation AI]
    """
    
    def get_customer_segments(self) -> Dict[str, List[Customer]]:
        """
        Get customers grouped by segments.
        
        Returns:
            Dict of segment name to list of customers
        """
        segments = {}
        
        # Get customers with segments
        customers = self.session.query(Customer).filter(
            Customer.customer_segment.is_not(None)
        ).all()
        
        for customer in customers:
            segment = customer.customer_segment
            if segment not in segments:
                segments[segment] = []
            segments[segment].append(customer)
        
        return segments
    
    def calculate_customer_lifetime_value(
        self,
        customer_id: int,
        months: int = 12
    ) -> Dict[str, Any]:
        """
        Calculate customer lifetime value.
        
        Args:
            customer_id: Customer ID
            months: Number of months to analyze
            
        Returns:
            CLV metrics
        """
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        # Get bookings
        bookings = self.session.query(Booking).filter(
            Booking.customer_id == customer_id,
            Booking.created_at >= cutoff_date,
            Booking.status != BookingStatus.CANCELLED
        ).all()
        
        if not bookings:
            return {
                'clv': 0,
                'total_spent': 0,
                'booking_count': 0,
                'avg_order_value': 0,
                'purchase_frequency': 0
            }
        
        total_spent = sum(float(b.total_price) for b in bookings)
        booking_count = len(bookings)
        
        # Calculate metrics
        avg_order_value = total_spent / booking_count
        purchase_frequency = booking_count / months
        
        # Simple CLV calculation
        clv = total_spent * purchase_frequency * 12  # Project to annual
        
        return {
            'clv': round(clv, 2),
            'total_spent': round(total_spent, 2),
            'booking_count': booking_count,
            'avg_order_value': round(avg_order_value, 2),
            'purchase_frequency': round(purchase_frequency, 2)
        }
    
    def get_customer_behavior_patterns(
        self,
        customer_id: int
    ) -> Dict[str, Any]:
        """
        Analyze customer behavior patterns.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Behavior patterns
        """
        # Get recent bookings
        bookings = self.session.query(Booking).filter(
            Booking.customer_id == customer_id,
            Booking.status != BookingStatus.CANCELLED
        ).order_by(
            desc(Booking.check_in)
        ).limit(10).all()
        
        if not bookings:
            return {
                'booking_count': 0,
                'patterns': {}
            }
        
        # Analyze patterns
        airports = {}
        service_types = {}
        booking_days = {}
        lead_times = []
        
        for booking in bookings:
            # Airport preference
            airports[booking.airport_code] = airports.get(booking.airport_code, 0) + 1
            
            # Service type preference
            for service in booking.services:
                service_type = service.service_type.value
                service_types[service_type] = service_types.get(service_type, 0) + 1
            
            # Day of week preference
            day_name = booking.check_in.strftime('%A')
            booking_days[day_name] = booking_days.get(day_name, 0) + 1
            
            # Lead time (days between booking and check-in)
            lead_time = (booking.check_in - booking.created_at).days
            lead_times.append(lead_time)
        
        # Calculate averages
        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
        
        return {
            'booking_count': len(bookings),
            'patterns': {
                'preferred_airports': sorted(
                    airports.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                'preferred_services': sorted(
                    service_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                'booking_days': booking_days,
                'avg_lead_time_days': round(avg_lead_time, 1)
            }
        }


class AIPredictionRepository(BaseRepository[ModelPrediction]):
    """
    AI prediction repository for model performance tracking.
    
    Reference: [Technical Stack: AI / ML: Python, forecasting models]
    """
    
    def get_model_predictions(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        entity_type: Optional[str] = None
    ) -> List[ModelPrediction]:
        """
        Get model predictions for analysis.
        
        Args:
            model_type: Model type
            start_date: Start date
            end_date: End date
            entity_type: Optional entity type filter
            
        Returns:
            List of predictions
        """
        query = self.session.query(ModelPrediction).filter(
            ModelPrediction.prediction_type == model_type,
            ModelPrediction.prediction_timestamp >= start_date,
            ModelPrediction.prediction_timestamp <= end_date
        )
        
        if entity_type:
            query = query.filter(ModelPrediction.entity_type == entity_type)
        
        return query.order_by(
            ModelPrediction.prediction_timestamp
        ).all()
    
    def calculate_model_accuracy(
        self,
        model_version_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Calculate model accuracy metrics.
        
        Args:
            model_version_id: Model version ID
            start_date: Start date
            end_date: End date
            
        Returns:
            Accuracy metrics
        """
        predictions = self.session.query(ModelPrediction).filter(
            ModelPrediction.model_version_id == model_version_id,
            ModelPrediction.prediction_timestamp >= start_date,
            ModelPrediction.prediction_timestamp <= end_date,
            ModelPrediction.actual_value.is_not(None),
            ModelPrediction.prediction_error.is_not(None)
        ).all()
        
        if not predictions:
            return {
                'sample_size': 0,
                'mae': 0,
                'rmse': 0,
                'accuracy': 0
            }
        
        # Calculate metrics
        errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = (sum(e ** 2 for e in errors) / len(errors)) ** 0.5
        
        # Simple accuracy (for classification)
        correct = sum(1 for p in predictions if p.prediction_error == 0)
        accuracy = correct / len(predictions) if predictions else 0
        
        return {
            'sample_size': len(predictions),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'accuracy': round(accuracy, 4),
            'confidence_avg': round(
                sum(p.confidence_score for p in predictions if p.confidence_score) /
                len([p for p in predictions if p.confidence_score]),
                4
            ) if any(p.confidence_score for p in predictions) else 0
        }
    
    def detect_prediction_drift(
        self,
        model_version_id: int,
        window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Detect prediction drift.
        
        Args:
            model_version_id: Model version ID
            window_days: Analysis window in days
            
        Returns:
            Drift detection results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)
        comparison_start = start_date - timedelta(days=window_days)
        
        # Get recent predictions
        recent_preds = self.get_model_predictions(
            model_type=self._get_model_type(model_version_id),
            start_date=start_date,
            end_date=end_date
        )
        
        # Get comparison predictions
        comp_preds = self.get_model_predictions(
            model_type=self._get_model_type(model_version_id),
            start_date=comparison_start,
            end_date=start_date
        )
        
        if not recent_preds or not comp_preds:
            return {
                'has_drift': False,
                'confidence': 0,
                'message': 'Insufficient data for drift detection'
            }
        
        # Calculate drift metrics (simplified)
        recent_errors = [p.prediction_error for p in recent_preds if p.prediction_error]
        comp_errors = [p.prediction_error for p in comp_preds if p.prediction_error]
        
        if not recent_errors or not comp_errors:
            return {
                'has_drift': False,
                'confidence': 0,
                'message': 'No error data available'
            }
        
        recent_avg = sum(recent_errors) / len(recent_errors)
        comp_avg = sum(comp_errors) / len(comp_errors)
        
        # Simple drift detection
        drift_ratio = abs(recent_avg - comp_avg) / max(abs(comp_avg), 0.001)
        has_drift = drift_ratio > 0.2  # 20% change threshold
        
        return {
            'has_drift': has_drift,
            'drift_ratio': round(drift_ratio, 4),
            'recent_avg_error': round(recent_avg, 4),
            'comparison_avg_error': round(comp_avg, 4),
            'confidence': min(drift_ratio, 1.0),
            'message': f'Error changed by {drift_ratio:.1%}'
        }
    
    def _get_model_type(self, model_version_id: int) -> str:
        """Get model type from version ID."""
        from .models import AIModelVersion
        model = self.session.query(AIModelVersion).get(model_version_id)
        return model.model_type.value if model else 'unknown'

# ============================================================================
# REPOSITORY FACTORY
# ============================================================================

class RepositoryFactory:
    """
    Factory for creating repository instances.
    
    Reference: [Required Experience: Production engineering role]
    """
    
    _repositories = {}
    
    @classmethod
    def get_repository(cls, repository_class: Type[BaseRepository], session: Optional[Session] = None):
        """
        Get repository instance.
        
        Args:
            repository_class: Repository class
            session: Optional database session
            
        Returns:
            Repository instance
        """
        key = (repository_class, session)
        
        if key not in cls._repositories:
            cls._repositories[key] = repository_class(session)
        
        return cls._repositories[key]
    
    @classmethod
    def clear_cache(cls):
        """Clear repository cache."""
        cls._repositories.clear()

# Convenience functions
def get_repository(repository_class: Type[BaseRepository], session: Optional[Session] = None):
    """
    Get repository instance.
    
    Args:
        repository_class: Repository class
        session: Optional database session
        
    Returns:
        Repository instance
    """
    return RepositoryFactory.get_repository(repository_class, session)

# ============================================================================
# QUERY UTILITIES
# ============================================================================

def paginate_query(query: Query, page: int = 1, per_page: int = 50) -> Query:
    """
    Apply pagination to any query.
    
    Args:
        query: SQLAlchemy query
        page: Page number (1-indexed)
        per_page: Items per page
        
    Returns:
        Paginated query
    """
    return query.offset((page - 1) * per_page).limit(per_page)

def filter_by_date_range(
    query: Query,
    date_field: str,
    start_date: date,
    end_date: date,
    model_class: Optional[Type[Base]] = None
) -> Query:
    """
    Filter query by date range.
    
    Args:
        query: SQLAlchemy query
        date_field: Name of date field
        start_date: Start date
        end_date: End date
        model_class: Model class (if not in query)
        
    Returns:
        Filtered query
    """
    if model_class:
        field = getattr(model_class, date_field)
    else:
        # Extract from query
        field = query.column_descriptions[0]['entity'].columns[date_field]
    
    return query.filter(between(field, start_date, end_date))

def apply_filters(query: Query, filters: Dict[str, Any], model_class: Type[Base]) -> Query:
    """
    Apply multiple filters to query.
    
    Args:
        query: SQLAlchemy query
        filters: Filter dictionary
        model_class: Model class
        
    Returns:
        Filtered query
    """
    for field, value in filters.items():
        if value is None:
            continue
        
        model_field = getattr(model_class, field, None)
        if model_field is None:
            continue
        
        if isinstance(value, (list, tuple)):
            query = query.filter(model_field.in_(value))
        else:
            query = query.filter(model_field == value)
    
    return query

# ============================================================================
# REPOSITORY CONTEXT MANAGER
# ============================================================================

@contextmanager
def repository_scope(repository_class: Type[BaseRepository]):
    """
    Context manager for repository operations.
    
    Usage:
        with repository_scope(BookingRepository) as repo:
            bookings = repo.get_active_bookings("JFK")
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    with session_scope() as session:
        repo = repository_class(session)
        try:
            yield repo
        except Exception:
            session.rollback()
            raise

# ============================================================================
# QUERY CACHING
# ============================================================================

class QueryCache:
    """
    Simple query caching for frequently accessed data.
    
    Reference: [Required Experience: Production engineering role]
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = {}
        self.logger = logging.getLogger("repository.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis:
                cached = self.redis.get(key)
                if cached:
                    return json.loads(cached)
            
            # Fall back to local cache
            return self.local_cache.get(key)
            
        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache."""
        try:
            # Set in Redis if available
            if self.redis:
                self.redis.setex(key, ttl, json.dumps(value))
            
            # Also set in local cache
            self.local_cache[key] = value
            
        except Exception as e:
            self.logger.error(f"Cache set error: {str(e)}")
    
    def delete(self, key: str):
        """Delete value from cache."""
        try:
            if self.redis:
                self.redis.delete(key)
            if key in self.local_cache:
                del self.local_cache[key]
        except Exception as e:
            self.logger.error(f"Cache delete error: {str(e)}")
    
    def generate_key(self, repo_class: Type, method: str, *args, **kwargs) -> str:
        """Generate cache key for query."""
        key_data = {
            'repo': repo_class.__name__,
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"repo_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

# Global cache instance
_query_cache = QueryCache()

def cached_query(ttl: int = 300):
    """
    Decorator for caching repository query results.
    
    Args:
        ttl: Cache time-to-live in seconds
        
    Reference: [Required Experience: Production engineering role]
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip cache if specified
            if kwargs.get('skip_cache'):
                return func(self, *args, **kwargs)
            
            # Generate cache key
            cache_key = _query_cache.generate_key(
                type(self),
                func.__name__,
                *args,
                **{k: v for k, v in kwargs.items() if k != 'skip_cache'}
            )
            
            # Try cache
            cached_result = _query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute query
            result = func(self, *args, **kwargs)
            
            # Cache result
            _query_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_repositories() -> None:
    """
    Initialize repository system.
    
    Reference: [Required Experience: Production engineering role]
    """
    logger = logging.getLogger("repository")
    
    # Test repository connectivity
    try:
        with repository_scope(BookingRepository) as repo:
            # Simple test query
            count = repo.count()
            logger.info(
                "Repository system initialized",
                extra={
                    "booking_count": count,
                    "repository_classes": [
                        "BookingRepository",
                        "PricingRepository", 
                        "MarketingRepository",
                        "FranchiseeRepository",
                        "CustomerRepository",
                        "AIPredictionRepository"
                    ]
                }
            )
            
    except Exception as e:
        logger.error(f"Repository initialization failed: {str(e)}")
        raise

# Auto-initialize
initialize_repositories()
