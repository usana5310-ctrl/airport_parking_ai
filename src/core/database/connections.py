"""
Database Connection Management
Project: AI Commercial Platform for Airport Parking

This module manages database connections, connection pooling, and session management
for optimal performance and reliability in production environments.

Reference: [Technical Stack: MySQL / PostgreSQL]
"""

import logging
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from sqlalchemy import create_engine, event, text, exc
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)

# Import settings
from ..config.settings import get_settings

# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class DatabaseType(str, Enum):
    """Supported database types."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

class ConnectionStatus(str, Enum):
    """Connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

class PoolStatus(str, Enum):
    """Connection pool status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    ERROR = "error"

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    overflow_connections: int = 0
    connection_errors: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    last_checked: datetime = field(default_factory=datetime.now)

@dataclass
class ConnectionConfig:
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
    pool_pre_ping: bool = True
    echo: bool = False
    echo_pool: bool = False
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Generate async SQLAlchemy connection string."""
        return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

# ============================================================================
# DATABASE CONNECTION MANAGER
# ============================================================================

class DatabaseConnection:
    """
    Manages database connections with pooling, monitoring, and failover.
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for database connection."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseConnection, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize database connection."""
        if self._initialized:
            return
        
        self.logger = logging.getLogger("database.connections")
        self.settings = get_settings()
        
        # Connection state
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._scoped_session_factory: Optional[scoped_session] = None
        
        # Monitoring
        self.metrics = ConnectionMetrics()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.pool_status = PoolStatus.ERROR
        self.last_health_check = None
        self.connection_errors = []
        
        # Configuration
        self.config = self._create_connection_config()
        
        # Initialize
        self._initialize_connection()
        
        self._initialized = True
        
        self.logger.info(
            "Database connection manager initialized",
            extra={
                "host": self.config.host,
                "database": self.config.database,
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow
            }
        )
    
    def _create_connection_config(self) -> ConnectionConfig:
        """Create connection configuration from settings."""
        return ConnectionConfig(
            host=self.settings.DATABASE_HOST,
            port=self.settings.DATABASE_PORT,
            username=self.settings.DATABASE_USER,
            password=self.settings.DATABASE_PASSWORD,
            database=self.settings.DATABASE_NAME,
            pool_size=self.settings.DATABASE_POOL_SIZE,
            max_overflow=self.settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=self.settings.is_development,
            echo_pool=self.settings.is_development
        )
    
    def _initialize_connection(self) -> None:
        """Initialize database connection and engine."""
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                # Connection arguments
                connect_args={
                    'connect_timeout': 10,
                    'charset': 'utf8mb4'
                }
            )
            
            # Configure session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
            
            # Configure scoped session for thread safety
            self._scoped_session_factory = scoped_session(self._session_factory)
            
            # Set up connection event listeners
            self._setup_event_listeners()
            
            # Test connection
            self.test()
            
            self.connection_status = ConnectionStatus.CONNECTED
            self._update_pool_status()
            
            self.logger.info(
                "Database connection established",
                extra={
                    "status": self.connection_status.value,
                    "pool_status": self.pool_status.value,
                    "dialect": self.get_dialect()
                }
            )
            
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self.pool_status = PoolStatus.ERROR
            self.connection_errors.append(str(e))
            
            self.logger.error(
                f"Failed to initialize database connection: {str(e)}",
                extra={
                    "host": self.config.host,
                    "database": self.config.database,
                    "error": str(e)
                }
            )
            raise
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self._engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Log when a new connection is created."""
            self.metrics.total_connections += 1
            self.logger.debug(
                "New database connection created",
                extra={
                    "connection_id": id(dbapi_connection),
                    "total_connections": self.metrics.total_connections
                }
            )
        
        @event.listens_for(self._engine, "close")
        def close(dbapi_connection, connection_record):
            """Log when a connection is closed."""
            self.logger.debug(
                "Database connection closed",
                extra={"connection_id": id(dbapi_connection)}
            )
        
        @event.listens_for(self._engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            """Log when a connection is checked out from the pool."""
            self.metrics.active_connections += 1
            connection_record.info['checkout_time'] = time.time()
            
            self.logger.debug(
                "Connection checked out from pool",
                extra={
                    "connection_id": id(dbapi_connection),
                    "active_connections": self.metrics.active_connections
                }
            )
        
        @event.listens_for(self._engine, "checkin")
        def checkin(dbapi_connection, connection_record):
            """Log when a connection is returned to the pool."""
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            
            # Calculate connection usage time
            checkout_time = connection_record.info.get('checkout_time')
            if checkout_time:
                usage_time = time.time() - checkout_time
                self.metrics.avg_query_time_ms = (
                    (self.metrics.avg_query_time_ms * self.metrics.total_queries + usage_time * 1000) /
                    (self.metrics.total_queries + 1)
                ) if self.metrics.total_queries > 0 else usage_time * 1000
                self.metrics.max_query_time_ms = max(self.metrics.max_query_time_ms, usage_time * 1000)
                self.metrics.total_queries += 1
            
            self.logger.debug(
                "Connection returned to pool",
                extra={
                    "connection_id": id(dbapi_connection),
                    "active_connections": self.metrics.active_connections
                }
            )
        
        @event.listens_for(self._engine, "handle_error")
        def handle_error(context):
            """Log database errors."""
            self.metrics.connection_errors += 1
            self.metrics.failed_queries += 1
            
            error = str(context.original_exception) if context.original_exception else "Unknown error"
            
            self.logger.error(
                "Database error occurred",
                extra={
                    "error": error,
                    "statement": context.statement[:200] if context.statement else None,
                    "connection_errors": self.metrics.connection_errors
                }
            )
    
    def _update_pool_status(self) -> None:
        """Update connection pool status based on current metrics."""
        try:
            if not self._engine:
                self.pool_status = PoolStatus.ERROR
                return
            
            pool = self._engine.pool
            total_in_pool = pool.size() + pool.checkedin() + pool.overflow()
            utilization = pool.checkedin() / max(pool.size(), 1)
            
            if utilization > 0.9:
                self.pool_status = PoolStatus.OVERLOADED
            elif utilization > 0.7:
                self.pool_status = PoolStatus.DEGRADED
            elif self.metrics.connection_errors > 10:
                self.pool_status = PoolStatus.DEGRADED
            else:
                self.pool_status = PoolStatus.HEALTHY
                
        except Exception as e:
            self.logger.error(f"Failed to update pool status: {str(e)}")
            self.pool_status = PoolStatus.ERROR
    
    def get_engine(self) -> Engine:
        """
        Get the SQLAlchemy engine.
        
        Returns:
            Engine: SQLAlchemy engine instance
            
        Raises:
            ConnectionError: If engine is not initialized
            
        Reference: [Technical Stack: MySQL / PostgreSQL]
        """
        if self._engine is None:
            raise ConnectionError("Database engine not initialized")
        return self._engine
    
    def get_async_engine(self) -> AsyncEngine:
        """
        Get async SQLAlchemy engine.
        
        Returns:
            AsyncEngine: Async SQLAlchemy engine
            
        Reference: [Required Experience: Production engineering role]
        """
        if self._async_engine is None:
            self._initialize_async_engine()
        return self._async_engine
    
    def _initialize_async_engine(self) -> None:
        """Initialize async database engine."""
        try:
            self._async_engine = create_async_engine(
                self.config.async_connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo
            )
            
            self._async_session_factory = async_sessionmaker(
                self._async_engine,
                expire_on_commit=False,
                class_=AsyncSession
            )
            
            self.logger.info("Async database engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async engine: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            Session: New SQLAlchemy session
            
        Reference: [Required Experience: Production AI/data engineering]
        """
        if self._session_factory is None:
            raise ConnectionError("Session factory not initialized")
        
        session = self._session_factory()
        
        # Configure session
        session.info['session_id'] = str(uuid.uuid4())[:8]
        session.info['created_at'] = datetime.now()
        
        self.logger.debug(
            "New database session created",
            extra={"session_id": session.info['session_id']}
        )
        
        return session
    
    def get_async_session(self) -> AsyncSession:
        """
        Get a new async database session.
        
        Returns:
            AsyncSession: New async SQLAlchemy session
        """
        if self._async_session_factory is None:
            self._initialize_async_engine()
        
        return self._async_session_factory()
    
    def get_session_factory(self) -> sessionmaker:
        """
        Get the session factory.
        
        Returns:
            sessionmaker: SQLAlchemy session factory
        """
        if self._session_factory is None:
            raise ConnectionError("Session factory not initialized")
        return self._session_factory
    
    def get_scoped_session(self) -> Session:
        """
        Get a scoped session for thread safety.
        
        Returns:
            Session: Scoped SQLAlchemy session
        """
        if self._scoped_session_factory is None:
            raise ConnectionError("Scoped session factory not initialized")
        return self._scoped_session_factory()
    
    def test(self, timeout: int = 5) -> bool:
        """
        Test database connection.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
            
        Reference: [Required Experience: Production engineering role]
        """
        try:
            with self.get_engine().connect() as connection:
                result = connection.execute(text("SELECT 1")).scalar()
                success = result == 1
                
                if success:
                    self.connection_status = ConnectionStatus.CONNECTED
                    self._update_pool_status()
                    self.last_health_check = datetime.now()
                    
                    self.logger.debug("Database connection test passed")
                else:
                    self.connection_status = ConnectionStatus.ERROR
                    self.logger.warning("Database connection test failed - unexpected result")
                
                return success
                
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self.connection_errors.append(str(e))
            
            self.logger.error(
                "Database connection test failed",
                extra={
                    "error": str(e),
                    "timeout": timeout,
                    "host": self.config.host
                }
            )
            return False
    
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """
        Get detailed connection pool status.
        
        Returns:
            Dict with pool status information
            
        Reference: [Absolutely Essential: Design with logging]
        """
        if not self._engine:
            return {
                "status": "error",
                "message": "Engine not initialized",
                "healthy": False
            }
        
        try:
            pool = self._engine.pool
            
            status = {
                "status": self.pool_status.value,
                "healthy": self.pool_status == PoolStatus.HEALTHY,
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "timeout": pool.timeout(),
                "recycle": pool._recycle,
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "connection_errors": self.metrics.connection_errors,
                "total_queries": self.metrics.total_queries,
                "failed_queries": self.metrics.failed_queries,
                "avg_query_time_ms": round(self.metrics.avg_query_time_ms, 2),
                "max_query_time_ms": round(self.metrics.max_query_time_ms, 2),
                "last_checked": self.last_health_check.isoformat() if self.last_health_check else None,
                "connection_status": self.connection_status.value
            }
            
            # Calculate utilization
            total_in_pool = pool.size() + pool.checkedin() + pool.overflow()
            if total_in_pool > 0:
                status["utilization_percent"] = round((pool.checkedout() / total_in_pool) * 100, 2)
            else:
                status["utilization_percent"] = 0
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "healthy": False
            }
    
    def configure_connection_pool(
        self,
        pool_size: Optional[int] = None,
        max_overflow: Optional[int] = None,
        pool_timeout: Optional[int] = None,
        pool_recycle: Optional[int] = None
    ) -> bool:
        """
        Dynamically configure connection pool.
        
        Args:
            pool_size: New pool size
            max_overflow: New max overflow
            pool_timeout: New pool timeout
            pool_recycle: New pool recycle time
            
        Returns:
            bool: True if configuration successful
            
        Reference: [Required Experience: Production engineering role]
        """
        try:
            # Update configuration
            if pool_size is not None:
                self.config.pool_size = pool_size
            if max_overflow is not None:
                self.config.max_overflow = max_overflow
            if pool_timeout is not None:
                self.config.pool_timeout = pool_timeout
            if pool_recycle is not None:
                self.config.pool_recycle = pool_recycle
            
            # Dispose old engine
            if self._engine:
                self._engine.dispose()
            
            # Reinitialize with new configuration
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool
            )
            
            # Update session factory
            self._session_factory = sessionmaker(bind=self._engine)
            self._scoped_session_factory = scoped_session(self._session_factory)
            
            self.logger.info(
                "Connection pool reconfigured",
                extra={
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                    "pool_recycle": self.config.pool_recycle
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to reconfigure connection pool: {str(e)}",
                extra={
                    "pool_size": pool_size,
                    "max_overflow": max_overflow,
                    "error": str(e)
                }
            )
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dict with connection statistics
        """
        self._update_pool_status()
        
        return {
            "connection_status": self.connection_status.value,
            "pool_status": self.pool_status.value,
            "metrics": {
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "idle_connections": self.metrics.total_connections - self.metrics.active_connections,
                "connection_errors": self.metrics.connection_errors,
                "total_queries": self.metrics.total_queries,
                "failed_queries": self.metrics.failed_queries,
                "success_rate": (
                    (self.metrics.total_queries - self.metrics.failed_queries) / 
                    max(self.metrics.total_queries, 1)
                ) * 100,
                "avg_query_time_ms": round(self.metrics.avg_query_time_ms, 2),
                "max_query_time_ms": round(self.metrics.max_query_time_ms, 2)
            },
            "configuration": {
                "host": self.config.host,
                "database": self.config.database,
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "pool_recycle": self.config.pool_recycle
            },
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    def get_dialect(self) -> str:
        """
        Get database dialect.
        
        Returns:
            str: Database dialect name
        """
        if self._engine:
            return self._engine.dialect.name
        return "unknown"
    
    def close_all_connections(self) -> None:
        """
        Close all database connections.
        
        Reference: [Required Experience: Production engineering role]
        """
        try:
            if self._engine:
                self._engine.dispose()
                self.logger.info("All database connections closed")
            
            self.connection_status = ConnectionStatus.DISCONNECTED
            self.pool_status = PoolStatus.ERROR
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")
    
    def reset_connection_pool(self) -> bool:
        """
        Reset connection pool.
        
        Returns:
            bool: True if reset successful
            
        Reference: [Required Experience: Production engineering role]
        """
        try:
            self.close_all_connections()
            self._initialize_connection()
            
            self.logger.info("Connection pool reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset connection pool: {str(e)}")
            return False

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def session_scope():
    """
    Context manager for database sessions with automatic commit/rollback.
    
    Usage:
        with session_scope() as session:
            session.add(object)
            # Session automatically committed on success, rolled back on exception
            
    Reference: [Required Experience: Production AI/data engineering]
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger = logging.getLogger("database.connections")
        logger.error(f"Session rollback due to error: {str(e)}")
        raise
    finally:
        session.close()

@contextmanager
def transaction_scope(isolation_level: str = "READ_COMMITTED"):
    """
    Context manager for database transactions with isolation level control.
    
    Args:
        isolation_level: SQL transaction isolation level
        
    Reference: [Required Experience: Production engineering role]
    """
    session = get_session()
    try:
        # Set isolation level
        session.connection(
            execution_options={"isolation_level": isolation_level}
        )
        
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger = logging.getLogger("database.connections")
        logger.error(f"Transaction rollback due to error: {str(e)}")
        raise
    finally:
        session.close()

@contextmanager
def connection_scope():
    """
    Context manager for raw database connections.
    
    Reference: [Technical Stack: MySQL / PostgreSQL]
    """
    engine = get_engine()
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

_db_connection: Optional[DatabaseConnection] = None

def get_db_connection() -> DatabaseConnection:
    """
    Get the global database connection instance.
    
    Returns:
        DatabaseConnection: Global database connection instance
        
    Reference: [Required Experience: Production AI/data engineering]
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    
    return _db_connection

def get_engine() -> Engine:
    """
    Get SQLAlchemy engine.
    
    Returns:
        Engine: SQLAlchemy engine
    """
    return get_db_connection().get_engine()

def get_async_engine() -> AsyncEngine:
    """
    Get async SQLAlchemy engine.
    
    Returns:
        AsyncEngine: Async SQLAlchemy engine
    """
    return get_db_connection().get_async_engine()

def get_session() -> Session:
    """
    Get a new database session.
    
    Returns:
        Session: New SQLAlchemy session
    """
    return get_db_connection().get_session()

def get_async_session() -> AsyncSession:
    """
    Get a new async database session.
    
    Returns:
        AsyncSession: New async SQLAlchemy session
    """
    return get_db_connection().get_async_session()

def get_session_factory() -> sessionmaker:
    """
    Get session factory.
    
    Returns:
        sessionmaker: SQLAlchemy session factory
    """
    return get_db_connection().get_session_factory()

def close_all_connections() -> None:
    """
    Close all database connections.
    """
    get_db_connection().close_all_connections()

def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful
    """
    return get_db_connection().test()

def get_connection_pool_status() -> Dict[str, Any]:
    """
    Get connection pool status.
    
    Returns:
        Dict with pool status information
    """
    return get_db_connection().get_connection_pool_status()

def configure_connection_pool(**kwargs) -> bool:
    """
    Configure connection pool.
    
    Args:
        **kwargs: Pool configuration parameters
        
    Returns:
        bool: True if configuration successful
    """
    return get_db_connection().configure_connection_pool(**kwargs)

def get_connection_statistics() -> Dict[str, Any]:
    """
    Get connection statistics.
    
    Returns:
        Dict with connection statistics
    """
    return get_db_connection().get_statistics()

# ============================================================================
# HEALTH CHECKS AND MONITORING
# ============================================================================

def perform_health_check() -> Dict[str, Any]:
    """
    Perform comprehensive database health check.
    
    Returns:
        Dict with health check results
        
    Reference: [Absolutely Essential: Design with safety, logging]
    """
    health_check = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "checks": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        connection = get_db_connection()
        
        # Check 1: Basic connection
        connection_check = connection.test()
        health_check["checks"]["connection"] = {
            "passed": connection_check,
            "message": "Database connection established" if connection_check else "Connection failed"
        }
        
        if not connection_check:
            health_check["overall_status"] = "unhealthy"
            health_check["errors"].append("Database connection failed")
        
        # Check 2: Connection pool
        pool_status = connection.get_connection_pool_status()
        health_check["checks"]["connection_pool"] = {
            "passed": pool_status.get("healthy", False),
            "message": pool_status.get("message", "Pool status check"),
            "details": pool_status
        }
        
        if not pool_status.get("healthy", False):
            health_check["overall_status"] = "degraded"
            health_check["warnings"].append(f"Connection pool issues: {pool_status.get('status')}")
        
        # Check 3: Query performance
        try:
            with get_session() as session:
                import time
                start_time = time.time()
                result = session.execute(text("SELECT 1")).scalar()
                query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                health_check["checks"]["query_performance"] = {
                    "passed": query_time < 100,  # Should be < 100ms
                    "message": f"Query executed in {query_time:.2f}ms",
                    "response_time_ms": query_time,
                    "threshold_ms": 100
                }
                
                if query_time >= 100:
                    health_check["overall_status"] = "degraded"
                    health_check["warnings"].append(f"Slow query response: {query_time:.2f}ms")
        except Exception as query_error:
            health_check["checks"]["query_performance"] = {
                "passed": False,
                "message": f"Query failed: {str(query_error)}"
            }
            health_check["overall_status"] = "unhealthy"
            health_check["errors"].append(f"Query execution failed: {str(query_error)}")
        
        # Check 4: Connection error rate
        stats = connection.get_statistics()
        error_rate = (
            stats["metrics"]["connection_errors"] / 
            max(stats["metrics"]["total_queries"], 1)
        ) * 100
        
        health_check["checks"]["error_rate"] = {
            "passed": error_rate < 1,  # Less than 1% error rate
            "message": f"Error rate: {error_rate:.2f}%",
            "error_rate_percent": error_rate,
            "threshold_percent": 1
        }
        
        if error_rate >= 1:
            health_check["overall_status"] = "degraded"
            health_check["warnings"].append(f"High error rate: {error_rate:.2f}%")
        
        # Check 5: Connection utilization
        if "utilization_percent" in pool_status:
            utilization = pool_status["utilization_percent"]
            health_check["checks"]["connection_utilization"] = {
                "passed": utilization < 80,  # Less than 80% utilization
                "message": f"Connection utilization: {utilization:.1f}%",
                "utilization_percent": utilization,
                "threshold_percent": 80
            }
            
            if utilization >= 80:
                health_check["overall_status"] = "degraded"
                health_check["warnings"].append(f"High connection utilization: {utilization:.1f}%")
        
        # Overall assessment
        passed_checks = sum(1 for check in health_check["checks"].values() if check["passed"])
        total_checks = len(health_check["checks"])
        
        health_check["summary"] = {
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "success_rate": (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        }
        
        logger = logging.getLogger("database.connections")
        logger.info(
            "Database health check completed",
            extra={
                "status": health_check["overall_status"],
                "checks_passed": passed_checks,
                "checks_total": total_checks,
                "error_count": len(health_check["errors"]),
                "warning_count": len(health_check["warnings"])
            }
        )
        
    except Exception as e:
        health_check["overall_status"] = "error"
        health_check["errors"].append(f"Health check failed: {str(e)}")
        
        logger = logging.getLogger("database.connections")
        logger.error(f"Database health check failed: {str(e)}")
    
    return health_check

def start_connection_monitoring(interval_seconds: int = 60) -> threading.Thread:
    """
    Start background thread for connection monitoring.
    
    Args:
        interval_seconds: Monitoring interval in seconds
        
    Returns:
        threading.Thread: Monitoring thread
        
    Reference: [Required Experience: Production engineering role]
    """
    def monitor_connections():
        """Background monitoring function."""
        logger = logging.getLogger("database.connections.monitoring")
        
        while True:
            try:
                # Perform health check
                health = perform_health_check()
                
                # Log if status is not healthy
                if health["overall_status"] != "healthy":
                    logger.warning(
                        f"Database health check: {health['overall_status']}",
                        extra={
                            "status": health["overall_status"],
                            "errors": health["errors"],
                            "warnings": health["warnings"]
                        }
                    )
                
                # Check connection pool status
                pool_status = get_connection_pool_status()
                if not pool_status.get("healthy", False):
                    logger.warning(
                        "Connection pool unhealthy",
                        extra={"pool_status": pool_status}
                    )
                
                # Sleep until next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {str(e)}")
                time.sleep(interval_seconds)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_connections,
        name="database_connection_monitor",
        daemon=True
    )
    monitor_thread.start()
    
    logger = logging.getLogger("database.connections")
    logger.info(
        f"Started database connection monitoring (interval: {interval_seconds}s)"
    )
    
    return monitor_thread

# ============================================================================
# CONNECTION UTILITIES
# ============================================================================

def execute_with_retry(
    func,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_exceptions: tuple = (exc.DBAPIError, exc.OperationalError)
):
    """
    Execute a database function with retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        retry_exceptions: Exceptions to retry on
        
    Returns:
        Function result
        
    Reference: [Required Experience: Production engineering role]
    """
    logger = logging.getLogger("database.connections.retry")
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_exceptions as e:
            if attempt == max_retries:
                logger.error(
                    f"Database operation failed after {max_retries} retries",
                    extra={
                        "error": str(e),
                        "max_retries": max_retries,
                        "attempt": attempt + 1
                    }
                )
                raise
            
            logger.warning(
                f"Database operation failed, retrying ({attempt + 1}/{max_retries})",
                extra={
                    "error": str(e),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "retry_delay": retry_delay
                }
            )
            
            time.sleep(retry_delay)
    
    raise RuntimeError("Maximum retries exceeded")

def get_database_size() -> Optional[int]:
    """
    Get database size in bytes.
    
    Returns:
        int: Database size in bytes or None if not available
    """
    try:
        with get_session() as session:
            # MySQL specific query
            result = session.execute(text(
                "SELECT SUM(data_length + index_length) as size "
                "FROM information_schema.TABLES "
                "WHERE table_schema = DATABASE()"
            )).scalar()
            
            return int(result) if result else None
            
    except Exception as e:
        logger = logging.getLogger("database.connections")
        logger.error(f"Failed to get database size: {str(e)}")
        return None

def get_table_row_counts() -> Dict[str, int]:
    """
    Get row counts for all tables.
    
    Returns:
        Dict mapping table names to row counts
    """
    table_counts = {}
    
    try:
        with get_session() as session:
            # Get all table names
            result = session.execute(text(
                "SELECT table_name FROM information_schema.TABLES "
                "WHERE table_schema = DATABASE()"
            )).fetchall()
            
            for table_name, in result:
                try:
                    count_result = session.execute(text(
                        f"SELECT COUNT(*) FROM `{table_name}`"
                    )).scalar()
                    table_counts[table_name] = count_result or 0
                except Exception:
                    table_counts[table_name] = 0
        
        logger = logging.getLogger("database.connections")
        logger.debug(f"Retrieved row counts for {len(table_counts)} tables")
        
    except Exception as e:
        logger = logging.getLogger("database.connections")
        logger.error(f"Failed to get table row counts: {str(e)}")
    
    return table_counts

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_database_connections() -> DatabaseConnection:
    """
    Initialize database connections and start monitoring.
    
    Returns:
        DatabaseConnection: Initialized database connection
        
    Reference: [Required Experience: Production engineering role]
    """
    connection = get_db_connection()
    
    # Perform initial health check
    health = perform_health_check()
    
    if health["overall_status"] != "healthy":
        logger = logging.getLogger("database.connections")
        logger.warning(
            "Database health check issues on initialization",
            extra={
                "status": health["overall_status"],
                "errors": health["errors"],
                "warnings": health["warnings"]
            }
        )
    
    # Start monitoring in production
    from ...config.settings import get_settings
    settings = get_settings()
    
    if settings.is_production:
        start_connection_monitoring(interval_seconds=300)  # Every 5 minutes
    
    logger = logging.getLogger("database.connections")
    logger.info(
        "Database connections initialized",
        extra={
            "status": health["overall_status"],
            "dialect": connection.get_dialect(),
            "pool_size": connection.config.pool_size,
            "monitoring_enabled": settings.is_production
        }
    )
    
    return connection

# Auto-initialize on module import
initialize_database_connections()
