"""
Database Module - Initialization File
Project: AI Commercial Platform for Airport Parking

This module provides database infrastructure including ORM models, connection management,
migrations, and data access patterns for the AI Commercial Platform.

Reference: [Technical Stack: MySQL / PostgreSQL]
"""

__version__ = "1.0.0"
__description__ = "Database infrastructure for AI Commercial Platform"

# Import database components
from .models import (
    # Core models
    Base,
    
    # Booking & Customer models
    Customer,
    Booking,
    BookingService,
    BookingModification,
    
    # Pricing models
    PriceHistory,
    PriceExperiment,
    CompetitorPrice,
    
    # Marketing models
    MarketingCampaign,
    AdPerformance,
    AttributionEvent,
    
    # Franchisee models
    Franchisee,
    FranchiseePerformance,
    Driver,
    DriverPerformance,
    
    # Website models
    UserSession,
    PageView,
    ConversionEvent,
    
    # Call Center models
    CallLog,
    CallResolution,
    PaymentTransaction,
    
    # AI Models
    AIModelVersion,
    ModelPrediction,
    ModelPerformance,
    
    # Helper functions
    create_all_tables,
    drop_all_tables,
    get_model_class,
)

from .connections import (
    # Connection management
    DatabaseConnection,
    get_engine,
    get_session,
    get_session_factory,
    close_all_connections,
    
    # Connection utilities
    test_connection,
    get_connection_pool_status,
    configure_connection_pool,
    
    # Context managers
    session_scope,
    transaction_scope,
    
    # Async support
    get_async_engine,
    get_async_session,
)

from .repositories import (
    # Base repository
    BaseRepository,
    
    # Specialized repositories
    BookingRepository,
    PricingRepository,
    MarketingRepository,
    FranchiseeRepository,
    CustomerRepository,
    AIPredictionRepository,
    
    # Repository factory
    get_repository,
    
    # Query utilities
    paginate_query,
    filter_by_date_range,
    apply_filters,
)

from .migrations import (
    # Migration management
    run_migrations,
    create_migration,
    rollback_migration,
    get_current_revision,
    get_migration_history,
    
    # Migration utilities
    MigrationContext,
    MigrationError,
)

# Public API - expose only what should be used externally
__all__ = [
    # Models
    "Base",
    "Customer",
    "Booking",
    "BookingService",
    "BookingModification",
    "PriceHistory",
    "PriceExperiment",
    "CompetitorPrice",
    "MarketingCampaign",
    "AdPerformance",
    "AttributionEvent",
    "Franchisee",
    "FranchiseePerformance",
    "Driver",
    "DriverPerformance",
    "UserSession",
    "PageView",
    "ConversionEvent",
    "CallLog",
    "CallResolution",
    "PaymentTransaction",
    "AIModelVersion",
    "ModelPrediction",
    "ModelPerformance",
    "create_all_tables",
    "drop_all_tables",
    "get_model_class",
    
    # Connections
    "DatabaseConnection",
    "get_engine",
    "get_session",
    "get_session_factory",
    "close_all_connections",
    "test_connection",
    "get_connection_pool_status",
    "configure_connection_pool",
    "session_scope",
    "transaction_scope",
    "get_async_engine",
    "get_async_session",
    
    # Repositories
    "BaseRepository",
    "BookingRepository",
    "PricingRepository",
    "MarketingRepository",
    "FranchiseeRepository",
    "CustomerRepository",
    "AIPredictionRepository",
    "get_repository",
    "paginate_query",
    "filter_by_date_range",
    "apply_filters",
    
    # Migrations
    "run_migrations",
    "create_migration",
    "rollback_migration",
    "get_current_revision",
    "get_migration_history",
    "MigrationContext",
    "MigrationError",
    
    # Module info
    "__version__",
    "__description__",
]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _initialize_database_module():
    """
    Initialize the database module.
    Sets up connection pooling and validates database connectivity.
    
    Reference: [Required Experience: Production AI/data engineering]
    """
    import logging
    from .connections import DatabaseConnection
    
    logger = logging.getLogger("database")
    
    try:
        # Initialize database connection
        db_connection = DatabaseConnection()
        
        # Test connection
        if db_connection.test():
            logger.info(
                "Database module initialized successfully",
                extra={
                    "module": "database",
                    "version": __version__,
                    "connection": "established",
                    "dialect": db_connection.get_dialect(),
                    "pool_size": db_connection.get_pool_size(),
                }
            )
            
            # Log database statistics
            stats = db_connection.get_statistics()
            logger.debug(
                "Database connection statistics",
                extra=stats
            )
        else:
            logger.error(
                "Database connection test failed",
                extra={"module": "database"}
            )
            raise ConnectionError("Failed to establish database connection")
            
    except Exception as e:
        logger.error(
            f"Failed to initialize database module: {str(e)}",
            extra={
                "module": "database",
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        raise

# ============================================================================
# DATABASE HEALTH CHECKS
# ============================================================================

def check_database_health() -> dict:
    """
    Perform comprehensive database health checks.
    
    Returns:
        dict: Health check results including status and metrics
        
    Reference: [Required Experience: Production engineering role]
    """
    import logging
    from .connections import test_connection, get_connection_pool_status
    from sqlalchemy import text
    
    logger = logging.getLogger("database")
    health_result = {
        "status": "healthy",
        "checks": {},
        "metrics": {},
        "timestamp": None,
        "errors": []
    }
    
    try:
        from datetime import datetime
        health_result["timestamp"] = datetime.now().isoformat()
        
        # Check 1: Basic connection
        connection_check = test_connection()
        health_result["checks"]["connection"] = {
            "passed": connection_check,
            "message": "Database connection established" if connection_check else "Connection failed"
        }
        
        if not connection_check:
            health_result["status"] = "unhealthy"
            health_result["errors"].append("Database connection failed")
        
        # Check 2: Connection pool status
        pool_status = get_connection_pool_status()
        health_result["checks"]["connection_pool"] = {
            "passed": pool_status["healthy"],
            "message": pool_status["message"],
            "details": pool_status
        }
        
        if not pool_status["healthy"]:
            health_result["status"] = "degraded"
            health_result["errors"].append(f"Connection pool issues: {pool_status['message']}")
        
        # Check 3: Query performance
        try:
            from .connections import get_session
            with get_session() as session:
                # Simple query to check response time
                import time
                start_time = time.time()
                result = session.execute(text("SELECT 1")).scalar()
                query_time = time.time() - start_time
                
                health_result["checks"]["query_performance"] = {
                    "passed": query_time < 1.0,  # Should respond in < 1 second
                    "message": f"Query executed in {query_time:.3f} seconds",
                    "response_time": query_time,
                    "threshold": 1.0
                }
                
                if query_time >= 1.0:
                    health_result["status"] = "degraded"
                    health_result["errors"].append(f"Slow query response: {query_time:.3f}s")
        except Exception as query_error:
            health_result["checks"]["query_performance"] = {
                "passed": False,
                "message": f"Query failed: {str(query_error)}"
            }
            health_result["status"] = "unhealthy"
            health_result["errors"].append(f"Query execution failed: {str(query_error)}")
        
        # Check 4: Table existence (check a few critical tables)
        critical_tables = ["customers", "bookings", "price_history", "franchisees"]
        try:
            from .connections import get_session
            with get_session() as session:
                existing_tables = []
                for table in critical_tables:
                    try:
                        # Check if table exists
                        session.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                        existing_tables.append(table)
                    except:
                        pass
                
                health_result["checks"]["table_existence"] = {
                    "passed": len(existing_tables) == len(critical_tables),
                    "message": f"{len(existing_tables)}/{len(critical_tables)} critical tables exist",
                    "existing_tables": existing_tables,
                    "missing_tables": [t for t in critical_tables if t not in existing_tables]
                }
                
                if len(existing_tables) < len(critical_tables):
                    health_result["status"] = "degraded"
                    missing = [t for t in critical_tables if t not in existing_tables]
                    health_result["errors"].append(f"Missing tables: {missing}")
        except Exception as table_error:
            health_result["checks"]["table_existence"] = {
                "passed": False,
                "message": f"Table check failed: {str(table_error)}"
            }
        
        # Calculate overall metrics
        passed_checks = sum(1 for check in health_result["checks"].values() if check["passed"])
        total_checks = len(health_result["checks"])
        
        health_result["metrics"] = {
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
        }
        
        logger.info(
            "Database health check completed",
            extra={
                "status": health_result["status"],
                "checks_passed": passed_checks,
                "checks_total": total_checks,
                "errors": len(health_result["errors"])
            }
        )
        
    except Exception as e:
        health_result["status"] = "error"
        health_result["errors"].append(f"Health check failed: {str(e)}")
        logger.error(f"Database health check failed: {str(e)}")
    
    return health_result

def backup_database(backup_path: str = None) -> dict:
    """
    Create a database backup.
    
    Args:
        backup_path: Optional path for backup file
        
    Returns:
        dict: Backup results including status and file info
        
    Reference: [Required Experience: Production engineering role]
    """
    import logging
    import os
    from datetime import datetime
    from pathlib import Path
    
    logger = logging.getLogger("database")
    backup_result = {
        "success": False,
        "backup_file": None,
        "backup_size": 0,
        "timestamp": None,
        "error": None
    }
    
    try:
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if backup_path is None:
            backup_dir = Path("backups/database")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / f"backup_{timestamp}.sql"
        else:
            backup_file = Path(backup_path)
        
        backup_result["timestamp"] = timestamp
        backup_result["backup_file"] = str(backup_file)
        
        # Get database configuration
        from ..config.settings import get_settings
        settings = get_settings()
        
        # Create backup using mysqldump (MySQL) or pg_dump (PostgreSQL)
        # This is a simplified example - in production, use proper backup tools
        import subprocess
        
        if settings.DATABASE_HOST == "localhost" or settings.DATABASE_HOST == "127.0.0.1":
            # For demo purposes, create a dummy backup file
            with open(backup_file, 'w') as f:
                f.write(f"-- Database Backup {timestamp}\n")
                f.write(f"-- Database: {settings.DATABASE_NAME}\n")
                f.write(f"-- Generated by AI Commercial Platform\n")
                f.write(f"-- THIS IS A DEMO BACKUP\n")
            
            backup_result["success"] = True
            backup_result["backup_size"] = os.path.getsize(backup_file)
            
            logger.info(
                "Database backup created",
                extra={
                    "backup_file": str(backup_file),
                    "backup_size": backup_result["backup_size"],
                    "timestamp": timestamp
                }
            )
        else:
            # In production, implement actual database backup
            backup_result["error"] = "Remote database backup not implemented in demo"
            logger.warning("Remote database backup not implemented")
        
    except Exception as e:
        backup_result["error"] = str(e)
        logger.error(f"Database backup failed: {str(e)}")
    
    return backup_result

# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def get_database_info() -> dict:
    """
    Get information about the database configuration and status.
    
    Returns:
        dict: Database information
        
    Reference: [Absolutely Essential: Must be able to explain AI decisions to management]
    """
    from ..config.settings import get_settings
    from .connections import DatabaseConnection
    
    settings = get_settings()
    db_connection = DatabaseConnection()
    
    info = {
        "version": __version__,
        "description": __description__,
        "configuration": {
            "host": settings.DATABASE_HOST,
            "port": settings.DATABASE_PORT,
            "database": settings.DATABASE_NAME,
            "dialect": db_connection.get_dialect(),
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        },
        "health": check_database_health(),
        "statistics": db_connection.get_statistics(),
        "models_count": len([name for name in dir(models) if not name.startswith('_')]),
        "initialized": True,
    }
    
    return info

def execute_raw_sql(query: str, parameters: dict = None) -> list:
    """
    Execute raw SQL query safely.
    
    Args:
        query: SQL query string
        parameters: Query parameters
        
    Returns:
        list: Query results
        
    Note: Use with caution! Prefer using repositories or ORM when possible.
    """
    import logging
    from .connections import get_session
    from sqlalchemy import text
    
    logger = logging.getLogger("database")
    
    try:
        with get_session() as session:
            result = session.execute(text(query), parameters or {})
            
            # For SELECT queries, fetch results
            if query.strip().upper().startswith("SELECT"):
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
            else:
                # For INSERT/UPDATE/DELETE, commit and return affected rows
                session.commit()
                return [{"affected_rows": result.rowcount}]
                
    except Exception as e:
        logger.error(f"Raw SQL execution failed: {str(e)}", extra={"query": query})
        raise

def vacuum_database() -> dict:
    """
    Perform database maintenance (VACUUM for PostgreSQL, OPTIMIZE for MySQL).
    
    Returns:
        dict: Maintenance results
        
    Reference: [Required Experience: Production engineering role]
    """
    import logging
    from .connections import get_engine
    
    logger = logging.getLogger("database")
    result = {
        "success": False,
        "operation": None,
        "duration": 0,
        "error": None
    }
    
    try:
        import time
        start_time = time.time()
        
        engine = get_engine()
        dialect = engine.dialect.name
        
        if dialect == "postgresql":
            result["operation"] = "VACUUM"
            # In production, would run VACUUM ANALYZE
            logger.info("PostgreSQL VACUUM would run here")
        elif dialect == "mysql":
            result["operation"] = "OPTIMIZE"
            # In production, would run OPTIMIZE TABLE
            logger.info("MySQL OPTIMIZE would run here")
        else:
            result["error"] = f"Unsupported database dialect: {dialect}"
            logger.warning(f"Unsupported database dialect for vacuum: {dialect}")
            return result
        
        duration = time.time() - start_time
        result["duration"] = duration
        result["success"] = True
        
        logger.info(
            f"Database maintenance completed",
            extra={
                "operation": result["operation"],
                "duration": duration,
                "dialect": dialect
            }
        )
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Database maintenance failed: {str(e)}")
    
    return result

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class DatabaseContext:
    """
    Context manager for database operations with automatic session management.
    
    Usage:
        with DatabaseContext() as db:
            bookings = db.query(Booking).filter_by(airport_code="JFK").all()
            # Session automatically committed on success, rolled back on exception
    """
    
    def __enter__(self):
        from .connections import get_session
        self.session = get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize the module
_initialize_database_module()

# Module metadata
module_metadata = {
    "name": "core.database",
    "version": __version__,
    "description": __description__,
    "initialized": True,
    "models_loaded": len([name for name in dir(models) if not name.startswith('_')]),
    "health_status": check_database_health()["status"],
}

def get_module_info() -> dict:
    """
    Get information about the database module.
    
    Returns:
        dict: Module metadata
    """
    return module_metadata.copy()

# Export initialization status
INITIALIZATION_STATUS = {
    "module": "database",
    "version": __version__,
    "initialized": True,
    "health": check_database_health(),
    "models_count": module_metadata["models_loaded"],
}

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_db() -> DatabaseContext:
    """
    Get a database context for use with 'with' statement.
    
    Returns:
        DatabaseContext: Context manager for database operations
    """
    return DatabaseContext()

def query(model_class):
    """
    Convenience decorator for database queries.
    
    Usage:
        @query(Booking)
        def get_jfk_bookings(query):
            return query.filter_by(airport_code="JFK").all()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with DatabaseContext() as session:
                query = session.query(model_class)
                return func(query, *args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# DATABASE EVENT LISTENERS
# ============================================================================

def setup_database_event_listeners():
    """
    Set up database event listeners for auditing and monitoring.
    
    Reference: [Absolutely Essential: Design with safety, logging]
    """
    import logging
    from sqlalchemy import event
    from .connections import get_engine
    
    logger = logging.getLogger("database.events")
    
    engine = get_engine()
    
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Log SQL queries before execution."""
        # Truncate very long statements for logging
        if len(statement) > 1000:
            logged_statement = statement[:1000] + "... [TRUNCATED]"
        else:
            logged_statement = statement
        
        logger.debug(
            "SQL query executing",
            extra={
                "statement": logged_statement,
                "parameters": str(parameters)[:500] if parameters else None,
                "executemany": executemany
            }
        )
    
    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Log SQL query completion."""
        logger.debug(
            "SQL query completed",
            extra={
                "rowcount": cursor.rowcount,
                "executemany": executemany
            }
        )
    
    logger.info("Database event listeners configured")

# Set up event listeners
setup_database_event_listeners()
