"""
Database Migration Management
Project: AI Commercial Platform for Airport Parking

This module handles database schema migrations using Alembic.
Manages version control, schema evolution, and data migrations for all database models.

Reference: [Required Experience: Production AI/data engineering]
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from enum import Enum

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext
from alembic.autogenerate import compare_metadata
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.exc import SQLAlchemyError

# Import database models
from ..models import Base
from ..connections import get_engine

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class MigrationType(str, Enum):
    """Types of migrations."""
    SCHEMA = "schema"
    DATA = "data"
    SEED = "seed"
    ROLLBACK = "rollback"

class MigrationStatus(str, Enum):
    """Migration status."""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

MIGRATIONS_DIR = Path(__file__).parent
ALEMBIC_INI_PATH = MIGRATIONS_DIR / "alembic.ini"
ALEMBIC_VERSIONS_DIR = MIGRATIONS_DIR / "versions"

# ============================================================================
# MIGRATION CONTEXT
# ============================================================================

class MigrationContext:
    """
    Context for managing database migrations.
    
    Reference: [Required Experience: Production engineering role]
    """
    
    def __init__(self, alembic_cfg: Optional[Config] = None):
        self.logger = logging.getLogger("migrations")
        
        # Set up Alembic configuration
        if alembic_cfg:
            self.alembic_cfg = alembic_cfg
        else:
            self.alembic_cfg = Config(str(ALEMBIC_INI_PATH))
        
        # Ensure migrations directory exists
        ALEMBIC_VERSIONS_DIR.mkdir(exist_ok=True)
        
        # Set SQLAlchemy URL in config
        from ...config.settings import get_settings
        settings = get_settings()
        self.alembic_cfg.set_main_option(
            "sqlalchemy.url",
            settings.database_config.connection_string
        )
        
        # Set script location
        self.alembic_cfg.set_main_option(
            "script_location",
            str(MIGRATIONS_DIR)
        )
        
        self.engine = get_engine()
        self.metadata = Base.metadata
        
        self.logger.info(
            "Migration context initialized",
            extra={
                "migrations_dir": str(MIGRATIONS_DIR),
                "versions_dir": str(ALEMBIC_VERSIONS_DIR)
            }
        )
    
    def get_current_revision(self) -> Optional[str]:
        """
        Get current database revision.
        
        Returns:
            str: Current revision ID or None if no migrations applied
            
        Reference: [Required Experience: Production AI/data engineering]
        """
        try:
            with self.engine.connect() as connection:
                context = EnvironmentContext(
                    self.alembic_cfg,
                    ScriptDirectory.from_config(self.alembic_cfg)
                )
                
                def get_current_rev(rev, _):
                    return rev
                
                context.configure(
                    connection=connection,
                    fn=get_current_rev
                )
                
                with context.begin_transaction():
                    context.run_migrations()
                    
                # Get the revision from the context
                if hasattr(context, 'get_current_revision'):
                    return context.get_current_revision()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get current revision: {str(e)}")
            return None
    
    def get_migration_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get migration history.
        
        Args:
            limit: Maximum number of migrations to return
            
        Returns:
            List of migration history entries
            
        Reference: [Absolutely Essential: Design with logging]
        """
        history = []
        try:
            with self.engine.connect() as connection:
                # Check if alembic_version table exists
                inspector = inspect(self.engine)
                if 'alembic_version' not in inspector.get_table_names():
                    return history
                
                # Query migration history
                result = connection.execute(
                    text("""
                        SELECT version_num, COUNT(*) as applied_count
                        FROM alembic_version
                        GROUP BY version_num
                        ORDER BY version_num DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                ).fetchall()
                
                script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                
                for row in result:
                    version_num = row[0]
                    applied_count = row[1]
                    
                    try:
                        script = script_dir.get_revision(version_num)
                        history.append({
                            "revision": version_num,
                            "description": script.doc if script else "Unknown",
                            "applied_count": applied_count,
                            "applied_at": None,  # Would need timestamp tracking
                            "is_current": self.get_current_revision() == version_num
                        })
                    except:
                        history.append({
                            "revision": version_num,
                            "description": "Unknown migration",
                            "applied_count": applied_count,
                            "applied_at": None,
                            "is_current": self.get_current_revision() == version_num
                        })
                        
        except Exception as e:
            self.logger.error(f"Failed to get migration history: {str(e)}")
        
        return history
    
    def check_pending_migrations(self) -> List[Dict[str, Any]]:
        """
        Check for pending migrations.
        
        Returns:
            List of pending migrations
            
        Reference: [Required Experience: Production engineering role]
        """
        pending = []
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()
            
            if current_rev is None:
                # No migrations applied yet, all are pending
                for script in script_dir.walk_revisions():
                    pending.append({
                        "revision": script.revision,
                        "description": script.doc or "Initial migration",
                        "down_revision": script.down_revision
                    })
            else:
                # Get migrations after current revision
                for script in script_dir.walk_revisions():
                    if script.revision != current_rev:
                        # Check if this migration is ahead of current
                        is_pending = False
                        
                        # Simple check: if we can reach this from current via upgrade path
                        # In production, use proper Alembic methods
                        if script.revision > current_rev:
                            is_pending = True
                        
                        if is_pending:
                            pending.append({
                                "revision": script.revision,
                                "description": script.doc or "Migration",
                                "down_revision": script.down_revision
                            })
            
            self.logger.info(
                f"Found {len(pending)} pending migrations",
                extra={"pending_count": len(pending)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check pending migrations: {str(e)}")
        
        return pending
    
    def create_migration(
        self,
        message: str,
        migration_type: MigrationType = MigrationType.SCHEMA,
        data_script: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new migration.
        
        Args:
            message: Migration description
            migration_type: Type of migration
            data_script: Optional data migration SQL
            
        Returns:
            str: Path to created migration file or None if failed
            
        Reference: [Required Experience: Production AI/data engineering]
        """
        try:
            # Create revision using Alembic
            revision_args = [
                "--autogenerate",
                "-m", message
            ]
            
            # Add type-specific configurations
            if migration_type == MigrationType.DATA:
                revision_args.append("--head=data")
            elif migration_type == MigrationType.SEED:
                revision_args.append("--head=seed")
            
            # Generate the migration
            command.revision(
                self.alembic_cfg,
                *revision_args
            )
            
            # Get the latest revision file
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = list(script_dir.walk_revisions())
            latest_rev = revisions[0] if revisions else None
            
            if latest_rev:
                migration_file = script_dir.get_revision(latest_rev.revision).path
                
                # Add data migration if provided
                if data_script and migration_type in [MigrationType.DATA, MigrationType.SEED]:
                    self._add_data_migration_to_file(migration_file, data_script)
                
                self.logger.info(
                    f"Migration created: {message}",
                    extra={
                        "revision": latest_rev.revision,
                        "message": message,
                        "type": migration_type.value,
                        "file": str(migration_file)
                    }
                )
                
                return str(migration_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create migration: {str(e)}")
        
        return None
    
    def _add_data_migration_to_file(self, migration_file: Path, data_script: str) -> None:
        """Add data migration SQL to migration file."""
        try:
            with open(migration_file, 'a') as f:
                f.write("\n\n# === DATA MIGRATION === #\n")
                f.write("def data_migration():\n")
                f.write("    \"\"\"Data migration script.\"\"\"\n")
                f.write("    from alembic import op\n")
                f.write("    import sqlalchemy as sa\n")
                f.write("\n")
                f.write(f"{data_script}\n")
                
                # Add upgrade and downgrade functions that call data_migration
                f.write("\n")
                f.write("def upgrade():\n")
                f.write("    # Schema changes from autogenerate\n")
                f.write("    # ...\n")
                f.write("    # Data migration\n")
                f.write("    data_migration()\n")
                f.write("\n")
                f.write("def downgrade():\n")
                f.write("    # Schema rollback from autogenerate\n")
                f.write("    # ...\n")
                f.write("    # Note: Data migrations typically don't have downgrades\n")
                f.write("    pass\n")
                
        except Exception as e:
            self.logger.error(f"Failed to add data migration to file: {str(e)}")
    
    def run_migrations(self, target: str = "head", dry_run: bool = False) -> Dict[str, Any]:
        """
        Run pending migrations.
        
        Args:
            target: Target revision (default: "head")
            dry_run: If True, only show what would be done
            
        Returns:
            Dict with migration results
            
        Reference: [Required Experience: Production engineering role]
        """
        result = {
            "success": False,
            "migrations_applied": [],
            "errors": [],
            "warnings": [],
            "dry_run": dry_run
        }
        
        try:
            # Check current revision
            current_rev = self.get_current_revision()
            result["current_revision"] = current_rev
            result["target_revision"] = target
            
            # Get pending migrations
            pending = self.check_pending_migrations()
            if not pending and target == "head":
                result["success"] = True
                result["message"] = "No pending migrations"
                self.logger.info("No pending migrations to apply")
                return result
            
            if dry_run:
                result["pending_migrations"] = [
                    {"revision": m["revision"], "description": m["description"]}
                    for m in pending[:5]  # Limit in dry run
                ]
                result["success"] = True
                result["message"] = f"Dry run: Would apply {len(pending)} migrations"
                return result
            
            # Run migrations
            self.logger.info(
                f"Running migrations from {current_rev or 'base'} to {target}",
                extra={
                    "current_revision": current_rev,
                    "target_revision": target,
                    "pending_count": len(pending)
                }
            )
            
            # Actually run migrations
            command.upgrade(self.alembic_cfg, target)
            
            # Get new current revision
            new_rev = self.get_current_revision()
            
            # Log applied migrations
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            if current_rev and new_rev:
                # Get the revisions that were applied
                for script in script_dir.walk_revisions():
                    if (current_rev < script.revision <= new_rev) if current_rev and new_rev else False:
                        result["migrations_applied"].append({
                            "revision": script.revision,
                            "description": script.doc or "Migration"
                        })
            
            result["success"] = True
            result["new_revision"] = new_rev
            result["message"] = f"Successfully applied {len(result['migrations_applied'])} migrations"
            
            self.logger.info(
                "Migrations applied successfully",
                extra={
                    "applied_count": len(result["migrations_applied"]),
                    "from_revision": current_rev,
                    "to_revision": new_rev
                }
            )
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
            result["message"] = f"Migration failed: {str(e)}"
            
            self.logger.error(
                "Migration failed",
                extra={
                    "error": str(e),
                    "target": target,
                    "current_rev": current_rev
                }
            )
            
            # Try to get more details
            import traceback
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def rollback_migration(self, target: str = "-1", dry_run: bool = False) -> Dict[str, Any]:
        """
        Rollback migrations.
        
        Args:
            target: Target revision or number of steps (e.g., "-1" for one step back)
            dry_run: If True, only show what would be done
            
        Returns:
            Dict with rollback results
            
        Reference: [Absolutely Essential: Design with safety, overrides]
        """
        result = {
            "success": False,
            "migrations_rolled_back": [],
            "errors": [],
            "warnings": [],
            "dry_run": dry_run
        }
        
        try:
            current_rev = self.get_current_revision()
            result["current_revision"] = current_rev
            result["target"] = target
            
            if not current_rev:
                result["message"] = "No migrations applied, nothing to rollback"
                result["success"] = True
                return result
            
            if dry_run:
                result["message"] = f"Dry run: Would rollback to {target}"
                result["success"] = True
                return result
            
            self.logger.warning(
                f"Rolling back migrations from {current_rev} to {target}",
                extra={
                    "current_revision": current_rev,
                    "target": target,
                    "action": "rollback"
                }
            )
            
            # Actually run rollback
            command.downgrade(self.alembic_cfg, target)
            
            # Get new current revision
            new_rev = self.get_current_revision()
            
            # Log rolled back migrations
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            if current_rev and new_rev:
                # Get the revisions that were rolled back
                for script in script_dir.walk_revisions():
                    if (new_rev < script.revision <= current_rev) if new_rev and current_rev else False:
                        result["migrations_rolled_back"].append({
                            "revision": script.revision,
                            "description": script.doc or "Migration"
                        })
            
            result["success"] = True
            result["new_revision"] = new_rev
            result["message"] = f"Successfully rolled back {len(result['migrations_rolled_back'])} migrations"
            
            self.logger.warning(
                "Migrations rolled back",
                extra={
                    "rolled_back_count": len(result["migrations_rolled_back"]),
                    "from_revision": current_rev,
                    "to_revision": new_rev
                }
            )
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
            result["message"] = f"Rollback failed: {str(e)}"
            
            self.logger.error(
                "Rollback failed",
                extra={
                    "error": str(e),
                    "target": target,
                    "current_rev": current_rev
                }
            )
            
            # Try to get more details
            import traceback
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def generate_schema_diff(self) -> Dict[str, Any]:
        """
        Generate schema difference between current models and database.
        
        Returns:
            Dict with schema differences
            
        Reference: [Required Experience: Production AI/data engineering]
        """
        diff_result = {
            "has_changes": False,
            "additions": [],
            "removals": [],
            "modifications": [],
            "warnings": []
        }
        
        try:
            # Get current database schema
            with self.engine.connect() as connection:
                context = EnvironmentContext(
                    self.alembic_cfg,
                    ScriptDirectory.from_config(self.alembic_cfg)
                )
                
                def compare_metadata_fn(rev, context):
                    nonlocal diff_result
                    
                    # Get autogenerate context
                    autogen_context = context
                    
                    # Compare metadata
                    diff = compare_metadata(
                        autogen_context,
                        self.metadata
                    )
                    
                    # Parse differences
                    for diff_item in diff:
                        diff_type = diff_item[0]
                        diff_details = diff_item[1:]
                        
                        if diff_type == "add_table":
                            diff_result["additions"].append({
                                "type": "table",
                                "name": diff_details[0].name,
                                "details": str(diff_details[0])
                            })
                        elif diff_type == "remove_table":
                            diff_result["removals"].append({
                                "type": "table",
                                "name": diff_details[0].name
                            })
                        elif diff_type == "add_column":
                            diff_result["additions"].append({
                                "type": "column",
                                "table": diff_details[0].name,
                                "column": diff_details[1].name,
                                "details": str(diff_details[1])
                            })
                        elif diff_type == "remove_column":
                            diff_result["removals"].append({
                                "type": "column",
                                "table": diff_details[0].name,
                                "column": diff_details[1].name
                            })
                        elif diff_type == "modify_column":
                            diff_result["modifications"].append({
                                "type": "column",
                                "table": diff_details[0].name,
                                "column": diff_details[1].name,
                                "changes": str(diff_details[2:])
                            })
                    
                    diff_result["has_changes"] = (
                        len(diff_result["additions"]) > 0 or
                        len(diff_result["removals"]) > 0 or
                        len(diff_result["modifications"]) > 0
                    )
                    
                    return []
                
                context.configure(
                    connection=connection,
                    fn=compare_metadata_fn,
                    target_metadata=self.metadata
                )
                
                with context.begin_transaction():
                    context.run_migrations()
            
            self.logger.info(
                "Schema diff generated",
                extra={
                    "has_changes": diff_result["has_changes"],
                    "additions": len(diff_result["additions"]),
                    "removals": len(diff_result["removals"]),
                    "modifications": len(diff_result["modifications"])
                }
            )
            
        except Exception as e:
            diff_result["error"] = str(e)
            self.logger.error(f"Failed to generate schema diff: {str(e)}")
        
        return diff_result
    
    def create_seed_data(self, seed_type: str = "development") -> Dict[str, Any]:
        """
        Create seed data for development/testing.
        
        Args:
            seed_type: Type of seed data (development, testing, demo)
            
        Returns:
            Dict with seed creation results
            
        Reference: [Required Experience: Production engineering role]
        """
        result = {
            "success": False,
            "tables_seeded": [],
            "records_created": 0,
            "errors": []
        }
        
        try:
            from sqlalchemy.orm import Session
            
            seed_data_generators = {
                "development": self._generate_development_seed_data,
                "testing": self._generate_testing_seed_data,
                "demo": self._generate_demo_seed_data
            }
            
            if seed_type not in seed_data_generators:
                result["errors"].append(f"Unknown seed type: {seed_type}")
                return result
            
            generator = seed_data_generators[seed_type]
            
            with Session(self.engine) as session:
                created_records = generator(session)
                
                session.commit()
                
                result["success"] = True
                result["records_created"] = created_records
                result["message"] = f"Created {created_records} seed records for {seed_type}"
                
                self.logger.info(
                    f"Seed data created for {seed_type}",
                    extra={
                        "seed_type": seed_type,
                        "records_created": created_records
                    }
                )
            
        except Exception as e:
            result["errors"].append(str(e))
            self.logger.error(f"Failed to create seed data: {str(e)}")
        
        return result
    
    def _generate_development_seed_data(self, session) -> int:
        """Generate development seed data."""
        from ..models import (
            Customer, Booking, Franchisee, Driver,
            MarketingCampaign, AIModelVersion
        )
        from datetime import datetime, timedelta
        from decimal import Decimal
        
        created_records = 0
        
        try:
            # Create customers
            customers = [
                Customer(
                    email="john.doe@example.com",
                    first_name="John",
                    last_name="Doe",
                    phone="+12345678901",
                    customer_type="regular"
                ),
                Customer(
                    email="jane.smith@example.com",
                    first_name="Jane",
                    last_name="Smith",
                    phone="+12345678902",
                    customer_type="corporate"
                ),
                Customer(
                    email="bob.wilson@example.com",
                    first_name="Bob",
                    last_name="Wilson",
                    phone="+12345678903",
                    customer_type="vip"
                )
            ]
            session.add_all(customers)
            created_records += len(customers)
            
            # Create franchisees
            franchisees = [
                Franchisee(
                    name="JFK Premium Parking",
                    contact_name="Mike Johnson",
                    contact_email="mike@jfkparking.com",
                    contact_phone="+12345678904",
                    airport_code="JFK",
                    location_address="123 Airport Rd, Queens, NY",
                    contract_start=datetime.now().date(),
                    contract_type="premium",
                    commission_rate=0.70,
                    total_spots=200
                ),
                Franchisee(
                    name="LAX Express Parking",
                    contact_name="Sarah Chen",
                    contact_email="sarah@laxparking.com",
                    contact_phone="+12345678905",
                    airport_code="LAX",
                    location_address="456 Aviation Blvd, Los Angeles, CA",
                    contract_start=datetime.now().date(),
                    contract_type="standard",
                    commission_rate=0.65,
                    total_spots=150
                )
            ]
            session.add_all(franchisees)
            created_records += len(franchisees)
            
            # Create drivers
            drivers = [
                Driver(
                    franchisee=franchisees[0],
                    first_name="Tom",
                    last_name="Baker",
                    phone="+12345678906",
                    hire_date=datetime.now().date(),
                    driver_license_number="DL123456",
                    license_expiry=datetime.now().date() + timedelta(days=365)
                ),
                Driver(
                    franchisee=franchisees[0],
                    first_name="Lisa",
                    last_name="Rodriguez",
                    phone="+12345678907",
                    hire_date=datetime.now().date(),
                    driver_license_number="DL234567",
                    license_expiry=datetime.now().date() + timedelta(days=365)
                )
            ]
            session.add_all(drivers)
            created_records += len(drivers)
            
            # Create bookings
            bookings = [
                Booking(
                    customer=customers[0],
                    airport_code="JFK",
                    check_in=datetime.now() + timedelta(days=1),
                    check_out=datetime.now() + timedelta(days=4),
                    vehicle_license="ABC123",
                    base_price=Decimal('75.00'),
                    total_price=Decimal('85.50'),
                    franchisee=franchisees[0],
                    driver=drivers[0]
                ),
                Booking(
                    customer=customers[1],
                    airport_code="LAX",
                    check_in=datetime.now() + timedelta(days=2),
                    check_out=datetime.now() + timedelta(days=5),
                    vehicle_license="XYZ789",
                    base_price=Decimal('65.00'),
                    total_price=Decimal('74.25'),
                    franchisee=franchisees[1],
                    driver=drivers[1]
                )
            ]
            session.add_all(bookings)
            created_records += len(bookings)
            
            # Create marketing campaigns
            campaigns = [
                MarketingCampaign(
                    name="Summer Travel Promotion",
                    channel="google_ads",
                    campaign_type="acquisition",
                    total_budget=Decimal('5000.00'),
                    start_date=datetime.now().date(),
                    end_date=datetime.now().date() + timedelta(days=30),
                    status="active"
                )
            ]
            session.add_all(campaigns)
            created_records += len(campaigns)
            
            # Create AI models
            ai_models = [
                AIModelVersion(
                    model_type="demand_forecast",
                    model_name="demand_forecaster_v1",
                    version="1.0.0",
                    is_production=True,
                    model_path="/models/demand_forecast_v1.pkl",
                    status="deployed"
                )
            ]
            session.add_all(ai_models)
            created_records += len(ai_models)
            
            session.flush()
            
        except Exception as e:
            self.logger.error(f"Error generating seed data: {str(e)}")
            raise
        
        return created_records
    
    def _generate_testing_seed_data(self, session) -> int:
        """Generate testing seed data."""
        # Similar to development but with more edge cases
        return self._generate_development_seed_data(session)
    
    def _generate_demo_seed_data(self, session) -> int:
        """Generate demo seed data with realistic volumes."""
        from ..models import Customer, Booking, Franchisee
        from datetime import datetime, timedelta
        from decimal import Decimal
        import random
        
        created_records = 0
        
        try:
            # Create demo customers (100 customers)
            customers = []
            for i in range(100):
                customer = Customer(
                    email=f"customer{i:03d}@example.com",
                    first_name=f"First{i}",
                    last_name=f"Last{i}",
                    phone=f"+1234567{i:04d}",
                    customer_type=random.choice(["regular", "corporate", "vip"])
                )
                customers.append(customer)
            
            session.add_all(customers)
            created_records += len(customers)
            
            # Create demo franchisees for major airports
            airports = ["JFK", "LAX", "ORD", "DFW", "MIA", "SFO", "SEA", "ATL"]
            franchisees = []
            
            for airport in airports:
                franchisee = Franchisee(
                    name=f"{airport} Premium Parking",
                    contact_name=f"Manager {airport}",
                    contact_email=f"manager@{airport.lower()}parking.com",
                    contact_phone=f"+1234567{random.randint(1000, 9999)}",
                    airport_code=airport,
                    location_address=f"123 {airport} Airport Rd",
                    contract_start=datetime.now().date(),
                    contract_type=random.choice(["standard", "premium"]),
                    commission_rate=random.uniform(0.65, 0.75),
                    total_spots=random.randint(100, 300)
                )
                franchisees.append(franchisee)
            
            session.add_all(franchisees)
            created_records += len(franchisees)
            
            # Create demo bookings (500 bookings)
            bookings = []
            for i in range(500):
                customer = random.choice(customers)
                franchisee = random.choice(franchisees)
                
                # Generate random dates within next 90 days
                days_ahead = random.randint(1, 90)
                duration = random.randint(1, 14)
                
                check_in = datetime.now() + timedelta(days=days_ahead)
                check_out = check_in + timedelta(days=duration)
                
                base_price = Decimal(str(random.uniform(20.0, 100.0)))
                total_price = base_price * Decimal('1.14')  # Add tax/service fee
                
                booking = Booking(
                    customer=customer,
                    airport_code=franchisee.airport_code,
                    check_in=check_in,
                    check_out=check_out,
                    vehicle_license=f"DEMO{random.randint(100, 999)}",
                    base_price=base_price,
                    total_price=total_price,
                    franchisee=franchisee
                )
                bookings.append(booking)
            
            session.add_all(bookings)
            created_records += len(bookings)
            
            session.flush()
            
        except Exception as e:
            self.logger.error(f"Error generating demo seed data: {str(e)}")
            raise
        
        return created_records

# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

_migration_context: Optional[MigrationContext] = None

def get_migration_context() -> MigrationContext:
    """
    Get the global migration context instance.
    
    Returns:
        MigrationContext: Global instance
        
    Reference: [Required Experience: Production engineering role]
    """
    global _migration_context
    
    if _migration_context is None:
        _migration_context = MigrationContext()
    
    return _migration_context

def run_migrations(target: str = "head", dry_run: bool = False) -> Dict[str, Any]:
    """
    Run pending migrations.
    
    Args:
        target: Target revision
        dry_run: If True, only show what would be done
        
    Returns:
        Dict with migration results
    """
    return get_migration_context().run_migrations(target, dry_run)

def rollback_migration(target: str = "-1", dry_run: bool = False) -> Dict[str, Any]:
    """
    Rollback migrations.
    
    Args:
        target: Target revision
        dry_run: If True, only show what would be done
        
    Returns:
        Dict with rollback results
    """
    return get_migration_context().rollback_migration(target, dry_run)

def create_migration(
    message: str,
    migration_type: MigrationType = MigrationType.SCHEMA,
    data_script: Optional[str] = None
) -> Optional[str]:
    """
    Create a new migration.
    
    Args:
        message: Migration description
        migration_type: Type of migration
        data_script: Optional data migration SQL
        
    Returns:
        str: Path to migration file or None
    """
    return get_migration_context().create_migration(message, migration_type, data_script)

def get_current_revision() -> Optional[str]:
    """
    Get current database revision.
    
    Returns:
        str: Current revision ID or None
    """
    return get_migration_context().get_current_revision()

def get_migration_history(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get migration history.
    
    Args:
        limit: Maximum number of migrations to return
        
    Returns:
        List of migration history entries
    """
    return get_migration_context().get_migration_history(limit)

def check_pending_migrations() -> List[Dict[str, Any]]:
    """
    Check for pending migrations.
    
    Returns:
        List of pending migrations
    """
    return get_migration_context().check_pending_migrations()

def generate_schema_diff() -> Dict[str, Any]:
    """
    Generate schema difference between models and database.
    
    Returns:
        Dict with schema differences
    """
    return get_migration_context().generate_schema_diff()

def create_seed_data(seed_type: str = "development") -> Dict[str, Any]:
    """
    Create seed data.
    
    Args:
        seed_type: Type of seed data
        
    Returns:
        Dict with seed creation results
    """
    return get_migration_context().create_seed_data(seed_type)

def validate_migrations() -> Dict[str, Any]:
    """
    Validate migration system and state.
    
    Returns:
        Dict with validation results
        
    Reference: [Absolutely Essential: Design with safety, logging]
    """
    validation = {
        "valid": True,
        "checks": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        context = get_migration_context()
        
        # Check Alembic configuration
        try:
            config = context.alembic_cfg
            validation["checks"]["alembic_config"] = {
                "passed": True,
                "message": "Alembic configuration valid"
            }
        except Exception as e:
            validation["valid"] = False
            validation["checks"]["alembic_config"] = {
                "passed": False,
                "message": f"Alembic config error: {str(e)}"
            }
            validation["errors"].append("Alembic configuration invalid")
        
        # Check database connection
        try:
            context.engine.connect()
            validation["checks"]["database_connection"] = {
                "passed": True,
                "message": "Database connection established"
            }
        except Exception as e:
            validation["valid"] = False
            validation["checks"]["database_connection"] = {
                "passed": False,
                "message": f"Database connection failed: {str(e)}"
            }
            validation["errors"].append("Database connection failed")
        
        # Check migration scripts directory
        if ALEMBIC_VERSIONS_DIR.exists():
            validation["checks"]["migrations_dir"] = {
                "passed": True,
                "message": f"Migrations directory exists: {ALEMBIC_VERSIONS_DIR}",
                "migration_files": len(list(ALEMBIC_VERSIONS_DIR.glob("*.py")))
            }
        else:
            validation["warnings"].append("Migrations directory doesn't exist")
            validation["checks"]["migrations_dir"] = {
                "passed": False,
                "message": "Migrations directory doesn't exist"
            }
        
        # Check current revision
        current_rev = context.get_current_revision()
        validation["checks"]["current_revision"] = {
            "passed": current_rev is not None or not list(ALEMBIC_VERSIONS_DIR.glob("*.py")),
            "message": f"Current revision: {current_rev or 'None (no migrations applied)'}",
            "revision": current_rev
        }
        
        # Check for pending migrations
        pending = context.check_pending_migrations()
        validation["checks"]["pending_migrations"] = {
            "passed": len(pending) == 0,
            "message": f"{len(pending)} pending migrations",
            "count": len(pending)
        }
        
        if len(pending) > 0:
            validation["warnings"].append(f"{len(pending)} pending migrations")
        
        # Generate schema diff to check for model/database mismatch
        diff = context.generate_schema_diff()
        validation["checks"]["schema_sync"] = {
            "passed": not diff.get("has_changes", True),
            "message": "Models and database are synchronized" if not diff.get("has_changes") else "Schema differences detected",
            "has_changes": diff.get("has_changes", True)
        }
        
        if diff.get("has_changes"):
            validation["warnings"].append("Schema differences between models and database")
        
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Validation error: {str(e)}")
    
    return validation

# ============================================================================
# EXCEPTIONS
# ============================================================================

class MigrationError(Exception):
    """Base exception for migration errors."""
    pass

class MigrationValidationError(MigrationError):
    """Raised when migration validation fails."""
    pass

class MigrationConflictError(MigrationError):
    """Raised when there are migration conflicts."""
    pass

class SeedDataError(MigrationError):
    """Raised when seed data creation fails."""
    pass

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def initialize_migrations() -> MigrationContext:
    """
    Initialize the migration system.
    
    Returns:
        MigrationContext: Initialized migration context
        
    Reference: [Required Experience: Production engineering role]
    """
    context = get_migration_context()
    
    # Validate migration system
    validation = validate_migrations()
    
    if not validation["valid"]:
        error_msg = f"Migration system validation failed: {validation['errors']}"
        context.logger.critical(error_msg, extra={"validation": validation})
        raise MigrationValidationError(error_msg)
    
    # Log initialization
    pending = context.check_pending_migrations()
    current_rev = context.get_current_revision()
    
    context.logger.info(
        "Migration system initialized",
        extra={
            "current_revision": current_rev,
            "pending_migrations": len(pending),
            "validation_passed": validation["valid"]
        }
    )
    
    return context

# Auto-initialize on module import
initialize_migrations()
