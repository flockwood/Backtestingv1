#!/usr/bin/env python3
"""Initialize the backtesting database."""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseConfig, DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize the database with schema."""

    print("="*60)
    print("DATABASE INITIALIZATION")
    print("="*60)

    # Load configuration
    config = DatabaseConfig.from_env()
    print(f"Database: {config.database}")
    print(f"Host: {config.host}:{config.port}")
    print(f"User: {config.user}")

    try:
        # Create database manager
        db_manager = DatabaseManager(config)

        # Create database if it doesn't exist
        print("\nCreating database if not exists...")
        db_manager.create_database()

        # Test connection
        print("Testing connection...")
        if db_manager.test_connection():
            print("✓ Connection successful")
        else:
            print("✗ Connection failed")
            return False

        # Execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        print(f"\nExecuting schema from {schema_path}...")
        db_manager.execute_schema(str(schema_path))
        print("✓ Schema executed successfully")

        # Verify tables
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()

        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table['table_name']}")

        print("\n" + "="*60)
        print("Database initialization complete!")
        print("="*60)

        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is installed and running")
        print("2. Check your database credentials")
        print("3. Set environment variables if needed:")
        print("   export DB_HOST=localhost")
        print("   export DB_PORT=5432")
        print("   export DB_NAME=backtesting_db")
        print("   export DB_USER=your_username")
        print("   export DB_PASSWORD=your_password")
        return False

    finally:
        if 'db_manager' in locals():
            db_manager.close()


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)