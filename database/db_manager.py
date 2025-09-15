"""Database connection and session management."""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from .config import DatabaseConfig


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager."""
        self.config = config or DatabaseConfig.from_env()
        self.pool: Optional[SimpleConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                1,  # min connections
                10,  # max connections
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            logger.info(f"Database connection pool initialized for {self.config.database}")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """Get a database connection from the pool."""
        connection = None
        try:
            connection = self.pool.getconn()
            yield connection
            connection.commit()
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection:
                self.pool.putconn(connection)

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Generator:
        """Get a database cursor."""
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = True):
        """Execute a query and optionally fetch results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return cursor.rowcount

    def execute_many(self, query: str, params_list: list):
        """Execute a query with multiple parameter sets."""
        with self.get_cursor(dict_cursor=False) as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result['?column?'] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def create_database(self):
        """Create database if it doesn't exist."""
        # Connect to default postgres database
        temp_config = DatabaseConfig(
            host=self.config.host,
            port=self.config.port,
            database="postgres",
            user=self.config.user,
            password=self.config.password
        )

        try:
            conn = psycopg2.connect(
                host=temp_config.host,
                port=temp_config.port,
                database=temp_config.database,
                user=temp_config.user,
                password=temp_config.password
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config.database,)
            )
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(f"CREATE DATABASE {self.config.database}")
                logger.info(f"Database {self.config.database} created")
            else:
                logger.info(f"Database {self.config.database} already exists")

            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Failed to create database: {e}")
            raise

    def execute_schema(self, schema_path: str):
        """Execute SQL schema file."""
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(schema_sql)
                logger.info("Schema executed successfully")
            except psycopg2.Error as e:
                logger.error(f"Failed to execute schema: {e}")
                raise
            finally:
                cursor.close()

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()