"""
Database connection utilities for Project Agri-Safe
"""

import os
from contextlib import contextmanager
from typing import Optional, Generator
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConnection:
    """
    Database connection manager with connection pooling
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_conn: int = 1,
        max_conn: int = 10
    ):
        """
        Initialize database connection pool

        Args:
            host: Database host (defaults to env var POSTGRES_HOST)
            port: Database port (defaults to env var POSTGRES_PORT)
            database: Database name (defaults to env var POSTGRES_DB)
            user: Database user (defaults to env var POSTGRES_USER)
            password: Database password (defaults to env var POSTGRES_PASSWORD)
            min_conn: Minimum number of connections in pool
            max_conn: Maximum number of connections in pool
        """
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', '5432'))
        self.database = database or os.getenv('POSTGRES_DB', 'agrisafe_db')
        self.user = user or os.getenv('POSTGRES_USER', 'agrisafe')
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'agrisafe_password')

        self.pool = SimpleConnectionPool(
            min_conn,
            max_conn,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a connection from the pool

        Yields:
            psycopg2 connection object
        """
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Generator:
        """
        Get a cursor from a pooled connection

        Args:
            dict_cursor: If True, use RealDictCursor for dict-like results

        Yields:
            psycopg2 cursor object
        """
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = True):
        """
        Execute a query and optionally fetch results

        Args:
            query: SQL query string
            params: Query parameters (tuple)
            fetch: Whether to fetch results

        Returns:
            Query results if fetch=True, otherwise None
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None

    def execute_many(self, query: str, params_list: list):
        """
        Execute a query with multiple parameter sets

        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)

    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()


# Global database connection instance
_db_instance: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """
    Get or create the global database connection instance

    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance
