
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

Row = dict[str, Any]
Params = Optional[tuple[Any, ...]]


class PostgresClient:
    """
    Thread-safe PostgreSQL client backed by a ``ThreadedConnectionPool``.
    All public methods are classmethods so the client can be used without
    instantiation, matching the original API exactly.

    Pool lifecycle
    One pool is created per process on first use.  Double-checked locking
    prevents concurrent threads from racing to initialise duplicate pools.
    """

    _pool: Optional[ThreadedConnectionPool] = None
    _lock: threading.Lock = threading.Lock()

    
    @classmethod
    def _initialize_pool(cls) -> None:
        """
        Create the connection pool. Called once per process.

        Uses ``ThreadedConnectionPool`` instead of ``SimpleConnectionPool``
        because psycopg2 documents ``SimpleConnectionPool`` as *not*
        thread-safe — concurrent Celery workers in the same process will
        corrupt it under load.
        """
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/ml_platform",
        )

        min_conn = int(os.getenv("DB_POOL_MIN_CONN", "1"))
        max_conn = int(os.getenv("DB_POOL_MAX_CONN", "20"))

        try:
            cls._pool = ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                dsn=db_url,
            )
            logger.info(
                "PostgreSQL connection pool initialised",
                extra={"min_conn": min_conn, "max_conn": max_conn},
            )
        except psycopg2.OperationalError as exc:
            raise RuntimeError(
                "Failed to connect to PostgreSQL. "
                "Check DATABASE_URL and network connectivity."
            ) from exc

    @classmethod
    def _get_pool(cls) -> ThreadedConnectionPool:
        """Return the pool, initialising it on first call (double-checked lock)."""
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    cls._initialize_pool()
        return cls._pool  # type: ignore[return-value]

    @classmethod
    @contextmanager
    def _connection(cls) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager that acquires a connection from the pool and
        guarantees it is returned — even if the caller raises.
        """
        conn = cls._get_pool().getconn()
        try:
            yield conn
        finally:
            cls._get_pool().putconn(conn)

    @classmethod
    def close_pool(cls) -> None:
        """
        Close all connections in the pool.

        Call this on application shutdown to release server-side resources
        cleanly (important in test teardown and Lambda-style short-lived workers).
        """
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
            logger.info("PostgreSQL connection pool closed")


    @classmethod
    def ping(cls) -> bool:
        """Return ``True`` if the database is reachable, ``False`` otherwise."""
        try:
            with cls._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return True
        except Exception as exc:
            logger.error("PostgreSQL ping failed", extra={"error": str(exc)})
            return False

    @classmethod
    def query(
        cls,
        sql_str: str,
        params: Params = None,
    ) -> list[Row]:
        """
        Execute a read-only SQL statement and return all rows as dicts.
        """
        with cls._connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql_str, params)
                    return [dict(row) for row in cur.fetchall()]
            except Exception:
                conn.rollback()
                raise

    @classmethod
    def execute(
        cls,
        sql_str: str,
        params: Params = None,
        returning: bool = False,
    ) -> Optional[Row]:
        """
        Execute a write statement (INSERT / UPDATE / DELETE) with an optional
        ``RETURNING`` clause.
        """
        with cls._connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql_str, params)
                    result: Optional[Row] = None
                    if returning:
                        row = cur.fetchone()
                        result = dict(row) if row else None
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                logger.error("PostgreSQL execute failed", exc_info=True)
                raise

    

    @classmethod
    def insert(
        cls,
        table: str,
        data: dict[str, Any],
        returning: Optional[str] = None,
    ) -> Optional[Row]:
        """
        Insert a single row into *table*.
        """
        if not data:
            raise ValueError("data must not be empty")

        columns = [sql.Identifier(col) for col in data.keys()]
        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(data))

        stmt = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders})").format(
            table=sql.Identifier(table),
            columns=sql.SQL(", ").join(columns),
            placeholders=placeholders,
        )

        if returning:
            stmt = sql.SQL("{stmt} RETURNING {col}").format(
                stmt=stmt,
                col=sql.Identifier(returning),
            )

        return cls.execute(stmt.as_string(cls._get_pool().getconn()), tuple(data.values()), returning=bool(returning))

    @classmethod
    def bulk_insert(
        cls,
        table: str,
        rows: list[dict[str, Any]],
        page_size: int = 1000,
    ) -> None:
        """
        Insert many rows into *table* efficiently using ``execute_values``.

        ``execute_values`` sends all rows in a single multi-row ``VALUES``
        statement — dramatically faster than the original loop of N individual
        ``execute`` calls.
        """
        if not rows:
            return

        columns = list(rows[0].keys())

        # Validate all rows have the same columns to catch schema drift early.
        for i, row in enumerate(rows):
            if list(row.keys()) != columns:
                raise ValueError(
                    f"Row {i} has different keys than row 0. "
                    f"Expected {columns}, got {list(row.keys())}"
                )

        col_identifiers = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
        stmt = sql.SQL("INSERT INTO {table} ({columns}) VALUES %s").format(
            table=sql.Identifier(table),
            columns=col_identifiers,
        )

        # Rows as a list of tuples preserving column order.
        values = [tuple(row[col] for col in columns) for row in rows]

        with cls._connection() as conn:
            try:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        stmt.as_string(conn),
                        values,
                        page_size=page_size,
                    )
                conn.commit()
                logger.info(
                    "Bulk insert complete",
                    extra={"table": table, "row_count": len(rows)},
                )
            except Exception:
                conn.rollback()
                logger.error(
                    "Bulk insert failed",
                    extra={"table": table, "row_count": len(rows)},
                    exc_info=True,
                )
                raise

    @classmethod
    def upsert(
        cls,
        table: str,
        data: dict[str, Any],
        conflict_columns: list[str],
        returning: Optional[str] = None,
    ) -> Optional[Row]:
        """
        Insert a row or update it on conflict (``INSERT … ON CONFLICT DO UPDATE``).

        Common pattern for job metadata: insert the record if it doesn't exist,
        update it if the job_id already exists (idempotent re-submission).
        """
        if not data:
            raise ValueError("data must not be empty")
        if not conflict_columns:
            raise ValueError("conflict_columns must not be empty")

        columns = list(data.keys())
        update_columns = [c for c in columns if c not in conflict_columns]

        if not update_columns:
            raise ValueError(
                "All columns are conflict columns — nothing to update. "
                "Use insert() instead."
            )

        col_ids = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(columns))
        conflict_ids = sql.SQL(", ").join(sql.Identifier(c) for c in conflict_columns)
        update_set = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
            for c in update_columns
        )

        stmt = sql.SQL(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        ).format(
            table=sql.Identifier(table),
            columns=col_ids,
            placeholders=placeholders,
            conflict=conflict_ids,
            updates=update_set,
        )

        if returning:
            stmt = sql.SQL("{stmt} RETURNING {col}").format(
                stmt=stmt,
                col=sql.Identifier(returning),
            )

        return cls.execute(
            stmt.as_string(cls._get_pool().getconn()),
            tuple(data.values()),
            returning=bool(returning),
        )