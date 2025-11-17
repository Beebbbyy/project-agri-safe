"""
Spark Session Management Utilities

Provides reusable SparkSession configuration for all ETL jobs.
"""

import os
from typing import Optional
from pyspark.sql import SparkSession
from loguru import logger


def get_spark_session(
    app_name: str = "AgriSafe-ETL",
    master: Optional[str] = None,
    config: Optional[dict] = None
) -> SparkSession:
    """
    Create or get existing Spark session with optimal configuration.

    Args:
        app_name: Name of the Spark application
        master: Spark master URL (default: local[*])
        config: Additional Spark configuration options

    Returns:
        SparkSession: Configured Spark session
    """
    if master is None:
        master = os.getenv("SPARK_MASTER", "local[*]")

    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master)

    # Default configurations
    default_config = {
        # Memory settings
        "spark.driver.memory": "2g",
        "spark.executor.memory": "2g",

        # PostgreSQL JDBC driver
        "spark.jars.packages": "org.postgresql:postgresql:42.7.1",

        # SQL configurations
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",

        # Serialization
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",

        # Shuffle settings
        "spark.sql.shuffle.partitions": "10",

        # Performance
        "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10MB
    }

    # Merge with user-provided config
    if config:
        default_config.update(config)

    # Apply all configurations
    for key, value in default_config.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    # Set log level
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Spark session created: {app_name}")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Master: {master}")

    return spark


def get_postgres_jdbc_url() -> str:
    """
    Get PostgreSQL JDBC URL from environment variables.

    Returns:
        str: JDBC connection URL
    """
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "agrisafe_db")

    return f"jdbc:postgresql://{host}:{port}/{database}"


def get_postgres_properties() -> dict:
    """
    Get PostgreSQL connection properties.

    Returns:
        dict: Connection properties including credentials
    """
    return {
        "user": os.getenv("POSTGRES_USER", "agrisafe"),
        "password": os.getenv("POSTGRES_PASSWORD", "agrisafe_password"),
        "driver": "org.postgresql.Driver",
        "stringtype": "unspecified",  # Handle VARCHAR properly
    }


def read_postgres_table(
    spark: SparkSession,
    table: str,
    predicates: Optional[list] = None
):
    """
    Read a PostgreSQL table into a Spark DataFrame.

    Args:
        spark: Spark session
        table: Table name or SQL query (wrapped in parentheses)
        predicates: List of WHERE clause predicates for parallel reading

    Returns:
        DataFrame: Spark DataFrame containing table data
    """
    jdbc_url = get_postgres_jdbc_url()
    properties = get_postgres_properties()

    logger.info(f"Reading table: {table}")

    reader = spark.read.jdbc(
        url=jdbc_url,
        table=table,
        properties=properties
    )

    # Use predicates for parallel reading if provided
    if predicates:
        reader = spark.read.jdbc(
            url=jdbc_url,
            table=table,
            properties=properties,
            predicates=predicates
        )

    df = reader.load()
    logger.info(f"Loaded {df.count()} rows from {table}")

    return df


def write_postgres_table(
    df,
    table: str,
    mode: str = "append"
):
    """
    Write a Spark DataFrame to PostgreSQL table.

    Args:
        df: Spark DataFrame to write
        table: Target table name
        mode: Write mode (append, overwrite, ignore, error)
    """
    jdbc_url = get_postgres_jdbc_url()
    properties = get_postgres_properties()

    logger.info(f"Writing to table: {table} (mode: {mode})")

    df.write.jdbc(
        url=jdbc_url,
        table=table,
        mode=mode,
        properties=properties
    )

    logger.info(f"Successfully wrote {df.count()} rows to {table}")


def stop_spark_session(spark: SparkSession):
    """
    Stop Spark session and release resources.

    Args:
        spark: Spark session to stop
    """
    logger.info("Stopping Spark session...")
    spark.stop()
    logger.info("Spark session stopped")
