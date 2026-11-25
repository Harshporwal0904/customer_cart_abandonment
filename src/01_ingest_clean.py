"""
01_ingest_clean.py:
Ingests raw Retailrocket events.csv, applies explicit schema, cleans missing/duplicate data,
handles optional sampling by visitorid, and writes processed Parquet.
"""

import argparse
import logging
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
)
import pyspark.sql.functions as F


def setup_logger() -> logging.Logger:
    """Configures and returns a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("01_ingest_clean")


def get_spark_session(app_name: str = "01_ingest_clean") -> SparkSession:
    """Initializes and returns a SparkSession with memory and partition configurations."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def get_raw_schema() -> StructType:
    """Returns the explicit StructType schema for events.csv."""
    return StructType([
        StructField("timestamp", LongType(), nullable=False),
        StructField("visitorid", LongType(), nullable=False),
        StructField("event", StringType(), nullable=False),
        StructField("itemid", LongType(), nullable=False),
        StructField("transactionid", StringType(), nullable=True),
    ])


def ingest_and_clean_data(
    spark: SparkSession,
    input_path: str,
    sample: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> DataFrame:
    """
    Reads events.csv, cleans data, converts timestamp from ms to timestamp type,
    applies visitorid-level sampling if specified, and drops duplicates/nulls.
    """
    if logger is None:
        logger = logging.getLogger("01_ingest_clean")

    logger.info("Loading raw data from %s", input_path)
    schema = get_raw_schema()
    raw_df = (
        spark.read.option("header", "true")
        .schema(schema)
        .csv(input_path)
    )

    initial_count = raw_df.count()
    logger.info("Initial raw event count: %d", initial_count)

    # Convert ms timestamp to TimestampType
    # Retailrocket timestamp is epoch in milliseconds: cast(timestamp / 1000 as timestamp)
    df = raw_df.withColumn("timestamp", F.to_timestamp(F.col("timestamp") / 1000.0))

    # Drop nulls on critical columns: visitorid, timestamp, event, itemid
    # Note: transactionid is naturally null for 'view' and 'addtocart' events
    df_clean = df.dropna(subset=["visitorid", "timestamp", "event", "itemid"])

    # Filter to valid events only
    df_clean = df_clean.filter(F.col("event").isin("view", "addtocart", "transaction"))

    # Drop exact duplicates across event fields
    df_clean = df_clean.dropDuplicates(["visitorid", "timestamp", "event", "itemid"])

    # Optional visitor-based sampling
    if sample is not None and 0.0 < sample < 1.0:
        logger.info("Sampling %0.2f fraction of unique visitors", sample)
        unique_visitors = df_clean.select("visitorid").distinct().sample(fraction=sample, seed=42)
        df_clean = df_clean.join(unique_visitors, on="visitorid", how="inner")

    cleaned_count = df_clean.count()
    unique_visitors_count = df_clean.select("visitorid").distinct().count()
    logger.info("Cleaned event count: %d (Unique visitors: %d)", cleaned_count, unique_visitors_count)

    # Print basic event distribution
    logger.info("Event counts by event type:")
    event_counts = df_clean.groupBy("event").count().collect()
    for row in event_counts:
        logger.info("  %s: %d", row["event"], row["count"])

    return df_clean


def main() -> None:
    """CLI entrypoint for 01_ingest_clean."""
    parser = argparse.ArgumentParser(description="Ingest and clean Retailrocket events dataset.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of visitors to sample (e.g. 0.1 for 10%%)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/events.csv",
        help="Path to raw events.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clean_events.parquet",
        help="Output path for cleaned Parquet dataset",
    )
    args = parser.parse_args()

    logger = setup_logger()
    spark = get_spark_session("01_ingest_clean")

    try:
        cleaned_df = ingest_and_clean_data(spark, args.input, args.sample, logger)
        logger.info("Writing cleaned events to Parquet: %s", args.output)
        cleaned_df.write.mode("overwrite").parquet(args.output)
        logger.info("Successfully completed 01_ingest_clean.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
