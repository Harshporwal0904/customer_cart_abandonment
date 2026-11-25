"""
02_sessionize_label.py:
Performs 30-minute inactivity sessionization on cleaned events using Window functions.
Labels cart sessions and computes the abandonment target.
Writes session-level and session-event level datasets to Parquet.
"""

import argparse
import logging
from typing import Optional
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F


def setup_logger() -> logging.Logger:
    """Configures and returns a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("02_sessionize_label")


def get_spark_session(app_name: str = "02_sessionize_label") -> SparkSession:
    """Initializes and returns a SparkSession with memory and partition configurations."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def sessionize_events(df: DataFrame, logger: Optional[logging.Logger] = None) -> DataFrame:
    """
    Groups events into distinct sessions per visitor based on a 30-minute (1800s) inactivity gap.

    Window Logic Explanation:
    1. Window `w_visitor`: Partitions by `visitorid` ordered by `timestamp` ascending.
    2. `prev_ts`: LAG(timestamp, 1) over `w_visitor` fetches the previous event's timestamp.
    3. `is_new_session`: If the elapsed seconds between current timestamp and `prev_ts` exceeds
       1800 seconds (or if `prev_ts` is null, i.e., visitor's first event), flag 1, else 0.
    4. `visitor_session_num`: SUM(is_new_session) over `w_visitor_accum` (unboundedPreceding to
       currentRow) creates a monotonically increasing session index (0, 1, 2, ...) per visitor.
    5. `session_id`: Concatenate visitorid and visitor_session_num (e.g. "{visitorid}_{visitor_session_num}")
       for a globally unique session identifier.
    """
    if logger is None:
        logger = logging.getLogger("02_sessionize_label")

    logger.info("Starting sessionization with 30-minute inactivity window...")

    w_visitor = Window.partitionBy("visitorid").orderBy("timestamp")
    w_visitor_accum = (
        Window.partitionBy("visitorid")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    # Calculate lag timestamp in seconds
    df_lag = df.withColumn("prev_ts", F.lag("timestamp").over(w_visitor))
    gap_seconds = F.col("timestamp").cast("long") - F.col("prev_ts").cast("long")

    df_flagged = df_lag.withColumn(
        "is_new_session",
        F.when(F.col("prev_ts").isNull() | (gap_seconds > 1800), 1).otherwise(0),
    )

    df_sessionized = df_flagged.withColumn(
        "visitor_session_num",
        F.sum("is_new_session").over(w_visitor_accum),
    ).withColumn(
        "session_id",
        F.concat(F.col("visitorid"), F.lit("_"), F.col("visitor_session_num")),
    ).drop("prev_ts", "is_new_session")

    return df_sessionized


def label_cart_sessions(
    df_sessionized: DataFrame,
    logger: Optional[logging.Logger] = None,
) -> DataFrame:
    """
    Filters and labels cart sessions:
    - Cart session: Session with at least one 'addtocart' event.
    - Prediction point: The first 'addtocart' in each cart session (first_cart_ts).
    - abandoned: 1 if the session has no transaction at or after first_cart_ts, else 0.
    """
    if logger is None:
        logger = logging.getLogger("02_sessionize_label")

    logger.info("Labeling cart sessions and identifying prediction points...")

    # For each session, compute session boundaries and event specific timestamps
    # First cart timestamp: min timestamp where event == 'addtocart'
    # Transactions at or after first cart: min timestamp where event == 'transaction' and timestamp >= first_cart_ts
    session_summary = df_sessionized.groupBy("session_id", "visitorid", "visitor_session_num").agg(
        F.min("timestamp").alias("session_start_ts"),
        F.max("timestamp").alias("session_end_ts"),
        F.min(
            F.when(F.col("event") == "addtocart", F.col("timestamp"))
        ).alias("first_cart_ts"),
        F.min(
            F.when(F.col("event") == "transaction", F.col("timestamp"))
        ).alias("first_trans_ts"),
    )

    # Filter to Cart Sessions only (sessions having at least one addtocart event)
    cart_sessions = session_summary.filter(F.col("first_cart_ts").isNotNull())

    # Label abandoned: 1 if no transaction at or after first_cart_ts, else 0
    # Also attach transaction_at_or_after_cart
    cart_sessions = cart_sessions.withColumn(
        "has_trans_after_cart",
        F.when(
            F.col("first_trans_ts").isNotNull() & (F.col("first_trans_ts") >= F.col("first_cart_ts")),
            1,
        ).otherwise(0),
    ).withColumn(
        "abandoned",
        F.when(F.col("has_trans_after_cart") == 1, 0).otherwise(1),
    )

    return cart_sessions


def main() -> None:
    """CLI entrypoint for 02_sessionize_label."""
    parser = argparse.ArgumentParser(description="Sessionize events and label cart abandonment.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of visitors to sample (if input is raw or for consistency)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/clean_events.parquet",
        help="Path to cleaned events Parquet dataset",
    )
    parser.add_argument(
        "--output-events",
        type=str,
        default="data/processed/sessionized_events.parquet",
        help="Output path for sessionized events Parquet",
    )
    parser.add_argument(
        "--output-sessions",
        type=str,
        default="data/processed/cart_sessions.parquet",
        help="Output path for labeled cart sessions Parquet",
    )
    args = parser.parse_args()

    logger = setup_logger()
    spark = get_spark_session("02_sessionize_label")

    try:
        logger.info("Reading input events from %s", args.input)
        df = spark.read.parquet(args.input)

        if args.sample is not None and 0.0 < args.sample < 1.0:
            logger.info("Sampling %0.2f fraction of visitors", args.sample)
            unique_visitors = df.select("visitorid").distinct().sample(fraction=args.sample, seed=42)
            df = df.join(unique_visitors, on="visitorid", how="inner")

        df_sessionized = sessionize_events(df, logger)
        logger.info("Writing sessionized events to %s", args.output_events)
        df_sessionized.write.mode("overwrite").parquet(args.output_events)

        cart_sessions_df = label_cart_sessions(df_sessionized, logger)
        total_sessions = df_sessionized.select("session_id").distinct().count()
        cart_count = cart_sessions_df.count()
        abandoned_count = cart_sessions_df.filter(F.col("abandoned") == 1).count()
        abandonment_rate = (abandoned_count / cart_count * 100.0) if cart_count > 0 else 0.0

        logger.info("Sessionization Summary:")
        logger.info("  Total sessions: %d", total_sessions)
        logger.info("  Cart sessions: %d", cart_count)
        logger.info("  Abandoned sessions: %d (%.2f%%)", abandoned_count, abandonment_rate)
        logger.info("  Converted sessions: %d (%.2f%%)", cart_count - abandoned_count, 100.0 - abandonment_rate)

        logger.info("Writing labeled cart sessions to %s", args.output_sessions)
        cart_sessions_df.write.mode("overwrite").parquet(args.output_sessions)
        logger.info("Successfully completed 02_sessionize_label.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

