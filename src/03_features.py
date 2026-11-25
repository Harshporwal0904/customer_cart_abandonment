"""
03_features.py:
Computes leakage-free predictive features for labeled cart sessions:
- views_before_cart: Count of 'view' events in the session strictly before or at first_cart_ts
- distinct_items_viewed: Count of distinct itemid viewed in session before or at first_cart_ts
- seconds_to_cart: Seconds from session_start_ts to first_cart_ts
- hour_of_day: Hour of first_cart_ts (0-23)
- day_of_week: Day of week of first_cart_ts (1=Sunday/2=Monday or Spark standard dayofweek)
- prior_sessions: Count of prior completed sessions by this visitor
- prior_purchases: Count of prior transactions across earlier sessions by this visitor

Data Leakage Prevention:
All intra-session features are computed ONLY from events occurring at or before the
prediction point (the timestamp of the visitor's first addtocart in that session).
Historical features (prior_sessions, prior_purchases) are strictly restricted to earlier
sessions (session_start_ts < current session_start_ts) so future activity is never accessed.
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
    return logging.getLogger("03_features")


def get_spark_session(app_name: str = "03_features") -> SparkSession:
    """Initializes and returns a SparkSession with memory and partition configurations."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def extract_features(
    cart_sessions_df: DataFrame,
    sessionized_events_df: DataFrame,
    logger: Optional[logging.Logger] = None,
) -> DataFrame:
    """
    Computes all features for cart sessions while ensuring strict temporal separation.
    """
    if logger is None:
        logger = logging.getLogger("03_features")

    logger.info("Computing leakage-free session and historical features...")

    # 1. Historical Visitor Features (earlier sessions only)
    # First, summarize each visitor's session start and whether it had transactions
    session_history = sessionized_events_df.groupBy("session_id", "visitorid").agg(
        F.min("timestamp").alias("session_start_time"),
        F.sum(F.when(F.col("event") == "transaction", 1).otherwise(0)).alias("session_transactions"),
    )

    # Use window over visitorid ordered by session_start_time
    # rowsBetween(Window.unboundedPreceding, -1) guarantees we only look at earlier sessions
    w_visitor_hist = (
        Window.partitionBy("visitorid")
        .orderBy("session_start_time")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    session_history = session_history.withColumn(
        "prior_sessions",
        F.coalesce(F.count("session_id").over(w_visitor_hist), F.lit(0)),
    ).withColumn(
        "prior_purchases",
        F.coalesce(F.sum("session_transactions").over(w_visitor_hist), F.lit(0)),
    ).select("session_id", "prior_sessions", "prior_purchases")

    # 2. Intra-Session Features (up to and including prediction point: first_cart_ts)
    # Join events with cart_sessions to get first_cart_ts
    cart_base = cart_sessions_df.select(
        "session_id",
        "visitorid",
        "session_start_ts",
        "first_cart_ts",
        "abandoned",
    )

    events_with_cart = sessionized_events_df.join(
        cart_base.select("session_id", "first_cart_ts"),
        on="session_id",
        how="inner",
    )

    # Filter events strictly <= first_cart_ts (no leakage of events after prediction point)
    events_before_cart = events_with_cart.filter(F.col("timestamp") <= F.col("first_cart_ts"))

    # Views before cart and distinct items viewed
    views_before_cart_df = (
        events_before_cart.filter(F.col("event") == "view")
        .groupBy("session_id")
        .agg(
            F.count("event").alias("views_before_cart"),
            F.countDistinct("itemid").alias("distinct_items_viewed"),
        )
    )

    # 3. Assemble all features together
    features_df = cart_base.join(views_before_cart_df, on="session_id", how="left")
    # Fill nulls for views_before_cart and distinct_items_viewed with 0
    features_df = features_df.fillna(
        {"views_before_cart": 0, "distinct_items_viewed": 0}
    )

    # Attach historical features
    features_df = features_df.join(session_history, on="session_id", how="left")
    features_df = features_df.fillna(
        {"prior_sessions": 0, "prior_purchases": 0}
    )

    # Time-based features
    # seconds_to_cart = first_cart_ts - session_start_ts
    features_df = features_df.withColumn(
        "seconds_to_cart",
        (F.col("first_cart_ts").cast("long") - F.col("session_start_ts").cast("long")).cast("double"),
    )

    # hour_of_day: 0-23
    features_df = features_df.withColumn("hour_of_day", F.hour(F.col("first_cart_ts")))

    # day_of_week: 1 (Sun) to 7 (Sat)
    features_df = features_df.withColumn("day_of_week", F.dayofweek(F.col("first_cart_ts")))

    # Cast integer feature columns to double for MLlib compatibility
    features_df = (
        features_df.withColumn("views_before_cart", F.col("views_before_cart").cast("double"))
        .withColumn("distinct_items_viewed", F.col("distinct_items_viewed").cast("double"))
        .withColumn("prior_sessions", F.col("prior_sessions").cast("double"))
        .withColumn("prior_purchases", F.col("prior_purchases").cast("double"))
        .withColumn("hour_of_day", F.col("hour_of_day").cast("double"))
        .withColumn("day_of_week", F.col("day_of_week").cast("double"))
        .withColumn("abandoned", F.col("abandoned").cast("double"))
    )

    return features_df


def main() -> None:
    """CLI entrypoint for 03_features."""
    parser = argparse.ArgumentParser(description="Extract leakage-free features for cart sessions.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional sampling fraction for visitors",
    )
    parser.add_argument(
        "--input-events",
        type=str,
        default="data/processed/sessionized_events.parquet",
        help="Path to sessionized events Parquet",
    )
    parser.add_argument(
        "--input-sessions",
        type=str,
        default="data/processed/cart_sessions.parquet",
        help="Path to labeled cart sessions Parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features.parquet",
        help="Output path for feature dataset Parquet",
    )
    args = parser.parse_args()

    logger = setup_logger()
    spark = get_spark_session("03_features")

    try:
        logger.info("Reading input datasets...")
        events_df = spark.read.parquet(args.input_events)
        cart_df = spark.read.parquet(args.input_sessions)

        if args.sample is not None and 0.0 < args.sample < 1.0:
            logger.info("Sampling %0.2f fraction of visitors", args.sample)
            sample_visitors = cart_df.select("visitorid").distinct().sample(fraction=args.sample, seed=42)
            cart_df = cart_df.join(sample_visitors, on="visitorid", how="inner")
            events_df = events_df.join(sample_visitors, on="visitorid", how="inner")

        features_df = extract_features(cart_df, events_df, logger)
        logger.info("Features schema:")
        features_df.printSchema()

        feature_count = features_df.count()
        logger.info("Total feature rows generated: %d", feature_count)

        logger.info("Writing features to %s", args.output)
        features_df.write.mode("overwrite").parquet(args.output)
        logger.info("Successfully completed 03_features.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

