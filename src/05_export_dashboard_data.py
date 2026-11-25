"""
05_export_dashboard_data.py:
Reads processed session and feature datasets and exports lightweight aggregated
Parquet/CSV/JSON files to outputs/ for dashboard visualization:
1. kpis.json: total sessions, cart sessions, abandonment rate, date range
2. abandonment_by_time.csv: abandonment rate by hour of day and by day of week
3. behavior_stats.csv: relationship between views before cart, time to cart, and abandonment
"""

import argparse
import json
import logging
import os
from typing import Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


def setup_logger() -> logging.Logger:
    """Configures and returns a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("05_export_dashboard_data")


def get_spark_session(app_name: str = "05_export_dashboard_data") -> SparkSession:
    """Initializes and returns a SparkSession with memory and partition configurations."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def export_kpis(
    sessionized_events_df: DataFrame,
    cart_sessions_df: DataFrame,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Exports global KPIs to JSON."""
    logger.info("Computing global KPIs...")
    total_sessions = sessionized_events_df.select("session_id").distinct().count()
    total_visitors = sessionized_events_df.select("visitorid").distinct().count()
    total_events = sessionized_events_df.count()

    cart_count = cart_sessions_df.count()
    abandoned_count = cart_sessions_df.filter(F.col("abandoned") == 1).count()
    converted_count = cart_count - abandoned_count
    abandonment_rate = (abandoned_count / cart_count * 100.0) if cart_count > 0 else 0.0

    # Date range
    min_date, max_date = sessionized_events_df.select(
        F.min("timestamp"), F.max("timestamp")
    ).first()

    kpi_dict = {
        "total_sessions": int(total_sessions),
        "total_visitors": int(total_visitors),
        "total_events": int(total_events),
        "cart_sessions": int(cart_count),
        "abandoned_sessions": int(abandoned_count),
        "converted_sessions": int(converted_count),
        "abandonment_rate": round(abandonment_rate, 2),
        "cart_conversion_rate": round(100.0 - abandonment_rate, 2),
        "start_date": str(min_date),
        "end_date": str(max_date),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kpi_dict, f, indent=2)
    logger.info("Exported KPIs to %s: %s", output_path, kpi_dict)


def export_time_aggregates(
    features_df: DataFrame,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Exports abandonment rates aggregated by hour of day and day of week."""
    logger.info("Computing temporal aggregates...")

    # By hour of day (0 to 23)
    hour_df = (
        features_df.groupBy("hour_of_day")
        .agg(
            F.count("session_id").alias("total_sessions"),
            F.sum(F.when(F.col("abandoned") == 1.0, 1).otherwise(0)).alias("abandoned_sessions"),
        )
        .withColumn(
            "abandonment_rate",
            F.round(F.col("abandoned_sessions") / F.col("total_sessions") * 100.0, 2),
        )
        .orderBy("hour_of_day")
        .toPandas()
    )
    hour_csv = os.path.join(output_dir, "abandonment_by_hour.csv")
    hour_df.to_csv(hour_csv, index=False)
    logger.info("Exported hourly abandonment to %s", hour_csv)

    # By day of week (1=Sun, 2=Mon, ..., 7=Sat)
    dow_df = (
        features_df.groupBy("day_of_week")
        .agg(
            F.count("session_id").alias("total_sessions"),
            F.sum(F.when(F.col("abandoned") == 1.0, 1).otherwise(0)).alias("abandoned_sessions"),
        )
        .withColumn(
            "abandonment_rate",
            F.round(F.col("abandoned_sessions") / F.col("total_sessions") * 100.0, 2),
        )
        .orderBy("day_of_week")
        .toPandas()
    )
    # Map day number to human readable day name
    day_map = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
    dow_df["day_name"] = dow_df["day_of_week"].map(day_map)

    dow_csv = os.path.join(output_dir, "abandonment_by_weekday.csv")
    dow_df.to_csv(dow_csv, index=False)
    logger.info("Exported weekday abandonment to %s", dow_csv)


def export_behavior_stats(
    features_df: DataFrame,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Exports binned behavior distributions (views before cart, time to cart)."""
    logger.info("Computing user behavior aggregates...")

    # Views before cart binned (0, 1, 2, 3, 4, 5, 6-10, 11+)
    views_binned = features_df.withColumn(
        "views_group",
        F.when(F.col("views_before_cart") == 0, "0")
        .when(F.col("views_before_cart") == 1, "1")
        .when(F.col("views_before_cart") == 2, "2")
        .when(F.col("views_before_cart") == 3, "3")
        .when(F.col("views_before_cart") == 4, "4")
        .when(F.col("views_before_cart") == 5, "5")
        .when(F.col("views_before_cart") <= 10, "6-10")
        .otherwise("11+"),
    )

    views_agg = (
        views_binned.groupBy("views_group")
        .agg(
            F.count("session_id").alias("total_sessions"),
            F.sum(F.when(F.col("abandoned") == 1.0, 1).otherwise(0)).alias("abandoned_sessions"),
            F.avg("abandoned").alias("abandonment_rate_raw"),
        )
        .withColumn("abandonment_rate", F.round(F.col("abandonment_rate_raw") * 100.0, 2))
        .drop("abandonment_rate_raw")
        .toPandas()
    )

    # Order bins logically
    order = ["0", "1", "2", "3", "4", "5", "6-10", "11+"]
    views_agg["order"] = views_agg["views_group"].map(lambda x: order.index(x) if x in order else 99)
    views_agg = views_agg.sort_values("order").drop(columns=["order"])
    views_csv = os.path.join(output_dir, "behavior_views.csv")
    views_agg.to_csv(views_csv, index=False)

    # Time to cart binned (< 1m, 1-5m, 5-15m, 15-30m, 30m+)
    time_binned = features_df.withColumn(
        "time_group",
        F.when(F.col("seconds_to_cart") < 60, "< 1 min")
        .when(F.col("seconds_to_cart") < 300, "1-5 min")
        .when(F.col("seconds_to_cart") < 900, "5-15 min")
        .when(F.col("seconds_to_cart") < 1800, "15-30 min")
        .otherwise("30+ min"),
    )

    time_agg = (
        time_binned.groupBy("time_group")
        .agg(
            F.count("session_id").alias("total_sessions"),
            F.sum(F.when(F.col("abandoned") == 1.0, 1).otherwise(0)).alias("abandoned_sessions"),
            F.avg("abandoned").alias("abandonment_rate_raw"),
        )
        .withColumn("abandonment_rate", F.round(F.col("abandonment_rate_raw") * 100.0, 2))
        .drop("abandonment_rate_raw")
        .toPandas()
    )

    t_order = ["< 1 min", "1-5 min", "5-15 min", "15-30 min", "30+ min"]
    time_agg["order"] = time_agg["time_group"].map(lambda x: t_order.index(x) if x in t_order else 99)
    time_agg = time_agg.sort_values("order").drop(columns=["order"])
    time_csv = os.path.join(output_dir, "behavior_time.csv")
    time_agg.to_csv(time_csv, index=False)
    logger.info("Exported behavior stats to %s and %s", views_csv, time_csv)


def main() -> None:
    """CLI entrypoint for 05_export_dashboard_data."""
    parser = argparse.ArgumentParser(description="Export dashboard metrics and charts data.")
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
        "--input-features",
        type=str,
        default="data/processed/features.parquet",
        help="Path to feature dataset Parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for exported metrics",
    )
    args = parser.parse_args()

    logger = setup_logger()
    spark = get_spark_session("05_export_dashboard_data")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        logger.info("Loading processed datasets...")
        events_df = spark.read.parquet(args.input_events)
        cart_df = spark.read.parquet(args.input_sessions)
        features_df = spark.read.parquet(args.input_features)

        if args.sample is not None and 0.0 < args.sample < 1.0:
            logger.info("Sampling %0.2f fraction of visitors", args.sample)
            v_sample = cart_df.select("visitorid").distinct().sample(fraction=args.sample, seed=42)
            cart_df = cart_df.join(v_sample, on="visitorid", how="inner")
            events_df = events_df.join(v_sample, on="visitorid", how="inner")
            features_df = features_df.join(v_sample, on="visitorid", how="inner")

        # 1. Export KPIs
        kpi_path = os.path.join(args.output_dir, "kpis.json")
        export_kpis(events_df, cart_df, kpi_path, logger)

        # 2. Export Temporal Aggregates
        export_time_aggregates(features_df, args.output_dir, logger)

        # 3. Export User Behavior stats
        export_behavior_stats(features_df, args.output_dir, logger)

        logger.info("Successfully completed 05_export_dashboard_data.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

