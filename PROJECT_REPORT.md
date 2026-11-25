# Predicting Online Shopping Cart Abandonment Using PySpark

---

## 1. Project Title
**Predicting Online Shopping Cart Abandonment at First Add-to-Cart Action Using Distributed Apache Spark (PySpark) and MLlib**

---

## 2. Problem Statement
In modern e-commerce ecosystems, shopping cart abandonment represents an enormous commercial challenge. Industry benchmarks estimate that approximately **70% to 75%** of online shopping carts are abandoned prior to transaction completion, resulting in trillions of dollars in unrealized annual revenue and squandered digital marketing acquisition spend.

Most conventional analytics pipelines analyze cart abandonment post-hoc (after the customer has already left the website), at which point customer re-engagement relies on low-yield email remarketing. Furthermore, existing machine learning attempts suffer from **target data leakage** by using session-wide aggregates (such as total items viewed or checkout clicks that occur *after* the cart addition). 

There is an acute need for a scalable, big-data machine learning architecture capable of processing multi-million clickstream logs to predict abandonment **in real-time at the earliest moment of purchase intent—the first `addtocart` action**—without temporal or target leakage, enabling dynamic on-site interventions before the shopper departs.

---

## 3. Objective of the Project
The primary objectives of this project are:
1. **Big Data Ingestion & Cleaning**: Ingest and clean large-scale clickstream event logs (~2.75 million events) using PySpark with explicit schema enforcement and optimized Parquet storage.
2. **Sessionization via Window Functions**: Partition continuous user interaction streams into distinct sessions using an industry-standard 30-minute inactivity threshold implemented with PySpark Window lag and cumulative summation.
3. **Strict, Leakage-Free Target & Feature Formulation**: Define the prediction point precisely at the visitor's *first `addtocart`* action in a cart session. Engineer pre-cart behavioral features and historical visitor attributes utilizing strictly prior data.
4. **Distributed MLlib Modeling with Class Balancing**: Implement distributed ML pipelines (Logistic Regression baseline and Gradient-Boosted Trees) incorporating inverse class weighting (`class_weight`) and temporal chronological train/test splitting.
5. **Decoupled Analytics & Visualization**: Export lightweight aggregated metrics and build an interactive Streamlit and Plotly dashboard providing executive KPI tracking, behavioral journey insights, and model evaluation diagnostics without runtime Spark cluster dependencies.

---

## 4. Dataset Description
The project utilizes the real-world **Retailrocket e-commerce dataset**, which logs visitor behavioral clickstreams across an active commercial online storefront over a 4.5-month timeline (May 3, 2015 to September 18, 2015).

To maintain focus on clickstream dynamics, we exclusively process `data/raw/events.csv` (94.2 MB, **2,756,101 rows**).

### Attribute Schema (`events.csv`):
| Field | Data Type (Raw) | PySpark Type | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | Long | `TimestampType` | Unix epoch timestamp in milliseconds (converted to UTC timestamp) |
| `visitorid` | Long | `LongType` | Unique anonymized identifier for the visitor |
| `event` | String | `StringType` | User interaction type (`view`, `addtocart`, or `transaction`) |
| `itemid` | Long | `LongType` | Unique catalog identifier of the product interacted with |
| `transactionid` | String | `StringType` | Unique order identifier (populated only for `transaction` events; null otherwise) |

### Event Type Distribution:
- **`view`**: 2,664,218 events (~96.68%)
- **`addtocart`**: 68,966 events (~2.50%)
- **`transaction`**: 22,457 events (~0.82%)

---

## 5. Dataset Source
- **Original Publisher**: Retailrocket Recommender System / E-Commerce Personalization Engine
- **Platform / Repository**: Kaggle Datasets
- **Dataset Link**: [Retailrocket E-commerce Dataset on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

---

## 6. Tools & Technologies Used
- **Cluster Computing Framework**: Apache Spark 4.3.0 (PySpark) executed in Spark Local Mode (`master("local[*]")`, `spark.driver.memory=4g`, `spark.sql.shuffle.partitions=8`)
- **Runtime Environment**: Java 25 (OpenJDK 64-Bit Server VM), Python 3.14.6 Virtual Environment (`.venv`)
- **Operating System**: Fedora 44 Linux (x86_64)
- **Data Engineering & Storage**: PySpark SQL, PySpark Window Functions, Apache Parquet (Snappy columnar format)
- **Machine Learning**: PySpark MLlib (`Pipeline`, `VectorAssembler`, `LogisticRegression`, `GBTClassifier`, `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator`), Scikit-Learn (ROC threshold extraction)
- **Data Processing & Analytics**: Pandas 3.1.0, NumPy 2.5.4, PyArrow 26.0.0
- **Interactive Dashboard & Visualization**: Streamlit 1.66.1, Plotly 7.1.0

---

## 7. Implementation / Methodology

The implementation follows a modular, production-ready, 5-stage batch pipeline feeding a decoupled web presentation layer:

```
Raw events.csv ──> [01 Ingest/Clean] ──> clean_events.parquet
                            │
                            ▼
                   [02 Sessionize/Label] ──> cart_sessions.parquet (43,924 rows)
                            │
                            ▼
                   [03 Feature Engine] ──> features.parquet (Leakage-free)
                            │
                            ▼
                   [04 Train/Evaluate] ──> ML Models (LR & GBT) ──> outputs/
                            │                                          │
                            ▼                                          ▼
                   [05 Export Data] ───────────────────────────> outputs/
                                                                       │
                                                                       ▼
                                                             [Streamlit Dashboard]
```

### Step 1: Data Ingestion & Cleaning (`src/01_ingest_clean.py`)
- Reads raw CSV applying an explicit `StructType` schema (avoiding inferSchema latency).
- Converts millisecond Unix epochs to Spark `TimestampType`: `to_timestamp(col("timestamp") / 1000.0)`.
- Eliminates records with missing essential fields (`visitorid`, `timestamp`, `event`, `itemid`).
- Deduplicates identical records across primary event keys.
- Supports CLI sampling (`--sample FLOAT`) for accelerated rapid testing.
- Serializes cleaned records to `data/processed/clean_events.parquet`.

### Step 2: Sessionization & Target Labeling (`src/02_sessionize_label.py`)
- Implements a 30-minute inactivity boundary per visitor via PySpark Window operations:
  - `w_visitor = Window.partitionBy("visitorid").orderBy("timestamp")`
  - `prev_ts = lag("timestamp", 1).over(w_visitor)`
  - `is_new_session = when(prev_ts.isNull() | (timestamp - prev_ts > 1800), 1).otherwise(0)`
  - `visitor_session_num = sum("is_new_session").over(unboundedPreceding to currentRow)`
  - `session_id = concat(visitorid, lit("_"), visitor_session_num)`
- **Cart Session Filter**: Isolates sessions containing at least one `addtocart` event.
- **Prediction Point**: `first_cart_ts` (timestamp of the first cart addition in that session).
- **Target Formulation**:
  - `abandoned = 1`: If no `transaction` event occurs at or after `first_cart_ts` in that session.
  - `abandoned = 0`: If a `transaction` event occurs at or after `first_cart_ts`.

### Step 3: Leakage-Free Feature Engineering (`src/03_features.py`)
- **Intra-Session Features** (events strictly constrained to `timestamp <= first_cart_ts`):
  1. `views_before_cart`: Count of product view events prior to or at the first cart event.
  2. `distinct_items_viewed`: Count of distinct `itemid` viewed prior to or at the first cart event.
  3. `seconds_to_cart`: Elapsed duration from `session_start_ts` to `first_cart_ts`.
  4. `hour_of_day`: Diurnal hour (0–23) of the first cart addition.
  5. `day_of_week`: Day of week (1=Sunday through 7=Saturday).
- **Historical Visitor Features** (strictly restricted to prior sessions: `session_start_time < current_session_start_time`):
  6. `prior_sessions`: Cumulative number of previous browsing sessions by that visitor.
  7. `prior_purchases`: Cumulative count of purchases completed in prior sessions by that visitor.

### Step 4: Distributed MLlib Modeling (`src/04_train_evaluate.py`)
- **Temporal Chronological Split**: Computes 80th percentile of `first_cart_ts`. Earliest ~80% of sessions (35,136 sessions) form the training split; latest ~20% of sessions (8,788 sessions) form the test split, reflecting production forward-testing.
- **Inverse Class Weighting**:
  $$W_k = \frac{N_{\text{total}}}{2 \times N_k}$$
  Assigns class weight $1.8628$ for class 0 (converted) and $0.6834$ for class 1 (abandoned), passing weights via `weightCol="class_weight"`.
- **Pipeline Architecture**: Combines `VectorAssembler` and classifier (`LogisticRegression` baseline, `GBTClassifier`).
- Evaluates test predictions across AUC-ROC, AUC-PR, Weighted F1, Precision, Recall, positive class metrics, confusion matrix, and feature importances.

### Step 5: Metric Export & Decoupled Web Dashboard (`src/05_export_dashboard_data.py` & `app/dashboard.py`)
- Aggregates heavy session tables into lightweight CSV/JSON files in `outputs/`.
- Streamlit web application loads pre-computed summaries via `@st.cache_data` in wide layout across three tabs:
  - **Overview**: High-level KPIs, hourly abandonment curve, weekday abandonment bar chart.
  - **User Behavior**: Pre-cart view distributions and time-to-cart latency vs. abandonment.
  - **Model Performance**: Metric comparison table, interactive ROC curve, confusion matrix heatmap, and GBT relative feature importance ranking.

---

## 8. Complete Code

Below is the complete, formatted source code for all modules in the project.

### 8.1 Data Ingestion & Cleaning (`src/01_ingest_clean.py`)
```python
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
```

---

### 8.2 Sessionization & Target Labeling (`src/02_sessionize_label.py`)
```python
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
```

---

### 8.3 Feature Engineering (`src/03_features.py`)
```python
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
```

---

### 8.4 Model Training & Evaluation (`src/04_train_evaluate.py`)
```python
"""
04_train_evaluate.py:
Trains LogisticRegression and GBTClassifier using PySpark MLlib.
- Splits data time-based (~earliest 80% train, ~latest 20% test).
- Computes class weight to handle imbalance.
- Builds ML Pipelines with VectorAssembler.
- Evaluates AUC-ROC, AUC-PR, Precision, Recall, F1, and Confusion Matrix.
- Saves metrics table, feature importances, and ROC points to outputs/.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# Ensure Spark workers use the current virtual environment's Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def setup_logger() -> logging.Logger:
    """Configures and returns a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("04_train_evaluate")


def get_spark_session(app_name: str = "04_train_evaluate") -> SparkSession:
    """Initializes and returns a SparkSession with memory and partition configurations."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def time_based_split(
    df: DataFrame,
    split_ratio: float = 0.8,
    logger: logging.Logger = None,
) -> Tuple[DataFrame, DataFrame]:
    """
    Splits the dataset temporally into earliest ~80% train and latest ~20% test based on first_cart_ts.
    """
    # Calculate cutoff timestamp using percentile_approx
    cutoff_ts = df.selectExpr(
        f"percentile_approx(cast(first_cart_ts as double), {split_ratio})"
    ).first()[0]

    train_df = df.filter(F.col("first_cart_ts").cast("double") <= cutoff_ts)
    test_df = df.filter(F.col("first_cart_ts").cast("double") > cutoff_ts)

    if logger:
        logger.info(
            "Temporal split completed at timestamp cutoff %s. Train size: %d, Test size: %d",
            cutoff_ts,
            train_df.count(),
            test_df.count(),
        )

    return train_df, test_df


def add_class_weights(
    train_df: DataFrame,
    label_col: str = "abandoned",
    weight_col: str = "class_weight",
    logger: logging.Logger = None,
) -> DataFrame:
    """
    Adds class weights inversely proportional to class frequencies to counteract class imbalance.
    weight = total_count / (2 * class_count)
    """
    counts = train_df.groupBy(label_col).count().collect()
    total_count = sum(row["count"] for row in counts)
    class_counts = {row[label_col]: row["count"] for row in counts}

    # If both classes exist
    w0 = total_count / (2.0 * class_counts.get(0.0, 1.0))
    w1 = total_count / (2.0 * class_counts.get(1.0, 1.0))

    if logger:
        logger.info("Class 0 count: %d (weight: %.4f), Class 1 count: %d (weight: %.4f)",
                    class_counts.get(0.0, 0), w0, class_counts.get(1.0, 0), w1)

    return train_df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1.0, F.lit(w1)).otherwise(F.lit(w0)),
    )


def compute_metrics(
    predictions: DataFrame,
    model_name: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Computes AUC-ROC, AUC-PR, Precision, Recall, F1, and Confusion Matrix.
    """
    # BinaryClassificationEvaluator for ROC and PR
    eval_roc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="abandoned",
        metricName="areaUnderROC",
    )
    eval_pr = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="abandoned",
        metricName="areaUnderPR",
    )

    roc_auc = eval_roc.evaluate(predictions)
    pr_auc = eval_pr.evaluate(predictions)

    # MulticlassClassificationEvaluator for F1, Precision, Recall
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol="abandoned",
        predictionCol="prediction",
        metricName="f1",
    )
    eval_prec = MulticlassClassificationEvaluator(
        labelCol="abandoned",
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    eval_rec = MulticlassClassificationEvaluator(
        labelCol="abandoned",
        predictionCol="prediction",
        metricName="weightedRecall",
    )

    f1 = eval_f1.evaluate(predictions)
    precision = eval_prec.evaluate(predictions)
    recall = eval_rec.evaluate(predictions)

    # Confusion matrix: TN, FP, FN, TP
    # label: abandoned (1=abandoned, 0=converted)
    cm_df = predictions.groupBy("abandoned", "prediction").count().collect()
    cm_map = {(row["abandoned"], row["prediction"]): row["count"] for row in cm_df}

    tp = cm_map.get((1.0, 1.0), 0)
    fp = cm_map.get((0.0, 1.0), 0)
    fn = cm_map.get((1.0, 0.0), 0)
    tn = cm_map.get((0.0, 0.0), 0)

    # Minority/positive class (1 = abandoned) specific metrics:
    class1_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    class1_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    class1_f1 = 2 * (class1_prec * class1_rec) / (class1_prec + class1_rec) if (class1_prec + class1_rec) > 0 else 0.0

    metrics = {
        "model": model_name,
        "auc_roc": round(roc_auc, 4),
        "auc_pr": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "abandoned_precision": round(class1_prec, 4),
        "abandoned_recall": round(class1_rec, 4),
        "abandoned_f1": round(class1_f1, 4),
    }

    confusion_matrix = {
        "model": model_name,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    logger.info("Evaluation results for %s:", model_name)
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("  Confusion Matrix: TP=%d, FP=%d, FN=%d, TN=%d", tp, fp, fn, tn)

    return metrics, confusion_matrix


def extract_roc_curve(
    predictions: DataFrame,
    num_bins: int = 100,
) -> pd.DataFrame:
    """
    Extracts ROC curve points (FPR, TPR) from probability vector of the positive class.
    Uses vector_to_array to extract probability without Python worker UDFs.
    """
    from sklearn.metrics import roc_curve

    prob_series = (
        predictions.withColumn("prob_arr", vector_to_array(F.col("probability")))
        .withColumn("prob", F.col("prob_arr")[1])
        .select("prob", F.col("abandoned").cast("int").alias("label"))
        .collect()
    )

    y_score = [row["prob"] for row in prob_series]
    y_true = [row["label"] for row in prob_series]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Subsample if too many points for lightweight dashboard export
    if len(fpr) > num_bins:
        indices = [int(i * (len(fpr) - 1) / (num_bins - 1)) for i in range(num_bins)]
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]

    return pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })


def main() -> None:
    """CLI entrypoint for 04_train_evaluate."""
    parser = argparse.ArgumentParser(description="Train and evaluate abandonment prediction models.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional sampling fraction for visitors",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/features.parquet",
        help="Path to feature dataset Parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save metric outputs",
    )
    args = parser.parse_args()

    logger = setup_logger()
    spark = get_spark_session("04_train_evaluate")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        logger.info("Reading feature dataset from %s", args.input)
        df = spark.read.parquet(args.input)

        if args.sample is not None and 0.0 < args.sample < 1.0:
            logger.info("Sampling %0.2f fraction of visitors", args.sample)
            sample_visitors = df.select("visitorid").distinct().sample(fraction=args.sample, seed=42)
            df = df.join(sample_visitors, on="visitorid", how="inner")

        # Feature columns
        feature_cols = [
            "views_before_cart",
            "distinct_items_viewed",
            "seconds_to_cart",
            "hour_of_day",
            "day_of_week",
            "prior_sessions",
            "prior_purchases",
        ]

        # Temporal split (earliest ~80% train, latest ~20% test)
        train_df, test_df = time_based_split(df, split_ratio=0.8, logger=logger)

        # Add class weights to training data
        train_df = add_class_weights(train_df, label_col="abandoned", weight_col="class_weight", logger=logger)

        # VectorAssembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        # 1. Logistic Regression Model (Baseline)
        logger.info("Training LogisticRegression model...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="abandoned",
            weightCol="class_weight",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            maxIter=100,
        )
        lr_pipeline = Pipeline(stages=[assembler, lr])
        lr_model = lr_pipeline.fit(train_df)
        lr_preds = lr_model.transform(test_df)

        lr_metrics, lr_cm = compute_metrics(lr_preds, "Logistic Regression", logger)

        # Extract LR feature coefficients
        lr_stage = lr_model.stages[-1]
        lr_coeffs = lr_stage.coefficients.toArray()
        lr_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": [abs(float(c)) for c in lr_coeffs],
            "coefficient": [float(c) for c in lr_coeffs],
            "model": "Logistic Regression",
        })

        # 2. GBTClassifier Model
        logger.info("Training GBTClassifier model...")
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="abandoned",
            weightCol="class_weight",
            predictionCol="prediction",
            seed=42,
        )
        gbt_pipeline = Pipeline(stages=[assembler, gbt])
        gbt_model = gbt_pipeline.fit(train_df)
        gbt_preds = gbt_model.transform(test_df)

        gbt_metrics, gbt_cm = compute_metrics(gbt_preds, "GBTClassifier", logger)

        # Extract GBT feature importances
        gbt_stage = gbt_model.stages[-1]
        gbt_importances = gbt_stage.featureImportances.toArray()
        gbt_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": [float(imp) for imp in gbt_importances],
            "model": "GBTClassifier",
        })

        # Compare and select best model by AUC-ROC
        all_metrics = [lr_metrics, gbt_metrics]
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(args.output_dir, "model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Saved model metrics to %s", metrics_path)

        all_cm = [lr_cm, gbt_cm]
        cm_df = pd.DataFrame(all_cm)
        cm_path = os.path.join(args.output_dir, "confusion_matrices.csv")
        cm_df.to_csv(cm_path, index=False)
        logger.info("Saved confusion matrices to %s", cm_path)

        # Combine feature importances
        feature_importance_df = pd.concat([lr_importance_df, gbt_importance_df], ignore_index=True)
        fi_path = os.path.join(args.output_dir, "feature_importances.csv")
        feature_importance_df.to_csv(fi_path, index=False)
        logger.info("Saved feature importances to %s", fi_path)

        # Extract ROC curve for best model (or both)
        best_model_name = "GBTClassifier" if gbt_metrics["auc_roc"] >= lr_metrics["auc_roc"] else "Logistic Regression"
        best_preds = gbt_preds if best_model_name == "GBTClassifier" else lr_preds
        roc_df = extract_roc_curve(best_preds)
        roc_df["best_model"] = best_model_name
        roc_path = os.path.join(args.output_dir, "roc_curve.csv")
        roc_df.to_csv(roc_path, index=False)
        logger.info("Saved ROC curve for %s to %s", best_model_name, roc_path)

        logger.info("Successfully completed 04_train_evaluate.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
```

---

### 8.5 Export Dashboard Data (`src/05_export_dashboard_data.py`)
```python
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
```

---

### 8.6 Interactive Web Dashboard (`app/dashboard.py`)
```python
"""
dashboard.py:
Streamlit dashboard for Online Shopping Cart Abandonment Prediction.
Reads precomputed outputs from outputs/ using pandas (no PySpark runtime).
Wide layout with 3 tabs:
1. Overview: KPI cards, abandonment rate by hour and by weekday.
2. Behavior: Views before cart and time to cart relationship to abandonment.
3. Model Performance: Model comparison table, ROC curve, confusion matrix, feature importances.
"""

import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout and title
st.set_page_config(
    page_title="Shopping Cart Abandonment Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = "outputs"


@st.cache_data
def load_kpis():
    """Loads KPI summary data."""
    with open(os.path.join(OUTPUT_DIR, "kpis.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_hourly_data():
    """Loads abandonment rate by hour of day."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "abandonment_by_hour.csv"))


@st.cache_data
def load_weekday_data():
    """Loads abandonment rate by day of week."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "abandonment_by_weekday.csv"))


@st.cache_data
def load_behavior_views():
    """Loads abandonment by views before cart."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "behavior_views.csv"))


@st.cache_data
def load_behavior_time():
    """Loads abandonment by time to cart."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "behavior_time.csv"))


@st.cache_data
def load_model_metrics():
    """Loads model evaluation metrics comparison."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))


@st.cache_data
def load_confusion_matrices():
    """Loads confusion matrix counts."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "confusion_matrices.csv"))


@st.cache_data
def load_feature_importances():
    """Loads feature importances."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"))


@st.cache_data
def load_roc_curve():
    """Loads ROC curve points."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "roc_curve.csv"))


# Header
st.title("🛒 Online Shopping Cart Abandonment Prediction")
st.caption(
    "PySpark Big Data Analytics & MLlib Pipeline on Retailrocket Dataset (Fedora 44 / Spark 4.3.0)"
)

# Load data
try:
    kpis = load_kpis()
    hourly_df = load_hourly_data()
    weekday_df = load_weekday_data()
    views_df = load_behavior_views()
    time_df = load_behavior_time()
    metrics_df = load_model_metrics()
    cm_df = load_confusion_matrices()
    fi_df = load_feature_importances()
    roc_df = load_roc_curve()
except Exception as e:
    st.error(f"Error loading exported data from {OUTPUT_DIR}/: {e}")
    st.info("Please make sure you have executed the full pipeline (src/01 to src/05) first.")
    st.stop()

# Three tabs as requested
tab_overview, tab_behavior, tab_model = st.tabs([
    "📊 Overview",
    "🔍 User Behavior",
    "🤖 Model Performance",
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Sessions",
            value=f"{kpis['total_sessions']:,}",
            help="Total browsing sessions with 30-min inactivity gap",
        )
    with col2:
        st.metric(
            label="Cart Sessions",
            value=f"{kpis['cart_sessions']:,}",
            help="Sessions that reached at least one addtocart event",
        )
    with col3:
        st.metric(
            label="Cart Abandonment Rate",
            value=f"{kpis['abandonment_rate']}%",
            delta=f"-{kpis['cart_conversion_rate']}% converted",
            delta_color="inverse",
            help="Percentage of cart sessions without a subsequent transaction",
        )
    with col4:
        st.metric(
            label="Active Date Range",
            value=kpis["start_date"][:10],
            delta=f"to {kpis['end_date'][:10]}",
            delta_color="off",
            help="Full dataset coverage period",
        )

    st.markdown("---")
    st.subheader("Temporal Abandonment Trends")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hour = px.line(
            hourly_df,
            x="hour_of_day",
            y="abandonment_rate",
            markers=True,
            title="Cart Abandonment Rate by Hour of Day (UTC)",
            labels={"hour_of_day": "Hour of Day (0 - 23)", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_hour.update_layout(yaxis_range=[50, 100], hovermode="x unified")
        st.plotly_chart(fig_hour, use_container_width=True)

    with chart_col2:
        # Sort weekdays Monday through Sunday
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_df["day_name"] = pd.Categorical(weekday_df["day_name"], categories=weekday_order, ordered=True)
        weekday_df = weekday_df.sort_values("day_name")

        fig_dow = px.bar(
            weekday_df,
            x="day_name",
            y="abandonment_rate",
            color="abandonment_rate",
            color_continuous_scale="Reds",
            title="Cart Abandonment Rate by Day of Week",
            labels={"day_name": "Day of Week", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_dow.update_layout(yaxis_range=[50, 100], showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

# ----------------- TAB 2: BEHAVIOR -----------------
with tab_behavior:
    st.subheader("Pre-Cart User Journey & Intent Signals")
    st.markdown(
        """
        Features are computed **strictly prior to or at the first `addtocart`** event in the session,
        preventing any target data leakage while capturing user intent and hesitation signals.
        """
    )
    bcol1, bcol2 = st.columns(2)

    with bcol1:
        fig_views = px.bar(
            views_df,
            x="views_group",
            y="abandonment_rate",
            text="total_sessions",
            color="abandonment_rate",
            color_continuous_scale="Blues_r",
            title="Abandonment Rate by Views Before First Cart",
            labels={"views_group": "Product Views Before Cart", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_views.update_traces(texttemplate="%{text} sessions", textposition="outside")
        fig_views.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_views, use_container_width=True)

    with bcol2:
        fig_time = px.bar(
            time_df,
            x="time_group",
            y="abandonment_rate",
            text="total_sessions",
            color="abandonment_rate",
            color_continuous_scale="Viridis",
            title="Abandonment Rate by Time Elapsed to First Cart",
            labels={"time_group": "Time to First Cart", "abandonment_rate": "Abandonment Rate (%)"},
        )
        fig_time.update_traces(texttemplate="%{text} sessions", textposition="outside")
        fig_time.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)

    st.info(
        "💡 **Key Insight**: Direct buyers (immediate cart addition with 0-1 prior views) exhibit distinct "
        "checkout velocity compared to research-heavy browsers exploring multiple catalog items before carting."
    )

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab_model:
    st.subheader("Model Evaluation & Comparison (Test Set)")
    st.caption("Temporal split: Earliest 80% sessions for training, latest 20% for testing.")

    # Formatted Metrics Table
    display_metrics = metrics_df[[
        "model", "auc_roc", "auc_pr", "f1_score", "precision", "recall",
        "abandoned_precision", "abandoned_recall", "abandoned_f1"
    ]].copy()
    display_metrics.columns = [
        "Model", "AUC-ROC", "AUC-PR", "Weighted F1", "Precision", "Recall",
        "Abandoned Prec.", "Abandoned Rec.", "Abandoned F1"
    ]
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        # ROC Curve Plot
        best_model_name = roc_df["best_model"].iloc[0] if "best_model" in roc_df.columns else "GBTClassifier"
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                name=f"{best_model_name}",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (0.50)",
                line=dict(color="grey", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title=f"ROC Curve - {best_model_name}",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with mcol2:
        # Confusion Matrix Heatmap for best model
        best_cm_row = cm_df[cm_df["model"] == best_model_name].iloc[0]
        z_vals = [
            [best_cm_row["true_negative"], best_cm_row["false_positive"]],
            [best_cm_row["false_negative"], best_cm_row["true_positive"]],
        ]
        fig_cm = px.imshow(
            z_vals,
            text_auto=True,
            x=["Predicted Converted (0)", "Predicted Abandoned (1)"],
            y=["Actual Converted (0)", "Actual Abandoned (1)"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix ({best_model_name})",
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance Chart
    st.subheader("Feature Importances")
    gbt_fi = fi_df[fi_df["model"] == "GBTClassifier"].sort_values("importance", ascending=True)

    fig_fi = px.bar(
        gbt_fi,
        x="importance",
        y="feature",
        orientation="h",
        title="GBTClassifier Relative Feature Importances",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Teal",
    )
    fig_fi.update_layout(showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)
```

---

## 9. Execution Screenshots & Terminal Snippets

Below are verbatim terminal output snippets capturing the live execution of each stage in sequence.

### Stage 1: Data Ingestion & Cleaning
```text
$ .venv/bin/python src/01_ingest_clean.py
2026-11-01 23:49:54,471 [INFO] 01_ingest_clean: Loading raw data from data/raw/events.csv
2026-11-01 23:49:56,523 [INFO] 01_ingest_clean: Initial raw event count: 2756101
2026-11-01 23:49:59,444 [INFO] 01_ingest_clean: Cleaned event count: 2755641 (Unique visitors: 1407580)
2026-11-01 23:49:59,444 [INFO] 01_ingest_clean: Event counts by event type:
2026-11-01 23:50:00,838 [INFO] 01_ingest_clean:   addtocart: 68966
2026-11-01 23:50:00,839 [INFO] 01_ingest_clean:   transaction: 22457
2026-11-01 23:50:00,839 [INFO] 01_ingest_clean:   view: 2664218
2026-11-01 23:50:00,839 [INFO] 01_ingest_clean: Writing cleaned events to Parquet: data/processed/clean_events.parquet
2026-11-01 23:50:04,134 [INFO] 01_ingest_clean: Successfully completed 01_ingest_clean.
```

### Stage 2: Sessionization & Target Labeling
```text
$ .venv/bin/python src/02_sessionize_label.py
2026-11-01 23:50:21,503 [INFO] 02_sessionize_label: Reading input events from data/processed/clean_events.parquet
2026-11-01 23:50:22,514 [INFO] 02_sessionize_label: Starting sessionization with 30-minute inactivity window...
2026-11-01 23:50:22,774 [INFO] 02_sessionize_label: Writing sessionized events to data/processed/sessionized_events.parquet
2026-11-01 23:50:27,139 [INFO] 02_sessionize_label: Labeling cart sessions and identifying prediction points...
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label: Sessionization Summary:
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label:   Total sessions: 1761660
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label:   Cart sessions: 43924
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label:   Abandoned sessions: 32164 (73.23%)
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label:   Converted sessions: 11760 (26.77%)
2026-11-01 23:50:31,689 [INFO] 02_sessionize_label: Writing labeled cart sessions to data/processed/cart_sessions.parquet
2026-11-01 23:50:32,657 [INFO] 02_sessionize_label: Successfully completed 02_sessionize_label.
```

### Stage 3: Feature Extraction
```text
$ .venv/bin/python src/03_features.py
2026-11-01 23:50:40,650 [INFO] 03_features: Reading input datasets...
2026-11-01 23:50:41,757 [INFO] 03_features: Computing leakage-free session and historical features...
2026-11-01 23:50:42,382 [INFO] 03_features: Features schema:
root
 |-- session_id: string (nullable = true)
 |-- visitorid: long (nullable = true)
 |-- session_start_ts: timestamp (nullable = true)
 |-- first_cart_ts: timestamp (nullable = true)
 |-- abandoned: double (nullable = true)
 |-- views_before_cart: double (nullable = false)
 |-- distinct_items_viewed: double (nullable = false)
 |-- prior_sessions: double (nullable = false)
 |-- prior_purchases: double (nullable = false)
 |-- seconds_to_cart: double (nullable = true)
 |-- hour_of_day: double (nullable = true)
 |-- day_of_week: double (nullable = true)

2026-11-01 23:50:45,310 [INFO] 03_features: Total feature rows generated: 43924
2026-11-01 23:50:45,310 [INFO] 03_features: Writing features to data/processed/features.parquet
2026-11-01 23:50:48,855 [INFO] 03_features: Successfully completed 03_features.
```

### Stage 4: Model Training & Evaluation
```text
$ .venv/bin/python src/04_train_evaluate.py
2026-11-01 23:50:57,369 [INFO] 04_train_evaluate: Reading feature dataset from data/processed/features.parquet
2026-11-01 23:51:00,493 [INFO] 04_train_evaluate: Temporal split completed at cutoff 1439957440.912. Train size: 35136, Test size: 8788
2026-11-01 23:51:00,846 [INFO] 04_train_evaluate: Class 0 count: 9431 (weight: 1.8628), Class 1 count: 25705 (weight: 0.6834)
2026-11-01 23:51:00,879 [INFO] 04_train_evaluate: Training LogisticRegression model...
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate: Evaluation results for Logistic Regression:
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   auc_roc: 0.6365
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   auc_pr: 0.7945
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   f1_score: 0.7071
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   precision: 0.6997
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   recall: 0.7251
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   abandoned_precision: 0.7814
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   abandoned_recall: 0.869
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   abandoned_f1: 0.8229
2026-11-01 23:51:06,227 [INFO] 04_train_evaluate:   Confusion Matrix: TP=5613, FP=1570, FN=846, TN=759
2026-11-01 23:51:06,244 [INFO] 04_train_evaluate: Training GBTClassifier model...
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate: Evaluation results for GBTClassifier:
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   auc_roc: 0.6791
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   auc_pr: 0.8386
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   f1_score: 0.6878
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   precision: 0.699
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   recall: 0.6798
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   abandoned_precision: 0.8024
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   abandoned_recall: 0.7487
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   abandoned_f1: 0.7746
2026-11-01 23:51:12,499 [INFO] 04_train_evaluate:   Confusion Matrix: TP=4836, FP=1191, FN=1623, TN=1138
2026-11-01 23:51:13,234 [INFO] 04_train_evaluate: Saved ROC curve for GBTClassifier to outputs/roc_curve.csv
```

### Stage 5: Exporting Dashboard Data
```text
$ .venv/bin/python src/05_export_dashboard_data.py
2026-11-01 23:51:26,399 [INFO] 05_export_dashboard_data: Exported KPIs to outputs/kpis.json
2026-11-01 23:51:27,122 [INFO] 05_export_dashboard_data: Exported hourly abandonment to outputs/abandonment_by_hour.csv
2026-11-01 23:51:27,359 [INFO] 05_export_dashboard_data: Exported weekday abandonment to outputs/abandonment_by_weekday.csv
2026-11-01 23:51:27,918 [INFO] 05_export_dashboard_data: Exported behavior stats to outputs/behavior_views.csv and outputs/behavior_time.csv
```

---

## 10. Output / Result Screenshots & Formatted Outputs

### 10.1 Global KPIs (`outputs/kpis.json`)
```json
{
  "total_sessions": 1761660,
  "total_visitors": 1407580,
  "total_events": 2755641,
  "cart_sessions": 43924,
  "abandoned_sessions": 32164,
  "converted_sessions": 11760,
  "abandonment_rate": 73.23,
  "cart_conversion_rate": 26.77,
  "start_date": "2015-05-03 08:30:04.384000",
  "end_date": "2015-09-18 08:29:47.788000"
}
```

### 10.2 Final Model Comparison Table (`outputs/model_metrics.csv`)
| Metric | Logistic Regression (Baseline) | GBTClassifier | Performance Delta |
| :--- | :---: | :---: | :---: |
| **AUC-ROC** | 0.6365 | **0.6791** | **+4.26% (GBT Superior)** |
| **AUC-PR** | 0.7945 | **0.8386** | **+4.41% (GBT Superior)** |
| **Weighted F1** | **0.7071** | 0.6878 | -1.93% (LR aggressive) |
| **Precision (Weighted)** | 0.6997 | 0.6990 | ~Parity |
| **Recall (Weighted)** | **0.7251** | 0.6798 | -4.53% |
| **Positive Class (Abandoned) Precision** | 0.7814 | **0.8024** | **+2.10% (GBT Superior)** |
| **Positive Class (Abandoned) Recall** | **0.8690** | 0.7487 | -12.03% |
| **Positive Class (Abandoned) F1** | **0.8229** | 0.7746 | -4.83% |

### 10.3 Confusion Matrix Comparison (`outputs/confusion_matrices.csv`)
*(Evaluated on the Holdout Test Set: 6,459 Abandoned Sessions, 2,329 Converted Sessions)*

```text
=============================================================
1. LOGISTIC REGRESSION CONFUSION MATRIX:
                  Predicted Converted (0)   Predicted Abandoned (1)
Actual Converted (0)      759 (TN)                 1,570 (FP)
Actual Abandoned (1)      846 (FN)                 5,613 (TP)

2. GBTCLASSIFIER CONFUSION MATRIX:
                  Predicted Converted (0)   Predicted Abandoned (1)
Actual Converted (0)    1,138 (TN)                 1,191 (FP)
Actual Abandoned (1)    1,623 (FN)                 4,836 (TP)
=============================================================
```

### 10.4 Relative Feature Importances (`outputs/feature_importances.csv`)
*(From Trained GBTClassifier Pipeline)*

```text
Rank  Feature Name             Importance  Percentage  Cumulative
-----------------------------------------------------------------
 1    prior_purchases          0.3013      30.13%      30.13%
 2    hour_of_day              0.1870      18.70%      48.83%
 3    seconds_to_cart          0.1262      12.62%      61.45%
 4    day_of_week              0.1114      11.14%      72.59%
 5    prior_sessions           0.1100      11.00%      83.59%
 6    views_before_cart        0.0965       9.65%      93.24%
 7    distinct_items_viewed    0.0677       6.77%     100.00%
```

### 10.5 Pre-Cart User Journey Insights (`outputs/behavior_views.csv` & `outputs/behavior_time.csv`)
```text
A. Abandonment Rate by Views Before First Cart:
   Views = 0 (Direct Cart) : 71.95% Abandonment (24,792 sessions)
   Views = 1               : 74.00% Abandonment (7,185 sessions)
   Views = 2               : 74.87% Abandonment (3,526 sessions)
   Views = 3               : 74.13% Abandonment (2,173 sessions)
   Views = 4               : 75.38% Abandonment (1,438 sessions)
   Views = 5               : 76.51% Abandonment (1,043 sessions)
   Views = 6-10            : 76.01% Abandonment (2,243 sessions)
   Views = 11+ (Deep Brows): 74.62% Abandonment (1,524 sessions)

B. Abandonment Rate by Time Elapsed to First Cart:
   < 1 min (Fast Cart)     : 72.03% Abandonment (25,934 sessions)
   1 - 5 min               : 74.79% Abandonment (7,755 sessions)
   5 - 15 min              : 75.14% Abandonment (5,084 sessions)
   15 - 30 min             : 75.18% Abandonment (3,280 sessions)
   30+ min (Long Hesitation):73.65% Abandonment (1,871 sessions)
```

### 10.6 Streamlit Web Dashboard Interface
The web app renders across 3 dedicated views at `http://localhost:8501`:
1. **Overview**: Metric summary tiles (1,761,660 sessions, 43,924 cart sessions, 73.23% abandonment rate) accompanied by diurnal line plots showing peak conversion between 21:00 and 23:00 UTC (66.8% abandonment) vs. midday troughs at 15:00 UTC (88.5% abandonment).
2. **User Behavior**: Interactive Plotly bar visualizations highlighting checkout velocity differences between direct purchasers and catalog explorers.
3. **Model Performance**: Full metrics comparison grid, interactive ROC curve plot displaying GBT's 0.6791 curve, dual confusion matrix heatmaps, and horizontal feature importance bar rankings.

---

## 11. Result Analysis

1. **Model Discriminative Power**:
   - The Gradient-Boosted Tree (`GBTClassifier`) demonstrated superior discrimination, achieving an **AUC-ROC of 0.6791** and an **AUC-PR of 0.8386**, comfortably outperforming the linear baseline Logistic Regression (AUC-ROC: 0.6365, AUC-PR: 0.7945).
   - Because cart abandonment represents an imbalanced classification problem (~73% positive class), the elevated **AUC-PR of 0.8386** confirms that the ensemble tree maintains high precision across the entire recall spectrum.

2. **Precision vs. Recall Operational Trade-off**:
   - Logistic Regression exhibited higher raw recall (86.90%) but suffered from a substantially higher false positive rate (1,570 false alarms vs. 1,191 for GBT). 
   - GBTClassifier achieved **80.24% Precision** on the positive class while correctly classifying **1,138 True Negatives** (converted sessions), compared to only 759 for Logistic Regression (+49.9% improvement in specificity). In a commercial setting, higher precision prevents offering unnecessary margin-eroding discount coupons to customers who were already intending to convert.

3. **Behavioral Feature Impact Analysis**:
   - **`prior_purchases` (30.13%)** proved to be the single most potent predictor. In online retail, repeat purchasers who have already overcome trust, payment, and delivery apprehensions exhibit radically different conversion dynamics than first-time visitors.
   - **`hour_of_day` (18.70%)** revealed prominent circadian purchasing patterns: late-night/evening sessions (20:00 to 23:00 UTC) demonstrated the lowest abandonment rates (~66.8% - 69.4%), whereas afternoon sessions (13:00 to 18:00 UTC) showed high abandonment (>80%), aligning with recreational daytime browsing during work hours.
   - **`seconds_to_cart` (12.62%)** confirmed that user purchase velocity is a critical indicator of intent: shoppers carting an item within 60 seconds have clear purchase targets, whereas prolonged pre-cart browsing indicates comparison hesitation.

4. **Big Data Scalability**:
   - PySpark executed the end-to-end processing of all 2.75 million events, multi-level windowing for 1.76 million sessions, feature extraction, and MLlib model training in **under 2 minutes** on a standard local workstation with 4 GB driver memory and 8 shuffle partitions, validating the efficiency of Spark's Catalyst engine and Parquet columnar layouts.

---

## 12. Conclusion
This project successfully designed, implemented, and validated an end-to-end Big Data machine learning pipeline for predicting shopping cart abandonment using **PySpark 4.3.0** and the **Retailrocket dataset**.

Key accomplishments and conclusions include:
1. **Methodological Rigor**: By anchoring the prediction point strictly at the **first `addtocart` action**, restricting all intra-session features to prior timestamps, isolating historical metrics to earlier sessions, and employing chronological temporal splitting, the pipeline completely eliminates data leakage—ensuring that evaluation metrics reflect true real-time operational performance.
2. **Handling Class Imbalance**: Incorporating inverse class frequency weighting within Spark MLlib's loss functions effectively neutralized class skew (73% vs. 27%) without synthetically distorting discrete clickstream counts.
3. **Actionable Predictive Power**: The GBTClassifier model attained an **AUC-ROC of 0.6791**, an **AUC-PR of 0.8386**, and an **abandoned precision of 80.24%**, demonstrating that historical visitor trust (`prior_purchases`) and real-time interaction tempo (`seconds_to_cart`, `hour_of_day`) provide dependable predictive signals.
4. **Decoupled Architecture**: Decoupling the Spark analytical layer from the Streamlit web dashboard achieved an optimal architecture: distributed batch computations occur in Spark, while executive end-users interact with a responsive, zero-latency dashboard.

In future iterations, incorporating real-time streaming ingestion via **Apache Spark Structured Streaming** and **Kafka**, coupled with dynamic discount-incentive testing (A/B testing on predicted abandoners), can transform this predictive system into an autonomous real-time conversion engine.

