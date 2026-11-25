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
