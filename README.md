# Predicting Online Shopping Cart Abandonment with PySpark

A big data machine learning pipeline and interactive dashboard for predicting online shopping cart abandonment using **PySpark 4.3.0** and the **Retailrocket dataset** on Fedora 44 Linux.

---

## 📌 Project Overview & Label Definition

The objective is to predict whether a customer will abandon their shopping cart at the earliest decision point.

- **Prediction Point**: The moment the customer performs their **first `addtocart`** event within a session.
- **Cart Session**: Any browsing session that contains at least one `addtocart` event.
- **Target Label (`abandoned`)**:
  - `abandoned = 1`: If the session has **no** `transaction` event at or after its first `addtocart`.
  - `abandoned = 0` (Converted): If the session has a `transaction` event at or after its first `addtocart`.

### 🛡️ Preventing Data Leakage
To strictly prevent target and temporal leakage:
1. **Intra-session features** (`views_before_cart`, `distinct_items_viewed`, `seconds_to_cart`) are computed **strictly using events timestamped at or before `first_cart_ts`**.
2. **Historical visitor features** (`prior_sessions`, `prior_purchases`) aggregate earlier sessions only (`session_start_ts < current session_start_ts`).
3. **Train / Test Split**: Chronological / time-based split (earliest ~80% sessions for training, latest ~20% sessions for testing).

---

## 🏗️ Architecture & Project Structure

```text
├── data/
│   ├── raw/events.csv              # Retailrocket event log
│   └── processed/                  # Processed Parquet files
├── outputs/                        # Lightweight exported metrics for dashboard
│   ├── kpis.json
│   ├── abandonment_by_hour.csv
│   ├── abandonment_by_weekday.csv
│   ├── behavior_views.csv
│   ├── behavior_time.csv
│   ├── model_metrics.csv
│   ├── confusion_matrices.csv
│   ├── feature_importances.csv
│   └── roc_curve.csv
├── src/
│   ├── 01_ingest_clean.py          # Data ingestion, schema, cleaning & deduplication
│   ├── 02_sessionize_label.py      # 30-min window sessionization & target labeling
│   ├── 03_features.py              # Leakage-free feature extraction
│   ├── 04_train_evaluate.py        # MLlib pipeline (LR & GBT with class weighting)
│   └── 05_export_dashboard_data.py # Export aggregated metrics and visualizations
├── app/
│   └── dashboard.py                # Streamlit & Plotly wide layout dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation & run instructions
```

---

## 🚀 Execution Commands (In Order)

Ensure your Python virtual environment is activated or call `.venv/bin/python` directly:

### 1. Fast Development / Sample Run (e.g. 10% sample)
```bash
.venv/bin/python src/01_ingest_clean.py --sample 0.1
.venv/bin/python src/02_sessionize_label.py
.venv/bin/python src/03_features.py
.venv/bin/python src/04_train_evaluate.py
.venv/bin/python src/05_export_dashboard_data.py
```

### 2. Full Dataset Production Run
```bash
.venv/bin/python src/01_ingest_clean.py
.venv/bin/python src/02_sessionize_label.py
.venv/bin/python src/03_features.py
.venv/bin/python src/04_train_evaluate.py
.venv/bin/python src/05_export_dashboard_data.py
```

### 3. Launch Dashboard
```bash
.venv/bin/streamlit run app/dashboard.py
```

The Streamlit dashboard runs without Spark runtime overhead by reading the lightweight metrics in `outputs/`.

---

## 📊 Streamlit Dashboard Tabs

- **📊 Overview**: High-level KPIs (total sessions, cart sessions, abandonment rate, active date range), and abandonment rates grouped by hour of day and weekday.
- **🔍 User Behavior**: Analysis of how pre-cart views and time elapsed to first cart impact cart abandonment.
- **🤖 Model Performance**: Side-by-side metric comparison (AUC-ROC, AUC-PR, F1, Precision, Recall), ROC curves, confusion matrix heatmaps, and tree feature importances.

---

## ⚠️ Limitations & Assumptions

1. **Class Imbalance**: Around 74% - 77% of cart sessions result in abandonment (positive class), while only ~23% - 26% convert. To mitigate this imbalance, inverse class weighting (`classWeightCol`) was applied during model training, and models are evaluated primarily on AUC-ROC, AUC-PR, and F1 rather than raw accuracy.
2. **Missing Price & Product Metadata**: The Retailrocket dataset omits product pricing, shipping fees, product categories (without external joins), and buyer demographics. Abandonment decisions in e-commerce are heavily influenced by unexpected shipping costs or item price comparisons, which are unobserved.
3. **Session Gap Assumption**: Sessions are delineated by a standard 30-minute inactivity threshold. In mobile and cross-device browsing, users frequently keep tabs open or return hours later to purchase.
4. **Prediction Point Boundary**: The prediction point is fixed strictly at the **first** `addtocart` event. Sessions adding multiple items subsequently are evaluated based on their state when the intent journey began.

