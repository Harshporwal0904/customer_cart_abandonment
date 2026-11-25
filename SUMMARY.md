# Project Summary: Predicting Online Shopping Cart Abandonment

A PySpark Big Data & Machine Learning project analyzing the **Retailrocket dataset** on Fedora 44 Linux with PySpark 4.3.0 and an interactive Streamlit dashboard.

---

## 1. Execution Commands (In Order)

All commands must be executed from the project root using `.venv/bin/python`.

### A. Fast Development / Sample Run (10% Visitor Sample)
```bash
.venv/bin/python src/01_ingest_clean.py --sample 0.1
.venv/bin/python src/02_sessionize_label.py
.venv/bin/python src/03_features.py
.venv/bin/python src/04_train_evaluate.py
.venv/bin/python src/05_export_dashboard_data.py
```

### B. Full Dataset Production Run
```bash
.venv/bin/python src/01_ingest_clean.py
.venv/bin/python src/02_sessionize_label.py
.venv/bin/python src/03_features.py
.venv/bin/python src/04_train_evaluate.py
.venv/bin/python src/05_export_dashboard_data.py
```

### C. Launch the Streamlit Dashboard
```bash
.venv/bin/streamlit run app/dashboard.py
```
*(The dashboard runs independently of Spark by reading the lightweight aggregated files in `outputs/` with `@st.cache_data`)*

---

## 2. Summary of Model Results (Full Dataset)

### Dataset Overview
- **Raw Events**: 2,756,101
- **Cleaned Events**: 2,755,641 (1,407,580 unique visitors)
- **Total Sessions (30-min window)**: 1,761,660
- **Cart Sessions**: 43,924
  - **Abandoned Sessions (`abandoned = 1`)**: 32,164 (**73.23%**)
  - **Converted Sessions (`abandoned = 0`)**: 11,760 (**26.77%**)
- **Date Range**: 2015-05-03 08:30:04 to 2015-09-18 08:29:47 UTC

### Train / Test Split & Weighting
- **Temporal Split**: Earliest ~80% sessions (35,136 sessions) for training; latest ~20% sessions (8,788 sessions) for testing.
- **Class Balancing**: Applied inverse class weighting (`class_weight`: 1.8628 for converted class 0, 0.6834 for abandoned class 1).

### Model Performance Comparison (Test Set)

| Metric | Logistic Regression (Baseline) | GBTClassifier |
| :--- | :---: | :---: |
| **AUC-ROC** | 0.6365 | **0.6791** |
| **AUC-PR** | 0.7945 | **0.8386** |
| **Weighted F1** | **0.7071** | 0.6878 |
| **Precision (Weighted)** | 0.6997 | 0.6990 |
| **Recall (Weighted)** | **0.7251** | 0.6798 |
| **Abandoned Precision (Class 1)** | 0.7814 | **0.8024** |
| **Abandoned Recall (Class 1)** | **0.8690** | 0.7487 |
| **Abandoned F1 (Class 1)** | **0.8229** | 0.7746 |

### Confusion Matrix (Test Set: 6,459 Abandoned, 2,329 Converted)
- **Logistic Regression**:
  - True Positives (TP): 5,613
  - False Positives (FP): 1,570
  - False Negatives (FN): 846
  - True Negatives (TN): 759
- **GBTClassifier**:
  - True Positives (TP): 4,836
  - False Positives (FP): 1,191
  - False Negatives (FN): 1,623
  - True Negatives (TN): 1,138

### Relative Feature Importances (GBTClassifier)
1. `prior_purchases` (**30.13%**): Visitors with prior transactions are substantially more likely to complete purchase.
2. `hour_of_day` (**18.70%**): Peak shopping vs. casual daytime/night browsing patterns.
3. `seconds_to_cart` (**12.62%**): Velocity to add items correlates strongly with immediate purchase intent.
4. `day_of_week` (**11.14%**): Weekday vs. weekend behavioral shifts.
5. `prior_sessions` (**11.00%**): Visitor familiarity and loyalty across repeat sessions.
6. `views_before_cart` (**9.65%**): Depth of product catalog browsing prior to carting.
7. `distinct_items_viewed` (**6.77%**): Exploration across different items in session.

---

## 3. Assumptions Made

1. **Session Boundary**: A 30-minute inactivity threshold (`gap_seconds > 1800`) defines distinct browsing sessions per visitor using PySpark window functions (`lag` and cumulative `sum`).
2. **Prediction Point & Target Definition**:
   - The prediction point is triggered at the **first `addtocart`** event within a session.
   - A session is marked as **`abandoned = 1`** if no `transaction` event occurs at or after that first `addtocart` in the session. If a transaction occurs, `abandoned = 0`.
3. **Data Leakage Boundaries**:
   - **Intra-session features** (`views_before_cart`, `distinct_items_viewed`, `seconds_to_cart`) strictly include events where `timestamp <= first_cart_ts`.
   - **Historical visitor features** (`prior_sessions`, `prior_purchases`) strictly include earlier sessions (`session_start_time < current_session_start_time`).
4. **Data Cleaning & Schema**:
   - Explicit schema enforcement on `events.csv`.
   - Rows missing `visitorid`, `timestamp`, `event`, or `itemid` are dropped.
   - Exact duplicate events are dropped. `transactionid` is permitted to be null for `view` and `addtocart` events.
5. **Class Imbalance Handling**:
   - Addressed via inverse class frequency weighting (`total_count / (2 * class_count)`) attached to the PySpark ML pipeline (`classWeightCol`), avoiding synthetic distortion of feature distributions.
6. **Decoupled Dashboard Architecture**:
   - The Streamlit dashboard does not initialize a Spark cluster; it reads lightweight pre-computed summaries, CSVs, and JSONs from `outputs/` using `@st.cache_data`.

