from pathlib import Path
import math
import time

import joblib
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "ml_training_data.csv"

DEMAND_MODEL_FILE = BASE_DIR / "demand_model.pkl"
STOCKOUT_MODEL_FILE = BASE_DIR / "stockout_model.pkl"
ANOMALY_MODEL_FILE = BASE_DIR / "anomaly_model.pkl"

OUTPUT_FILE = BASE_DIR / "merchandise_ai_result.csv"


# =========================================================
# FEATURES
# Must match train.py
# =========================================================

DEMAND_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_avg_7",
    "sales_avg_14",
]


STOCKOUT_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_avg_7",
    "sales_avg_14",
    "closing_stock",
    "supplier_lead_time_days",
    "safety_stock",
    "min_stock",
    "max_stock",
    "case_pack",
    "days_of_stock",
    "safety_stock_ratio",
]


ANOMALY_FEATURES = [
    "units_sold",
    "closing_stock",
    "received_qty",
    "transfer_in",
    "transfer_out",
    "damaged_qty",
    "expired_qty",
    "adjustment_qty",
]


# =========================================================
# REQUIRED DATA COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "store_name",
    "region",
    "store_type",
    "sku",
    "product_name",
    "category",
    "units_sold",
    "closing_stock",
    "received_qty",
    "transfer_in",
    "transfer_out",
    "damaged_qty",
    "expired_qty",
    "adjustment_qty",
    "supplier_lead_time_days",
    "safety_stock",
    "min_stock",
    "max_stock",
    "case_pack",
]


# =========================================================
# LOAD MODEL
# =========================================================

def load_model_bundle(
    file_path,
    default_features,
):

    bundle = joblib.load(
        file_path
    )

    if isinstance(bundle, dict):

        model = bundle["model"]

        features = bundle.get(
            "features",
            default_features,
        )

    else:

        model = bundle
        features = default_features

    return model, features


# =========================================================
# MAIN
# =========================================================

def main():

    start_time = time.time()

    print()
    print("=" * 65)
    print("Everrise AI - Batch Merchandise Scoring")
    print("=" * 65)

    # =====================================================
    # LOAD DATA
    # =====================================================

    print()
    print("Loading merchandise data...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=REQUIRED_COLUMNS,
        parse_dates=["date"],
        low_memory=False,
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.sort_values(
        [
            "sku",
            "store_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Products: {df['sku'].nunique():,}"
    )

    print(
        f"Stores: {df['store_id'].nunique():,}"
    )

    # =====================================================
    # LOAD MODELS
    # =====================================================

    print()
    print("Loading trained models...")

    demand_model, demand_features = (
        load_model_bundle(
            DEMAND_MODEL_FILE,
            DEMAND_FEATURES,
        )
    )

    stockout_model, stockout_features = (
        load_model_bundle(
            STOCKOUT_MODEL_FILE,
            STOCKOUT_FEATURES,
        )
    )

    anomaly_model, anomaly_features = (
        load_model_bundle(
            ANOMALY_MODEL_FILE,
            ANOMALY_FEATURES,
        )
    )

    print("Models loaded.")

    # =====================================================
    # SALES HISTORY MATRIX
    # =====================================================

    print()
    print(
        "Building SKU-store sales history..."
    )

    sales_matrix = df.pivot_table(
        index=[
            "sku",
            "store_id",
        ],
        columns="date",
        values="units_sold",
        aggfunc="sum",
        fill_value=0,
    )

    sales_matrix = (
        sales_matrix
        .sort_index(axis=1)
    )

    history = (
        sales_matrix
        .to_numpy(
            dtype=np.float32
        )
    )

    series_index = (
        sales_matrix.index
    )

    combination_count = len(
        series_index
    )

    print(
        f"SKU-store combinations: "
        f"{combination_count:,}"
    )

    if history.shape[1] < 15:

        raise ValueError(
            "At least 15 days of sales "
            "history are required."
        )

    # =====================================================
    # LATEST INVENTORY ROW
    # =====================================================

    latest = (
        df
        .drop_duplicates(
            subset=[
                "sku",
                "store_id",
            ],
            keep="last",
        )
        .set_index(
            [
                "sku",
                "store_id",
            ]
        )
        .reindex(
            series_index
        )
        .reset_index()
    )

    last_date = (
        sales_matrix
        .columns
        .max()
    )

    # =====================================================
    # FEATURE ENGINEERING
    #
    # Matches train.py:
    # shift(1) means current day's sales
    # are NOT used as sales_lag_1.
    # =====================================================

    sales_lag_1 = (
        history[:, -2]
    )

    sales_lag_7 = (
        history[:, -8]
    )

    sales_avg_7 = (
        history[:, -8:-1]
        .mean(axis=1)
    )

    sales_avg_14 = (
        history[:, -15:-1]
        .mean(axis=1)
    )

    latest[
        "sales_lag_1"
    ] = sales_lag_1

    latest[
        "sales_lag_7"
    ] = sales_lag_7

    latest[
        "sales_avg_7"
    ] = sales_avg_7

    latest[
        "sales_avg_14"
    ] = sales_avg_14

    latest[
        "day_of_week"
    ] = last_date.dayofweek

    latest[
        "month"
    ] = last_date.month

    latest[
        "is_weekend"
    ] = (
        1
        if last_date.dayofweek >= 5
        else 0
    )

    # =====================================================
    # DAYS OF STOCK
    # =====================================================

    safe_avg = (
        latest[
            "sales_avg_7"
        ]
        .replace(
            0,
            0.1,
        )
    )

    latest[
        "days_of_stock"
    ] = (
        latest[
            "closing_stock"
        ]
        / safe_avg
    )

    safe_safety = (
        latest[
            "safety_stock"
        ]
        .replace(
            0,
            1,
        )
    )

    latest[
        "safety_stock_ratio"
    ] = (
        latest[
            "closing_stock"
        ]
        / safe_safety
    )

    # =====================================================
    # DEMAND MODEL
    #
    # train.py target = future_7day_sales
    #
    # Therefore ONE prediction = next 7-day demand total.
    # =====================================================

    print()
    print(
        "Running XGBoost 7-day demand "
        "forecast in batch..."
    )

    X_demand = latest[
        demand_features
    ].copy()

    total_forecast = (
        demand_model.predict(
            X_demand
        )
    )

    total_forecast = (
        np.maximum(
            total_forecast,
            0,
        )
    )

    total_forecast = (
        np.rint(
            total_forecast
        )
        .astype(int)
    )

    latest[
        "total_forecast"
    ] = total_forecast

    latest[
        "average_daily_forecast"
    ] = (
        latest[
            "total_forecast"
        ]
        / 7
    )

    # =====================================================
    # STOCKOUT MODEL
    # =====================================================

    print(
        "Running XGBoost stockout "
        "classifier in batch..."
    )

    X_stockout = latest[
        stockout_features
    ].copy()

    stockout_probabilities = (
        stockout_model
        .predict_proba(
            X_stockout
        )[:, 1]
    )

    latest[
        "stockout_probability"
    ] = stockout_probabilities

    latest[
        "stockout_risk"
    ] = np.select(
        [
            stockout_probabilities >= 0.70,
            stockout_probabilities >= 0.40,
        ],
        [
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    print(
        "Running Isolation Forest "
        "in batch..."
    )

    X_anomaly = latest[
        anomaly_features
    ].copy()

    anomaly_predictions = (
        anomaly_model.predict(
            X_anomaly
        )
    )

    anomaly_scores = (
        anomaly_model
        .decision_function(
            X_anomaly
        )
    )

    latest[
        "anomaly_status"
    ] = np.where(
        anomaly_predictions == -1,
        "ANOMALY",
        "NORMAL",
    )

    latest[
        "anomaly_score"
    ] = anomaly_scores

    # =====================================================
    # REORDER BUSINESS RULES
    # =====================================================

    print(
        "Calculating inventory "
        "recommendations..."
    )

    current_stock = (
        latest[
            "closing_stock"
        ]
        .to_numpy(
            dtype=float
        )
    )

    safety_stock = (
        latest[
            "safety_stock"
        ]
        .to_numpy(
            dtype=float
        )
    )

    min_stock = (
        latest[
            "min_stock"
        ]
        .to_numpy(
            dtype=float
        )
    )

    max_stock = (
        latest[
            "max_stock"
        ]
        .to_numpy(
            dtype=float
        )
    )

    case_pack = (
        latest[
            "case_pack"
        ]
        .replace(
            0,
            1,
        )
        .to_numpy(
            dtype=float
        )
    )

    raw_order = np.maximum(
        0,
        total_forecast
        + safety_stock
        - current_stock,
    )

    recommended_cartons = (
        np.ceil(
            raw_order
            / case_pack
        )
        .astype(int)
    )

    recommended_order = (
        recommended_cartons
        * case_pack
    ).astype(int)

    # Never order if already overstocked
    overstock_mask = (
        current_stock
        > max_stock
    )

    recommended_cartons[
        overstock_mask
    ] = 0

    recommended_order[
        overstock_mask
    ] = 0

    latest[
        "recommended_cartons"
    ] = recommended_cartons

    latest[
        "recommended_order"
    ] = recommended_order

    # =====================================================
    # INVENTORY STATUS
    # =====================================================

    latest[
        "inventory_status"
    ] = np.select(
        [
            current_stock > max_stock,
            current_stock < min_stock,
            recommended_order > 0,
        ],
        [
            "OVERSTOCK",
            "BELOW MINIMUM STOCK",
            "REORDER REQUIRED",
        ],
        default="HEALTHY",
    )

    # =====================================================
    # RESULT TABLE
    # =====================================================

    latest[
        "date"
    ] = last_date

    result = latest[
        [
            "date",

            "store_id",
            "store_name",
            "region",
            "store_type",

            "sku",
            "product_name",
            "category",

            "closing_stock",

            "total_forecast",
            "average_daily_forecast",

            "stockout_probability",
            "stockout_risk",

            "anomaly_status",
            "anomaly_score",

            "days_of_stock",

            "supplier_lead_time_days",

            "safety_stock",
            "min_stock",
            "max_stock",
            "case_pack",

            "inventory_status",

            "recommended_order",
            "recommended_cartons",
        ]
    ].copy()

    result = result.rename(
        columns={
            "closing_stock":
                "current_stock",
        }
    )

    # =====================================================
    # ROUND DISPLAY VALUES
    # =====================================================

    result[
        "stockout_probability"
    ] = (
        result[
            "stockout_probability"
        ]
        .round(4)
    )

    result[
        "anomaly_score"
    ] = (
        result[
            "anomaly_score"
        ]
        .round(4)
    )

    result[
        "days_of_stock"
    ] = (
        result[
            "days_of_stock"
        ]
        .round(1)
    )

    result[
        "average_daily_forecast"
    ] = (
        result[
            "average_daily_forecast"
        ]
        .round(2)
    )

    # =====================================================
    # SAVE
    # =====================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 65)
    print("SCORING COMPLETED")
    print("=" * 65)

    print(
        f"Result rows: "
        f"{len(result):,}"
    )

    print(
        f"Products: "
        f"{result['sku'].nunique():,}"
    )

    print(
        f"Stores: "
        f"{result['store_id'].nunique():,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f} seconds"
    )

    print()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()