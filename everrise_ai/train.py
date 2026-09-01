from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)


# =====================================
# FILE LOCATIONS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "ml_training_data.csv"

DEMAND_MODEL_FILE = BASE_DIR / "demand_model.pkl"

STOCKOUT_MODEL_FILE = BASE_DIR / "stockout_model.pkl"

ANOMALY_MODEL_FILE = BASE_DIR / "anomaly_model.pkl"


# =====================================
# CHECK DATA FILE
# =====================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "ml_training_data.csv was not found. "
        "Put it inside the everrise_ai folder."
    )


# =====================================
# LOAD FABRIC DATA
# =====================================

print()
print("Loading Fabric merchandise data...")


df = pd.read_csv(
    DATA_FILE
)


print(
    "Total rows:",
    len(df)
)


# =====================================
# REQUIRED COLUMNS
# =====================================

required_columns = [

    "date",
    "store_id",
    "store_name",

    "sku",

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

    "case_pack"

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# =====================================
# DATE CONVERSION
# =====================================

df["date"] = pd.to_datetime(
    df["date"]
)


# =====================================
# SORT DATA
# =====================================

df = df.sort_values(

    [
        "sku",
        "store_id",
        "date"
    ]

).reset_index(
    drop=True
)


# =====================================
# DATE FEATURES
# =====================================

df["day_of_week"] = (
    df["date"]
    .dt.dayofweek
)


df["month"] = (
    df["date"]
    .dt.month
)


df["is_weekend"] = (

    df["day_of_week"]
    >= 5

).astype(int)


# =====================================
# GROUP SALES BY SKU + STORE
# =====================================

group = df.groupby(

    [
        "sku",
        "store_id"
    ]

)["units_sold"]


# =====================================
# SALES HISTORY FEATURES
# =====================================

df["sales_lag_1"] = (
    group.shift(1)
)


df["sales_lag_7"] = (
    group.shift(7)
)


df["sales_avg_7"] = (

    group.transform(

        lambda series:

        series
        .shift(1)
        .rolling(7)
        .mean()

    )

)


df["sales_avg_14"] = (

    group.transform(

        lambda series:

        series
        .shift(1)
        .rolling(14)
        .mean()

    )

)


# =====================================
# FUTURE 7-DAY SALES
# =====================================

future_columns = []


for day in range(
    1,
    8
):

    column_name = (
        f"future_sales_{day}"
    )


    df[column_name] = (
        group.shift(
            -day
        )
    )


    future_columns.append(
        column_name
    )


df["future_7day_sales"] = (

    df[
        future_columns
    ]

    .sum(
        axis=1,
        min_count=7
    )

)


# =====================================
# STOCKOUT TARGET
# =====================================

# 1 = current stock will not cover
# future 7-day demand
#
# 0 = stock should be enough

df["stockout"] = (

    df["closing_stock"]
    <
    df["future_7day_sales"]

).astype(int)


# =====================================
# INVENTORY FEATURES
# =====================================

df["days_of_stock"] = (

    df["closing_stock"]

    /

    df["sales_avg_7"].replace(
        0,
        0.1
    )

)


df["safety_stock_ratio"] = (

    df["closing_stock"]

    /

    df["safety_stock"].replace(
        0,
        1
    )

)


# =====================================
# DEMAND MODEL FEATURES
# =====================================

DEMAND_FEATURES = [

    "day_of_week",

    "month",

    "is_weekend",

    "sales_lag_1",

    "sales_lag_7",

    "sales_avg_7",

    "sales_avg_14"

]


# =====================================
# STOCKOUT MODEL FEATURES
# =====================================

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

    "safety_stock_ratio"

]


# =====================================
# ANOMALY MODEL FEATURES
# =====================================

ANOMALY_FEATURES = [

    "units_sold",

    "closing_stock",

    "received_qty",

    "transfer_in",

    "transfer_out",

    "damaged_qty",

    "expired_qty",

    "adjustment_qty"

]


# =====================================
# REMOVE INCOMPLETE ROWS
# =====================================

model_df = df.dropna(

    subset=(

        STOCKOUT_FEATURES

        + ANOMALY_FEATURES

        + [
            "future_7day_sales",
            "units_sold"
        ]

    )

).copy()


print(
    "Usable ML rows:",
    len(model_df)
)


# =====================================
# TIME-BASED SPLIT
# =====================================

dates_available = sorted(

    model_df[
        "date"
    ].unique()

)


cutoff_index = int(

    len(dates_available)
    * 0.80

)


cutoff_date = (

    dates_available[
        cutoff_index
    ]

)


train_df = model_df[

    model_df["date"]
    <
    cutoff_date

].copy()


test_df = model_df[

    model_df["date"]
    >=
    cutoff_date

].copy()


print()

print(
    "Training rows:",
    len(train_df)
)


print(
    "Testing rows:",
    len(test_df)
)


print(
    "Test period starts:",
    pd.Timestamp(
        cutoff_date
    ).date()
)


# =====================================
# XGBOOST DEMAND MODEL
# =====================================

X_train = (
    train_df[
        DEMAND_FEATURES
    ]
)


y_train = (
    train_df[
        "units_sold"
    ]
)


X_test = (
    test_df[
        DEMAND_FEATURES
    ]
)


y_test = (
    test_df[
        "units_sold"
    ]
)


demand_model = XGBRegressor(

    n_estimators=350,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    objective="reg:squarederror",

    random_state=42

)


print()
print("-----------------------------------")
print("TRAINING XGBOOST DEMAND MODEL")
print("-----------------------------------")


demand_model.fit(
    X_train,
    y_train
)


demand_predictions = (
    demand_model.predict(
        X_test
    )
)


demand_mae = (
    mean_absolute_error(

        y_test,

        demand_predictions

    )
)


# =====================================
# SAVE DEMAND MODEL
# =====================================

joblib.dump(

    {

        "model":
            demand_model,

        "features":
            DEMAND_FEATURES

    },

    DEMAND_MODEL_FILE

)


# =====================================
# XGBOOST STOCKOUT MODEL
# =====================================

Xs_train = (
    train_df[
        STOCKOUT_FEATURES
    ]
)


ys_train = (
    train_df[
        "stockout"
    ]
)


Xs_test = (
    test_df[
        STOCKOUT_FEATURES
    ]
)


ys_test = (
    test_df[
        "stockout"
    ]
)


print()
print("Stockout training distribution:")

print(
    ys_train.value_counts()
)


stockout_model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    eval_metric="logloss",

    random_state=42

)


print()
print("-----------------------------------")
print("TRAINING XGBOOST STOCKOUT MODEL")
print("-----------------------------------")


stockout_model.fit(
    Xs_train,
    ys_train
)


stockout_predictions = (
    stockout_model.predict(
        Xs_test
    )
)


stockout_probabilities = (
    stockout_model.predict_proba(
        Xs_test
    )[:, 1]
)


stockout_accuracy = (
    accuracy_score(

        ys_test,

        stockout_predictions

    )
)


try:

    stockout_auc = (
        roc_auc_score(

            ys_test,

            stockout_probabilities

        )
    )


    stockout_auc_text = (
        f"{stockout_auc:.3f}"
    )


except ValueError:

    stockout_auc_text = "N/A"


# =====================================
# SAVE STOCKOUT MODEL
# =====================================

joblib.dump(

    {

        "model":
            stockout_model,

        "features":
            STOCKOUT_FEATURES

    },

    STOCKOUT_MODEL_FILE

)


# =====================================
# ISOLATION FOREST
# ANOMALY DETECTION
# =====================================

Xa_train = (
    train_df[
        ANOMALY_FEATURES
    ]
)


Xa_test = (
    test_df[
        ANOMALY_FEATURES
    ]
)


anomaly_model = IsolationForest(

    n_estimators=250,

    contamination=0.03,

    random_state=42

)


print()
print("-----------------------------------")
print("TRAINING ISOLATION FOREST")
print("-----------------------------------")


anomaly_model.fit(
    Xa_train
)


# =====================================
# TEST ANOMALY MODEL
# =====================================

anomaly_predictions = (

    anomaly_model.predict(
        Xa_test
    )

)


# Isolation Forest:
#
# 1  = normal
# -1 = anomaly

anomaly_count = int(

    (
        anomaly_predictions
        ==
        -1
    ).sum()

)


normal_count = int(

    (
        anomaly_predictions
        ==
        1
    ).sum()

)


# =====================================
# SAVE ANOMALY MODEL
# =====================================

joblib.dump(

    {

        "model":
            anomaly_model,

        "features":
            ANOMALY_FEATURES

    },

    ANOMALY_MODEL_FILE

)


# =====================================
# FINAL RESULTS
# =====================================

print()
print("===================================")
print("TRAINING COMPLETE")
print("===================================")


print()

print(
    "Demand Forecast MAE:",
    round(
        demand_mae,
        2
    ),
    "units"
)


print(
    "Stockout Accuracy:",
    round(
        stockout_accuracy,
        3
    )
)


print(
    "Stockout ROC-AUC:",
    stockout_auc_text
)


print()

print(
    "Isolation Forest Test Records:"
)


print(
    "Normal:",
    normal_count
)


print(
    "Anomalies:",
    anomaly_count
)


print()

print(
    "Models saved:"
)


print(
    DEMAND_MODEL_FILE
)


print(
    STOCKOUT_MODEL_FILE
)


print(
    ANOMALY_MODEL_FILE
)