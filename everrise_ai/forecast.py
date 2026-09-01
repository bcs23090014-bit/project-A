from pathlib import Path

import math
import time

import joblib
import numpy as np
import pandas as pd

from google import genai

from api import GEMINI_API_KEY


# =====================================
# FILE LOCATIONS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "ml_training_data.csv"

DEMAND_MODEL_FILE = BASE_DIR / "demand_model.pkl"

STOCKOUT_MODEL_FILE = BASE_DIR / "stockout_model.pkl"

ANOMALY_MODEL_FILE = BASE_DIR / "anomaly_model.pkl"


# =====================================
# CHECK FILES
# =====================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "ml_training_data.csv was not found."
    )


if not DEMAND_MODEL_FILE.exists():

    raise FileNotFoundError(
        "demand_model.pkl was not found. "
        "Run train.py first."
    )


if not STOCKOUT_MODEL_FILE.exists():

    raise FileNotFoundError(
        "stockout_model.pkl was not found. "
        "Run train.py first."
    )


if not ANOMALY_MODEL_FILE.exists():

    raise FileNotFoundError(
        "anomaly_model.pkl was not found. "
        "Run train.py first."
    )


# =====================================
# LOAD MODELS
# =====================================

demand_bundle = joblib.load(
    DEMAND_MODEL_FILE
)

stockout_bundle = joblib.load(
    STOCKOUT_MODEL_FILE
)

anomaly_bundle = joblib.load(
    ANOMALY_MODEL_FILE
)


demand_model = (
    demand_bundle[
        "model"
    ]
)

stockout_model = (
    stockout_bundle[
        "model"
    ]
)

anomaly_model = (
    anomaly_bundle[
        "model"
    ]
)


DEMAND_FEATURES = (
    demand_bundle[
        "features"
    ]
)

STOCKOUT_FEATURES = (
    stockout_bundle[
        "features"
    ]
)

ANOMALY_FEATURES = (
    anomaly_bundle[
        "features"
    ]
)


# =====================================
# LOAD CSV ONCE
# =====================================

fabric_df = pd.read_csv(
    DATA_FILE
)

fabric_df["date"] = pd.to_datetime(
    fabric_df["date"]
)


# =====================================
# LOAD PRODUCT / STORE DATA
# =====================================

def load_series(
    sku,
    store
):

    sku_search = (
        str(sku)
        .strip()
        .upper()
    )

    store_search = (
        str(store)
        .strip()
        .lower()
    )


    product = fabric_df[

        (
            fabric_df["sku"]
            .astype(str)
            .str.upper()
            ==
            sku_search
        )

        &

        (
            (
                fabric_df["store_id"]
                .astype(str)
                .str.lower()
                ==
                store_search
            )

            |

            (
                fabric_df["store_name"]
                .astype(str)
                .str.lower()
                ==
                store_search
            )
        )

    ].sort_values(
        "date"
    ).copy()


    if product.empty:

        raise ValueError(
            f"No Fabric data found for "
            f"{sku} at {store}."
        )


    if len(product) < 14:

        raise ValueError(
            f"Not enough historical data for "
            f"{sku} at {store}."
        )


    return product


# =====================================
# XGBOOST REGRESSOR
# 7-DAY DEMAND FORECAST
# =====================================

def forecast_7_days(
    sku,
    store
):

    product = load_series(
        sku,
        store
    )


    history = (

        product[
            "units_sold"
        ]

        .astype(float)

        .tolist()

    )


    last_date = (
        product[
            "date"
        ].max()
    )


    forecasts = []


    for i in range(
        1,
        8
    ):

        next_date = (

            last_date

            +

            pd.Timedelta(
                days=i
            )

        )


        day_of_week = (
            next_date.dayofweek
        )


        is_weekend = (

            1

            if day_of_week >= 5

            else 0

        )


        feature_values = {

            "day_of_week":
                day_of_week,

            "month":
                next_date.month,

            "is_weekend":
                is_weekend,

            "sales_lag_1":
                history[-1],

            "sales_lag_7":
                history[-7],

            "sales_avg_7":
                float(
                    np.mean(
                        history[-7:]
                    )
                ),

            "sales_avg_14":
                float(
                    np.mean(
                        history[-14:]
                    )
                )

        }


        model_input = pd.DataFrame(
            [
                feature_values
            ]
        )


        model_input = (
            model_input[
                DEMAND_FEATURES
            ]
        )


        prediction = float(

            demand_model.predict(
                model_input
            )[0]

        )


        # Demand cannot be negative
        prediction = max(
            0,
            prediction
        )


        forecasts.append(
            {

                "date":
                    next_date.strftime(
                        "%Y-%m-%d"
                    ),

                "predicted_sales":
                    round(
                        prediction,
                        1
                    )

            }
        )


        # Recursive forecast
        history.append(
            prediction
        )


    return forecasts


# =====================================
# XGBOOST CLASSIFIER
# STOCKOUT PREDICTION
# =====================================

def predict_stockout(
    sku,
    store,
    current_stock=None
):

    product = load_series(
        sku,
        store
    )


    latest = (
        product.iloc[-1]
    )


    sales_history = (

        product[
            "units_sold"
        ]

        .astype(float)

        .tolist()

    )


    if current_stock is None:

        current_stock = int(
            latest[
                "closing_stock"
            ]
        )


    safety_stock = int(
        latest[
            "safety_stock"
        ]
    )


    lead_time = int(
        latest[
            "supplier_lead_time_days"
        ]
    )


    min_stock = int(
        latest[
            "min_stock"
        ]
    )


    max_stock = int(
        latest[
            "max_stock"
        ]
    )


    case_pack = int(
        latest[
            "case_pack"
        ]
    )


    sales_avg_7 = float(

        np.mean(
            sales_history[-7:]
        )

    )


    sales_avg_14 = float(

        np.mean(
            sales_history[-14:]
        )

    )


    day_of_week = (
        latest[
            "date"
        ].dayofweek
    )


    is_weekend = (

        1

        if day_of_week >= 5

        else 0

    )


    if sales_avg_7 > 0:

        days_of_stock = (

            current_stock
            /
            sales_avg_7

        )

    else:

        days_of_stock = 0


    if safety_stock > 0:

        safety_stock_ratio = (

            current_stock
            /
            safety_stock

        )

    else:

        safety_stock_ratio = 0


    feature_values = {

        "day_of_week":
            day_of_week,

        "month":
            latest[
                "date"
            ].month,

        "is_weekend":
            is_weekend,

        "sales_lag_1":
            sales_history[-1],

        "sales_lag_7":
            sales_history[-7],

        "sales_avg_7":
            sales_avg_7,

        "sales_avg_14":
            sales_avg_14,

        "closing_stock":
            current_stock,

        "supplier_lead_time_days":
            lead_time,

        "safety_stock":
            safety_stock,

        "min_stock":
            min_stock,

        "max_stock":
            max_stock,

        "case_pack":
            case_pack,

        "days_of_stock":
            days_of_stock,

        "safety_stock_ratio":
            safety_stock_ratio

    }


    model_input = pd.DataFrame(
        [
            feature_values
        ]
    )


    model_input = (
        model_input[
            STOCKOUT_FEATURES
        ]
    )


    probability = float(

        stockout_model.predict_proba(
            model_input
        )[0][1]

    )


    if probability >= 0.70:

        risk = "HIGH"

    elif probability >= 0.40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    return {

        "probability":
            probability,

        "risk":
            risk,

        "current_stock":
            current_stock,

        "safety_stock":
            safety_stock,

        "lead_time":
            lead_time,

        "min_stock":
            min_stock,

        "max_stock":
            max_stock,

        "case_pack":
            case_pack

    }


# =====================================
# ISOLATION FOREST
# ANOMALY DETECTION
# =====================================

def detect_anomaly(
    sku,
    store,
    current_stock=None
):

    product = load_series(
        sku,
        store
    )


    latest = (
        product.iloc[-1]
    )


    if current_stock is None:

        anomaly_stock = (
            latest[
                "closing_stock"
            ]
        )

    else:

        anomaly_stock = (
            current_stock
        )


    feature_values = {

        "units_sold":
            latest[
                "units_sold"
            ],

        "closing_stock":
            anomaly_stock,

        "received_qty":
            latest[
                "received_qty"
            ],

        "transfer_in":
            latest[
                "transfer_in"
            ],

        "transfer_out":
            latest[
                "transfer_out"
            ],

        "damaged_qty":
            latest[
                "damaged_qty"
            ],

        "expired_qty":
            latest[
                "expired_qty"
            ],

        "adjustment_qty":
            latest[
                "adjustment_qty"
            ]

    }


    model_input = pd.DataFrame(
        [
            feature_values
        ]
    )


    model_input = (
        model_input[
            ANOMALY_FEATURES
        ]
    )


    prediction = int(

        anomaly_model.predict(
            model_input
        )[0]

    )


    score = float(

        anomaly_model.decision_function(
            model_input
        )[0]

    )


    if prediction == -1:

        status = "ANOMALY"

    else:

        status = "NORMAL"


    return {

        "status":
            status,

        "score":
            round(
                score,
                4
            )

    }


# =====================================
# PYTHON MERCHANDISE
# RECOMMENDATION ENGINE
# =====================================

def create_recommendation(
    sku,
    store,
    current_stock=None
):

    product = load_series(
        sku,
        store
    )


    latest = (
        product.iloc[-1]
    )


    # =================================
    # PRODUCT INFORMATION
    # =================================

    product_name = str(
        latest[
            "product_name"
        ]
    )


    category = str(
        latest[
            "category"
        ]
    )


    store_id = str(
        latest[
            "store_id"
        ]
    )


    store_name = str(
        latest[
            "store_name"
        ]
    )


    region = str(
        latest[
            "region"
        ]
    )


    source_layer = str(
        latest[
            "source_layer"
        ]
    )


    if current_stock is None:

        current_stock = int(
            latest[
                "closing_stock"
            ]
        )


    # =================================
    # DEMAND FORECAST
    # =================================

    forecast = forecast_7_days(
        sku,
        store
    )


    total_forecast = round(

        sum(

            day[
                "predicted_sales"
            ]

            for day in forecast

        )

    )


    average_daily_demand = (

        total_forecast / 7

        if total_forecast > 0

        else 0

    )


    # =================================
    # STOCKOUT MODEL
    # =================================

    stockout = predict_stockout(

        sku=
            sku,

        store=
            store,

        current_stock=
            current_stock

    )


    safety_stock = (
        stockout[
            "safety_stock"
        ]
    )


    min_stock = (
        stockout[
            "min_stock"
        ]
    )


    max_stock = (
        stockout[
            "max_stock"
        ]
    )


    case_pack = (
        stockout[
            "case_pack"
        ]
    )


    lead_time = (
        stockout[
            "lead_time"
        ]
    )


    probability = (
        stockout[
            "probability"
        ]
    )


    risk = (
        stockout[
            "risk"
        ]
    )


    # =================================
    # ANOMALY MODEL
    # =================================

    anomaly = detect_anomaly(

        sku=
            sku,

        store=
            store,

        current_stock=
            current_stock

    )


    # =================================
    # DAYS OF STOCK
    # =================================

    if average_daily_demand > 0:

        days_of_stock = (

            current_stock
            /
            average_daily_demand

        )

    else:

        days_of_stock = 0


    # =================================
    # RAW REORDER REQUIREMENT
    # =================================

    raw_order = max(

        0,

        total_forecast
        +
        safety_stock
        -
        current_stock

    )


    # =================================
    # CASE PACK ROUNDING
    # =================================

    if raw_order > 0:

        cartons = math.ceil(

            raw_order
            /
            case_pack

        )


        recommended_order = (

            cartons
            *
            case_pack

        )

    else:

        cartons = 0

        recommended_order = 0


    # =================================
    # INVENTORY STATUS
    # =================================

    if current_stock > max_stock:

        inventory_status = (
            "OVERSTOCK"
        )


    elif current_stock < min_stock:

        inventory_status = (
            "BELOW MINIMUM STOCK"
        )


    elif recommended_order > 0:

        inventory_status = (
            "REORDER REQUIRED"
        )


    else:

        inventory_status = (
            "HEALTHY"
        )


    # =================================
    # RETURN RESULT
    # =================================

    return {

        "source_layer":
            source_layer,

        "sku":
            sku,

        "product_name":
            product_name,

        "category":
            category,

        "store_id":
            store_id,

        "store_name":
            store_name,

        "region":
            region,

        "current_stock":
            current_stock,

        "min_stock":
            min_stock,

        "max_stock":
            max_stock,

        "safety_stock":
            safety_stock,

        "supplier_lead_time_days":
            lead_time,

        "case_pack":
            case_pack,

        "forecast":
            forecast,

        "forecast_total":
            total_forecast,

        "average_daily_demand":
            round(
                average_daily_demand,
                1
            ),

        "days_of_stock":
            round(
                days_of_stock,
                1
            ),

        "stockout_probability":
            round(
                probability * 100,
                1
            ),

        "risk":
            risk,

        "anomaly_status":
            anomaly[
                "status"
            ],

        "anomaly_score":
            anomaly[
                "score"
            ],

        "inventory_status":
            inventory_status,

        "raw_order_requirement":
            raw_order,

        "recommended_cartons":
            cartons,

        "recommended_order":
            recommended_order

    }


# =====================================
# GEMINI AI
# 5-PILLAR PROMPT + STRUCTURED REASONING
# =====================================

def explain_all_with_gemini(
    results
):

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "Gemini API key is empty."
        )


    client = genai.Client(
        api_key=
            GEMINI_API_KEY
    )


    # =================================
    # BUILD PRODUCT DATA FOR AI
    # =================================

    product_text = ""


    for result in results:

        product_text += f"""

==================================================
PRODUCT
==================================================

Product: {result["product_name"]}
SKU: {result["sku"]}
Category: {result["category"]}

STORE
Store: {result["store_name"]}
Store ID: {result["store_id"]}
Region: {result["region"]}

INVENTORY
Current Stock: {result["current_stock"]} units
Minimum Stock: {result["min_stock"]} units
Maximum Stock: {result["max_stock"]} units
Safety Stock: {result["safety_stock"]} units
Estimated Days of Stock: {result["days_of_stock"]} days

DEMAND FORECAST
7-Day Demand Forecast: {result["forecast_total"]} units
Average Daily Demand: {result["average_daily_demand"]} units

XGBOOST STOCKOUT MODEL
Stockout Probability: {result["stockout_probability"]}%
Risk Level: {result["risk"]}

ISOLATION FOREST
Anomaly Status: {result["anomaly_status"]}
Anomaly Score: {result["anomaly_score"]}

INVENTORY STATUS
{result["inventory_status"]}

SUPPLIER / ORDER INFORMATION
Supplier Lead Time: {result["supplier_lead_time_days"]} days
Case Pack: {result["case_pack"]} units/carton

PYTHON RECOMMENDATION
Raw Requirement: {result["raw_order_requirement"]} units
Recommended Order: {result["recommended_order"]} units
Recommended Cartons: {result["recommended_cartons"]}

"""


    # =====================================
    # FIVE-PILLAR PROMPT
    # =====================================

    prompt = f"""

==================================================
PILLAR 1 — ROLE
==================================================

You are an AI Merchandise Decision Assistant
for Everrise supermarket.

You support the HQ merchandise manager by interpreting
machine-learning and inventory analysis results and
turning them into clear business actions.

You are an interpretation and decision-support layer.

You are NOT responsible for calculating new forecasts,
stockout probabilities, anomaly scores,
inventory quantities or reorder quantities.

All calculations have already been completed by
the machine-learning models and Python.


==================================================
PILLAR 2 — CONTEXT
==================================================

The merchandise information below represents data
extracted from a simulated Microsoft Fabric Gold Layer.

The data has already been processed by:

1. XGBoost Regressor

Purpose:
Predict future product demand.


2. XGBoost Classifier

Purpose:
Predict stockout probability and classify
stockout risk as HIGH, MEDIUM or LOW.


3. Isolation Forest

Purpose:
Detect unusual sales or inventory behaviour.


4. Python Merchandise Rules Engine

Purpose:
Calculate:

- Inventory status
- Safety-stock requirement
- Reorder requirement
- Case-pack/carton rounding
- Recommended order quantity


The following values are authoritative system outputs.

Do NOT replace them with your own calculations.


{product_text}


==================================================
PILLAR 3 — TASK
==================================================

Analyze all five products together from the perspective
of an Everrise HQ merchandise manager.

Your objectives are:

- Identify the products requiring immediate attention.

- Compare the stockout risk between products.

- Identify products with abnormal behaviour
  detected by Isolation Forest.

- Identify products with healthy inventory.

- Identify products with overstock.

- Identify products below minimum stock.

- Identify products requiring replenishment.

- Identify products that do not currently
  require replenishment.

- Review whether current inventory can support
  expected demand.

- Consider supplier lead time when discussing urgency.

- Explain why each recommended action makes
  business sense.

- Prioritize the most important merchandise actions.

- Give the merchandise manager a short,
  useful decision summary.


==================================================
STRUCTURED REASONING PROCESS
==================================================

Before producing your final answer,
evaluate each product internally using
the following sequence.

STEP 1 — INVENTORY POSITION

Review:

- Current Stock
- Minimum Stock
- Maximum Stock
- Safety Stock

Determine whether inventory is:

- Below minimum
- Within a normal range
- Above maximum


STEP 2 — DEMAND

Review:

- 7-Day Demand Forecast
- Average Daily Demand

Understand how quickly the product
is expected to move.


STEP 3 — DAYS OF STOCK

Review Estimated Days of Stock.

Consider whether the current inventory
can support expected demand.


STEP 4 — STOCKOUT MODEL

Review:

- Stockout Probability
- Risk Level

HIGH risk should receive more attention
than MEDIUM or LOW risk.


STEP 5 — ANOMALY MODEL

Review:

- Anomaly Status
- Anomaly Score

If the status is ANOMALY,
recommend that the merchandise manager
checks the unusual inventory or sales activity.

The anomaly score is NOT a percentage.

Lower anomaly scores indicate a more
unusual observation.


STEP 6 — SUPPLIER LEAD TIME

Compare supplier lead time with
Estimated Days of Stock.

If available stock may not survive
the supplier lead time,
treat the situation as more urgent.


STEP 7 — REORDER RECOMMENDATION

Review:

- Raw Requirement
- Recommended Order
- Recommended Cartons
- Case Pack

Use the Python recommendation exactly.

If the recommended order was rounded upward
because of the case pack,
mention this clearly.


STEP 8 — PRIORITY

Prioritize merchandise issues generally
in this order:

1. HIGH stockout risk
2. Below minimum stock
3. Anomaly requiring investigation
4. Reorder required
5. MEDIUM stockout risk
6. Healthy inventory
7. Overstock / slow-moving concern


Perform this structured reasoning internally.

Do NOT output hidden chain-of-thought,
private reasoning or a long step-by-step
reasoning transcript.

Only provide concise business rationale
supporting the final decisions.


==================================================
PILLAR 4 — CONSTRAINTS
==================================================

You MUST follow these rules:

- Use only the supplied system results.

- Do NOT recalculate numerical values.

- Do NOT change any calculated value.

- Do NOT invent sales data.

- Do NOT invent inventory data.

- Do NOT invent supplier information.

- Do NOT invent purchase orders.

- Do NOT invent promotions.

- Do NOT invent stock transfers.

- Do NOT invent delivery dates.

- Do NOT invent product information.

- Do NOT create new stockout probabilities.

- Do NOT modify anomaly scores.

- Do NOT modify recommended order quantities.

- Do NOT modify recommended carton quantities.


STOCKOUT PROBABILITY:

Treat the supplied percentage as the
official XGBoost prediction.


ANOMALY DETECTION:

Use the supplied NORMAL or ANOMALY
classification as the main indicator.

The anomaly score is not a percentage.


RECOMMENDED ORDER:

Use exactly the Python-calculated
recommended order.

When an order exists,
mention carton/case-pack rounding.


MISSING INFORMATION:

If information is not supplied,
state that the information is unavailable.

Do not guess.


COMMUNICATION STYLE:

- Simple business English
- Professional
- Concise
- Management-friendly
- Action-oriented


==================================================
PILLAR 5 — OUTPUT FORMAT
==================================================

Return exactly these seven sections.


1. Merchandise Overview

Give a short summary of the overall
inventory situation across all five products.


2. Products Requiring Immediate Attention

List the most urgent products first.

For each relevant product use:

Product:
Issue:
Reason:
Action:


3. Stockout Risk

Compare products based on:

- HIGH
- MEDIUM
- LOW

Include the supplied stockout probability
where useful.


4. Anomaly Detection

Identify products marked ANOMALY.

Explain briefly what the merchandise
manager should verify.

If all products are NORMAL,
clearly state that no anomaly
was detected in the latest records.


5. Healthy / Overstock Inventory

Identify:

- Healthy products
- Overstock products
- Products that currently
  do not require ordering


6. Recommended Actions

Create a priority action list.

Use:

Priority:
Product:
Action:
Recommended Order:
Cartons:


7. Management Summary

Finish with a short executive summary.

Give the 2 to 4 most important decisions
the merchandise manager should take.

Do not provide hidden chain-of-thought.

Only provide concise decision rationale.

"""


    # =====================================
    # GEMINI MODEL FALLBACK
    # =====================================

    models = [

        "gemini-3.7-flash",

        "gemini-3.6-flash",

        "gemini-3.5-flash-lite"

    ]


    # =====================================
    # TRY GEMINI MODELS
    # =====================================

    for model_name in models:

        print()

        print(
            "Trying Gemini model:",
            model_name
        )


        for attempt in range(3):

            try:

                response = (
                    client.models.generate_content(

                        model=
                            model_name,

                        contents=
                            prompt

                    )
                )


                print(
                    "Gemini response received from",
                    model_name
                )


                return (
                    response.text
                )


            except Exception as error:

                error_text = str(
                    error
                )


                # =================================
                # TEMPORARY API ERRORS
                # =================================

                if (
                    "503"
                    in error_text

                    or

                    "UNAVAILABLE"
                    in error_text

                    or

                    "429"
                    in error_text

                    or

                    "RESOURCE_EXHAUSTED"
                    in error_text
                ):

                    wait_time = (

                        3
                        *
                        (2 ** attempt)

                    )


                    print(
                        model_name,
                        "temporarily unavailable."
                    )


                    if attempt < 2:

                        print(
                            "Retrying in",
                            wait_time,
                            "seconds..."
                        )


                        time.sleep(
                            wait_time
                        )


                    else:

                        print(
                            model_name,
                            "failed after 3 attempts."
                        )


                else:

                    print(
                        model_name,
                        "returned an error:"
                    )


                    print(
                        error
                    )


                    break


    raise RuntimeError(
        "All Gemini models are currently unavailable."
    )


# =====================================
# MAIN PROGRAM
# =====================================

if __name__ == "__main__":


    # =================================
    # STORE TO ANALYZE
    # =================================

    STORE = "Vivacity"


    # =================================
    # FIVE PRODUCTS
    # =================================

    PRODUCTS = [

        "COKE15L",

        "MILO1KG",

        "RICE5KG",

        "BREAD400G",

        "DETERGENT2KG"

    ]


    results = []


    print()

    print(
        "=========================================="
    )

    print(
        "EVERISE AI MERCHANDISE ANALYSIS"
    )

    print(
        "=========================================="
    )


    print(
        "Store:",
        STORE
    )


    print(
        "Data Source: "
        "Microsoft Fabric Gold Layer (Simulated)"
    )


    print()

    print(
        "ML / AI COMPONENTS:"
    )

    print(
        "1. XGBoost Demand Forecast"
    )

    print(
        "2. XGBoost Stockout Prediction"
    )

    print(
        "3. Isolation Forest Anomaly Detection"
    )

    print(
        "4. Python Merchandise Rules"
    )

    print(
        "5. Gemini AI Decision Assistant"
    )


    # =====================================
    # PROCESS ALL FIVE PRODUCTS
    # =====================================

    for sku in PRODUCTS:

        print()
        print()

        print(
            "=========================================="
        )

        print(
            "ANALYZING:",
            sku
        )

        print(
            "=========================================="
        )


        result = (
            create_recommendation(

                sku=
                    sku,

                store=
                    STORE,

                current_stock=
                    None

            )
        )


        results.append(
            result
        )


        # =================================
        # PRODUCT
        # =================================

        print()

        print(
            "PRODUCT INFORMATION"
        )

        print(
            "------------------------------------------"
        )


        print(
            "Product:",
            result[
                "product_name"
            ]
        )


        print(
            "SKU:",
            result[
                "sku"
            ]
        )


        print(
            "Category:",
            result[
                "category"
            ]
        )


        print(
            "Store:",
            result[
                "store_name"
            ],
            f"({result['store_id']})"
        )


        # =================================
        # INVENTORY
        # =================================

        print()

        print(
            "INVENTORY"
        )

        print(
            "------------------------------------------"
        )


        print(
            "Current Stock:",
            result[
                "current_stock"
            ],
            "units"
        )


        print(
            "Minimum Stock:",
            result[
                "min_stock"
            ],
            "units"
        )


        print(
            "Maximum Stock:",
            result[
                "max_stock"
            ],
            "units"
        )


        print(
            "Safety Stock:",
            result[
                "safety_stock"
            ],
            "units"
        )


        # =================================
        # DEMAND FORECAST
        # =================================

        print()

        print(
            "7-DAY DEMAND FORECAST"
        )

        print(
            "------------------------------------------"
        )


        for day in result[
            "forecast"
        ]:

            print(

                day[
                    "date"
                ],

                ":",

                day[
                    "predicted_sales"
                ],

                "units"

            )


        # =================================
        # AI / ML ANALYSIS
        # =================================

        print()

        print(
            "AI / ML ANALYSIS"
        )

        print(
            "------------------------------------------"
        )


        print(
            "7-Day Demand:",
            result[
                "forecast_total"
            ],
            "units"
        )


        print(
            "Average Daily Demand:",
            result[
                "average_daily_demand"
            ],
            "units"
        )


        print(
            "Estimated Days of Stock:",
            result[
                "days_of_stock"
            ],
            "days"
        )


        print()

        print(
            "Stockout Probability:",
            result[
                "stockout_probability"
            ],
            "%"
        )


        print(
            "Stockout Risk:",
            result[
                "risk"
            ]
        )


        print()

        print(
            "Anomaly Detection:",
            result[
                "anomaly_status"
            ]
        )


        print(
            "Anomaly Score:",
            result[
                "anomaly_score"
            ]
        )


        print()

        print(
            "Inventory Status:",
            result[
                "inventory_status"
            ]
        )


        # =================================
        # RECOMMENDATION
        # =================================

        print()

        print(
            "MERCHANDISE RECOMMENDATION"
        )

        print(
            "------------------------------------------"
        )


        print(
            "Supplier Lead Time:",
            result[
                "supplier_lead_time_days"
            ],
            "days"
        )


        print(
            "Case Pack:",
            result[
                "case_pack"
            ],
            "units/carton"
        )


        print(
            "Raw Requirement:",
            result[
                "raw_order_requirement"
            ],
            "units"
        )


        print(
            "Recommended Cartons:",
            result[
                "recommended_cartons"
            ]
        )


        print(
            "Recommended Order:",
            result[
                "recommended_order"
            ],
            "units"
        )


    # =====================================
    # MERCHANDISE SUMMARY TABLE
    # =====================================

    summary_rows = []


    for result in results:

        summary_rows.append(
            {

                "Product":
                    result[
                        "product_name"
                    ],

                "Stock":
                    result[
                        "current_stock"
                    ],

                "7D Forecast":
                    result[
                        "forecast_total"
                    ],

                "Days Stock":
                    result[
                        "days_of_stock"
                    ],

                "Stockout %":
                    result[
                        "stockout_probability"
                    ],

                "Risk":
                    result[
                        "risk"
                    ],

                "Anomaly":
                    result[
                        "anomaly_status"
                    ],

                "Status":
                    result[
                        "inventory_status"
                    ],

                "Order":
                    result[
                        "recommended_order"
                    ],

                "Cartons":
                    result[
                        "recommended_cartons"
                    ]

            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    print()
    print()

    print(
        "=========================================="
    )

    print(
        "MERCHANDISE SUMMARY"
    )

    print(
        "=========================================="
    )

    print()


    print(
        summary_df.to_string(
            index=False
        )
    )


    # =====================================
    # GEMINI AI MANAGEMENT ANALYSIS
    # =====================================

    print()
    print()

    print(
        "------------------------------------------"
    )

    print(
        "GENERATING AI MERCHANDISE SUMMARY..."
    )

    print(
        "------------------------------------------"
    )


    try:

        explanation = (
            explain_all_with_gemini(
                results
            )
        )


        print()

        print(
            "AI MERCHANDISE ANALYSIS"
        )

        print(
            "=========================================="
        )


        print(
            explanation
        )


        print(
            "=========================================="
        )


    except Exception as error:

        print()

        print(
            "Gemini explanation failed."
        )


        print(
            "The XGBoost, Isolation Forest "
            "and Python merchandise results "
            "above are still valid."
        )


        print()

        print(
            "Gemini Error:"
        )


        print(
            error
        )