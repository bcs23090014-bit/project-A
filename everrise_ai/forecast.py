import math
import time

from google import genai

from api import GEMINI_API_KEY


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# If Google AI Studio gives you a different
# model name, change it here.
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


# =========================================================
# SAFE VALUE
# =========================================================

def get_value(
    item,
    *keys,
    default=0,
):

    for key in keys:

        if key not in item:
            continue

        value = item.get(
            key
        )

        if value is None:
            continue

        if (
            isinstance(
                value,
                float,
            )
            and
            math.isnan(
                value
            )
        ):
            continue

        return value

    return default


# =========================================================
# GEMINI EXPLANATION
# =========================================================

def explain_all_with_gemini(
    results,
):

    if not results:

        return (
            "No merchandise results "
            "available for AI analysis."
        )

    product_sections = []

    # =====================================================
    # CREATE SMALL AI INPUT
    # =====================================================

    for number, result in enumerate(
        results,
        start=1,
    ):

        sku = get_value(
            result,
            "sku",
            default="Unknown SKU",
        )

        product_name = get_value(
            result,
            "product_name",
            default=sku,
        )

        store_name = get_value(
            result,
            "store_name",
            default="Unknown Store",
        )

        current_stock = get_value(
            result,
            "current_stock",
            "closing_stock",
            default=0,
        )

        # Supports both names
        total_forecast = get_value(
            result,
            "total_forecast",
            "forecast_total",
            default=0,
        )

        average_daily_forecast = (
            get_value(
                result,
                "average_daily_forecast",
                "avg_daily_forecast",
                default=0,
            )
        )

        days_of_stock = get_value(
            result,
            "days_of_stock",
            default=0,
        )

        probability = get_value(
            result,
            "stockout_probability",
            default=0,
        )

        risk = get_value(
            result,
            "stockout_risk",
            "risk",
            default="UNKNOWN",
        )

        anomaly_status = get_value(
            result,
            "anomaly_status",
            "anomaly",
            default="NORMAL",
        )

        anomaly_score = get_value(
            result,
            "anomaly_score",
            default=0,
        )

        inventory_status = get_value(
            result,
            "inventory_status",
            "status",
            default="UNKNOWN",
        )

        safety_stock = get_value(
            result,
            "safety_stock",
            default=0,
        )

        min_stock = get_value(
            result,
            "min_stock",
            default=0,
        )

        max_stock = get_value(
            result,
            "max_stock",
            default=0,
        )

        lead_time = get_value(
            result,
            "supplier_lead_time_days",
            "lead_time",
            default=0,
        )

        case_pack = get_value(
            result,
            "case_pack",
            default=1,
        )

        recommended_order = get_value(
            result,
            "recommended_order",
            "reorder_qty",
            default=0,
        )

        recommended_cartons = get_value(
            result,
            "recommended_cartons",
            "cartons",
            default=0,
        )

        # =================================================
        # FORMAT STOCKOUT %
        # =================================================

        try:

            probability = float(
                probability
            )

            if probability <= 1:

                probability_percent = (
                    probability
                    * 100
                )

            else:

                probability_percent = (
                    probability
                )

        except Exception:

            probability_percent = 0.0

        # =================================================
        # PRODUCT SUMMARY
        # =================================================

        product_text = f"""
Product #{number}

Product: {product_name}
SKU: {sku}
Store: {store_name}

Inventory
- Current Stock: {current_stock}
- Safety Stock: {safety_stock}
- Minimum Stock: {min_stock}
- Maximum Stock: {max_stock}
- Inventory Status: {inventory_status}
- Days of Stock: {days_of_stock}

Demand Forecast
- 7-Day Demand: {total_forecast}
- Average Daily Demand: {average_daily_forecast}

Stockout Prediction
- Probability: {probability_percent:.1f}%
- Risk: {risk}

Anomaly Detection
- Status: {anomaly_status}
- Score: {anomaly_score}

Supply
- Supplier Lead Time: {lead_time} days
- Case Pack: {case_pack}

Recommendation
- Recommended Order: {recommended_order} units
- Recommended Cartons: {recommended_cartons}
"""

        product_sections.append(
            product_text
        )

    merchandise_data = (
        "\n".join(
            product_sections
        )
    )

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
ROLE

You are an AI Merchandise Decision Assistant
for Everrise supermarket management.

Your job is to explain the supplied machine-learning
and inventory results.

You are not responsible for calculating new forecasts,
stockout probabilities or order quantities.


CONTEXT

The results were generated using:

- Microsoft Fabric-style merchandise data
- XGBoost 7-day demand forecasting
- XGBoost stockout classification
- Isolation Forest anomaly detection
- Python inventory and replenishment rules

All supplied numerical values are official system
results.


TASK

Provide management with a concise merchandise
intelligence summary.

Identify:

1. Products requiring immediate attention.
2. Understock and replenishment problems.
3. Overstock products.
4. Important anomalies.
5. Healthy inventory.
6. Recommended actions.


PRIORITY

Always prioritize products in this order:

1. UNDERSTOCK / BELOW MINIMUM /
   REORDER REQUIRED

2. OVERSTOCK

3. HEALTHY


Within understock products:

HIGH stockout risk first,
then MEDIUM,
then LOW.


Consider:

- Current stock
- 7-day demand
- Days of stock
- Stockout probability
- Stockout risk
- Supplier lead time
- Anomaly status
- Recommended order quantity


CONSTRAINTS

Use only the supplied results.

Do not invent:

- sales
- inventory
- suppliers
- promotions
- purchase orders
- transfers
- delivery dates
- logistics events

Do not recalculate or modify:

- demand forecast
- stockout probability
- anomaly result
- recommended order
- carton quantity

The anomaly score is NOT a probability.

Keep your private reasoning internal.

Give only concise business reasoning.

Use simple professional management English.


OUTPUT

Use these sections:

1. Merchandise Overview

2. Immediate Attention

3. Stockout Risk

4. Overstock / Inventory Health

5. Anomaly Detection

6. Recommended Actions

7. Management Summary


MERCHANDISE RESULTS

{merchandise_data}
"""

    # =====================================================
    # GEMINI API
    # =====================================================

    last_error = None

    for model_name in GEMINI_MODELS:

        try:

            print()
            print(
                f"Trying Gemini model: "
                f"{model_name}"
            )

            chat = (
                client.chats.create(
                    model=model_name
                )
            )

            response = (
                chat.send_message(
                    prompt
                )
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            if response_text:

                print(
                    "Gemini response received."
                )

                return response_text

        except Exception as error:

            last_error = error

            print(
                f"Gemini model "
                f"{model_name} failed:"
            )

            print(
                error
            )

            time.sleep(2)

    raise RuntimeError(
        f"Gemini failed: "
        f"{last_error}"
    )