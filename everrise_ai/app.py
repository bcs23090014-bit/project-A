from pathlib import Path
from math import ceil

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
)

from forecast import (
    explain_all_with_gemini,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    BASE_DIR
    / "merchandise_ai_result.csv"
)

PAGE_SIZE = 50

app = Flask(
    __name__
)


# =========================================================
# CACHE
# =========================================================

_result_cache = None
_result_modified_time = None


# =========================================================
# LOAD SCORED RESULT
# =========================================================

def load_results():

    global _result_cache
    global _result_modified_time

    if not RESULT_FILE.exists():

        raise FileNotFoundError(
            "merchandise_ai_result.csv "
            "was not found. "
            "Run python score_all.py first."
        )

    modified_time = (
        RESULT_FILE
        .stat()
        .st_mtime
    )

    if (
        _result_cache is None
        or
        modified_time
        != _result_modified_time
    ):

        print()
        print(
            "Loading merchandise "
            "AI result..."
        )

        _result_cache = pd.read_csv(
            RESULT_FILE
        )

        _result_modified_time = (
            modified_time
        )

        print(
            f"Loaded "
            f"{len(_result_cache):,} "
            f"result rows."
        )

    return _result_cache


# =========================================================
# STORES
# =========================================================

def get_stores(
    df,
):

    stores = (
        df[
            [
                "store_id",
                "store_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "store_name"
        )
    )

    return stores.to_dict(
        orient="records"
    )


# =========================================================
# STORE DATA
# =========================================================

def get_store_results(
    df,
    store,
):

    store_text = (
        str(store)
        .strip()
        .lower()
    )

    return df[
        (
            df[
                "store_name"
            ]
            .astype(str)
            .str.lower()
            == store_text
        )
        |
        (
            df[
                "store_id"
            ]
            .astype(str)
            .str.lower()
            == store_text
        )
    ].copy()


# =========================================================
# KPI
# =========================================================

def calculate_kpis(
    df,
):

    if df.empty:

        return {
            "total_products": 0,
            "total_stock": 0,
            "total_forecast": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "reorder_count": 0,
            "anomaly_count": 0,
            "overstock_count": 0,
            "healthy_count": 0,
            "below_minimum_count": 0,
            "total_order": 0,
        }

    return {

        "total_products":
            int(
                df[
                    "sku"
                ].nunique()
            ),

        "total_stock":
            int(
                df[
                    "current_stock"
                ].sum()
            ),

        "total_forecast":
            int(
                df[
                    "total_forecast"
                ].sum()
            ),

        "high_risk":
            int(
                (
                    df[
                        "stockout_risk"
                    ]
                    == "HIGH"
                ).sum()
            ),

        "medium_risk":
            int(
                (
                    df[
                        "stockout_risk"
                    ]
                    == "MEDIUM"
                ).sum()
            ),

        "low_risk":
            int(
                (
                    df[
                        "stockout_risk"
                    ]
                    == "LOW"
                ).sum()
            ),

        "reorder_count":
            int(
                (
                    df[
                        "recommended_order"
                    ]
                    > 0
                ).sum()
            ),

        "anomaly_count":
            int(
                (
                    df[
                        "anomaly_status"
                    ]
                    == "ANOMALY"
                ).sum()
            ),

        "overstock_count":
            int(
                (
                    df[
                        "inventory_status"
                    ]
                    == "OVERSTOCK"
                ).sum()
            ),

        "healthy_count":
            int(
                (
                    df[
                        "inventory_status"
                    ]
                    == "HEALTHY"
                ).sum()
            ),

        "below_minimum_count":
            int(
                (
                    df[
                        "inventory_status"
                    ]
                    == "BELOW MINIMUM STOCK"
                ).sum()
            ),

        "total_order":
            int(
                df[
                    "recommended_order"
                ].sum()
            ),
    }


# =========================================================
# INVENTORY GROUP
# =========================================================

def add_inventory_group(
    df,
):

    df = df.copy()

    conditions = [

        df[
            "inventory_status"
        ].isin(
            [
                "BELOW MINIMUM STOCK",
                "REORDER REQUIRED",
            ]
        ),

        df[
            "inventory_status"
        ].eq(
            "OVERSTOCK"
        ),

        df[
            "inventory_status"
        ].eq(
            "HEALTHY"
        ),
    ]

    choices = [
        "UNDERSTOCK",
        "OVERSTOCK",
        "HEALTHY",
    ]

    df[
        "inventory_group"
    ] = pd.Series(
        data=pd.NA,
        index=df.index,
        dtype="object",
    )

    import numpy as np

    df[
        "inventory_group"
    ] = np.select(
        conditions,
        choices,
        default="OTHER",
    )

    return df


# =========================================================
# PRIORITY SORT
# =========================================================

def sort_priority(
    df,
):

    df = add_inventory_group(
        df
    )

    inventory_priority = {
        "UNDERSTOCK": 3,
        "OVERSTOCK": 2,
        "HEALTHY": 1,
        "OTHER": 0,
    }

    risk_priority = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    df[
        "_inventory_priority"
    ] = (
        df[
            "inventory_group"
        ]
        .map(
            inventory_priority
        )
        .fillna(0)
    )

    df[
        "_risk_priority"
    ] = (
        df[
            "stockout_risk"
        ]
        .map(
            risk_priority
        )
        .fillna(0)
    )

    df[
        "_anomaly_priority"
    ] = (
        df[
            "anomaly_status"
        ]
        .eq(
            "ANOMALY"
        )
        .astype(int)
    )

    df = df.sort_values(
        by=[
            "_inventory_priority",
            "_risk_priority",
            "_anomaly_priority",
            "recommended_order",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return df.drop(
        columns=[
            "_inventory_priority",
            "_risk_priority",
            "_anomaly_priority",
        ]
    )


# =========================================================
# FILTER
# =========================================================

def apply_status_filter(
    df,
    status_filter,
):

    if status_filter in [
        "UNDERSTOCK",
        "OVERSTOCK",
        "HEALTHY",
    ]:

        return df[
            df[
                "inventory_group"
            ]
            == status_filter
        ]

    return df


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def index():

    try:

        all_data = (
            load_results()
        )

    except Exception as error:

        return f"""
        <h2>Dashboard cannot load</h2>
        <p>{error}</p>
        <p>
            Run:
            <strong>
                python score_all.py
            </strong>
        </p>
        """

    stores = get_stores(
        all_data
    )

    if not stores:

        return "No stores found."

    selected_store = (
        request.args.get(
            "store"
        )
        or
        stores[0][
            "store_name"
        ]
    )

    # =====================================================
    # STORE RESULTS
    # =====================================================

    store_data = (
        get_store_results(
            all_data,
            selected_store,
        )
    )

    # KPI is always based on entire store
    kpis = calculate_kpis(
        store_data
    )

    # =====================================================
    # SORT
    # =====================================================

    filtered = sort_priority(
        store_data
    )

    # =====================================================
    # STATUS FILTER
    # =====================================================

    status_filter = (
        request.args.get(
            "status",
            "ALL",
        )
        .strip()
        .upper()
    )

    filtered = (
        apply_status_filter(
            filtered,
            status_filter,
        )
    )

    # =====================================================
    # SEARCH
    # =====================================================

    search_query = (
        request.args.get(
            "q",
            "",
        )
        .strip()
    )

    if search_query:

        q = (
            search_query
            .lower()
        )

        filtered = filtered[
            (
                filtered[
                    "sku"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    q,
                    regex=False,
                )
            )
            |
            (
                filtered[
                    "product_name"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    q,
                    regex=False,
                )
            )
        ]

    # =====================================================
    # PAGINATION
    # =====================================================

    page = request.args.get(
        "page",
        1,
        type=int,
    )

    total_filtered = len(
        filtered
    )

    total_pages = max(
        1,
        ceil(
            total_filtered
            / PAGE_SIZE
        ),
    )

    page = max(
        1,
        min(
            page,
            total_pages,
        ),
    )

    start = (
        page - 1
    ) * PAGE_SIZE

    end = (
        start
        + PAGE_SIZE
    )

    page_data = (
        filtered
        .iloc[
            start:end
        ]
    )

    results = (
        page_data
        .to_dict(
            orient="records"
        )
    )

    return render_template(
        "index.html",

        results=results,
        kpis=kpis,
        stores=stores,

        selected_store=
            selected_store,

        ai_summary=None,

        search_query=
            search_query,

        status_filter=
            status_filter,

        page=page,

        total_pages=
            total_pages,

        total_filtered=
            total_filtered,

        page_size=
            PAGE_SIZE,
    )


# =========================================================
# GEMINI MANAGEMENT SUMMARY
# =========================================================

@app.route(
    "/ai-summary",
    methods=["POST"],
)
def ai_summary():

    all_data = (
        load_results()
    )

    stores = get_stores(
        all_data
    )

    selected_store = (
        request.form.get(
            "store"
        )
        or
        stores[0][
            "store_name"
        ]
    )

    store_data = (
        get_store_results(
            all_data,
            selected_store,
        )
    )

    store_data = sort_priority(
        store_data
    )

    kpis = calculate_kpis(
        store_data
    )

    # =====================================================
    # IMPORTANT:
    # Do NOT send all 1,500 products to Gemini.
    #
    # Only top 30 priority products.
    # =====================================================

    ai_products = (
        store_data
        .head(30)
        .to_dict(
            orient="records"
        )
    )

    try:

        summary = (
            explain_all_with_gemini(
                ai_products
            )
        )

    except Exception as error:

        summary = (
            "Gemini AI summary failed: "
            + str(error)
        )

    first_page = (
        store_data
        .head(
            PAGE_SIZE
        )
        .to_dict(
            orient="records"
        )
    )

    total_pages = max(
        1,
        ceil(
            len(store_data)
            / PAGE_SIZE
        ),
    )

    return render_template(
        "index.html",

        results=first_page,
        kpis=kpis,
        stores=stores,

        selected_store=
            selected_store,

        ai_summary=
            summary,

        search_query="",

        status_filter="ALL",

        page=1,

        total_pages=
            total_pages,

        total_filtered=
            len(store_data),

        page_size=
            PAGE_SIZE,
    )


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )