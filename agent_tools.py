import os
import re
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), "ecommerce_daily_transactions.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_FUNCTIONS = {
    "query_sales_metrics",
    "check_inventory_status",
    "generate_analytics_chart",
}

VALID_CATEGORIES = {"Electronics", "Apparel", "Home & Kitchen", "Beauty"}
VALID_REGIONS = {"Klang Valley", "Penang", "Johor"}
VALID_METRICS = {"gmv_myr", "net_revenue_myr", "orders_count", "units_sold", "refund_amount_myr"}
VALID_CHART_TYPES = {"sales_trend", "category_revenue", "inventory_health"}
VALID_GROUP_BY = {"date", "category", "region"}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|your)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
    r"print\s+(your\s+)?(system\s+prompt|instructions)",
    r"jailbreak",
    r"bypass\s+(security|guardrail|allowlist|validation)",
    r"act\s+as\s+unrestricted",
]

UNSAFE_ACTION_PATTERNS = [
    r"\b(delete|drop|truncate|wipe|erase)\b.*\b(record|row|table|database|file|data)\b",
    r"\b(update|modify|overwrite|change|set)\b.*\b(stock|inventory|price|revenue|record)\b",
    r"\b(issue|process|send)\b.*\b(refund|payout|payment)\b",
    r"\b(credit\s*card|password|customer\s*email|phone\s*number|ic\s*number|pii)\b",
    r"delete_all_records",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["date"] = df["date"].astype(str)
    return df


def check_guardrails(user_prompt: str) -> tuple[bool, str, str]:
    """
    Code-level safety guardrail executed before LLM routing.
    Returns (is_safe, block_code, message).
    """
    text = user_prompt.strip().lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text):
            return (
                False,
                "PROMPT_INJECTION_BLOCKED",
                "Request blocked by safety guardrail: prompt injection or instruction-override attempt detected. "
                "I can only answer verified read-only sales and inventory analytics queries.",
            )
    for pat in UNSAFE_ACTION_PATTERNS:
        if re.search(pat, text):
            return (
                False,
                "UNSAFE_ACTION_BLOCKED",
                "Action refused by safety guardrail: modifying inventory, issuing financial refunds, deleting records, "
                "or accessing customer PII is restricted. Please route mutating operations to an authorized human operator in the ERP system.",
            )
    return True, "PASSED", "Input passed safety and scope guardrails."


def validate_tool_call(func_name: str, arguments: dict) -> tuple[bool, str]:
    """
    Strict allowlist and parameter schema validation prior to executing any Python tool.
    """
    if func_name not in ALLOWED_FUNCTIONS:
        return False, f"Validation Error: Function '{func_name}' is not on the approved allowlist."

    if not isinstance(arguments, dict):
        return False, "Validation Error: Tool arguments must be a JSON dictionary."

    for key in ["start_date", "end_date"]:
        val = arguments.get(key)
        if val is not None:
            if not isinstance(val, str):
                return False, f"Validation Error: '{key}' must be a string in YYYY-MM-DD format."
            try:
                datetime.strptime(val, "%Y-%m-%d")
            except ValueError:
                return False, f"Validation Error: '{key}' ('{val}') is not a valid YYYY-MM-DD date."

    if arguments.get("start_date") and arguments.get("end_date"):
        if arguments["start_date"] > arguments["end_date"]:
            return False, "Validation Error: 'start_date' cannot be later than 'end_date'."

    cat = arguments.get("category")
    if cat is not None and cat not in VALID_CATEGORIES:
        return False, f"Validation Error: Unknown category '{cat}'. Allowed: {sorted(VALID_CATEGORIES)}."

    reg = arguments.get("region")
    if reg is not None and reg not in VALID_REGIONS:
        return False, f"Validation Error: Unknown region '{reg}'. Allowed: {sorted(VALID_REGIONS)}."

    if func_name == "query_sales_metrics":
        metric = arguments.get("metric", "net_revenue_myr")
        if metric not in VALID_METRICS:
            return False, f"Validation Error: Unsupported metric '{metric}'."

    if func_name == "check_inventory_status":
        low_only = arguments.get("only_low_stock", False)
        if not isinstance(low_only, bool):
            return False, "Validation Error: 'only_low_stock' must be a boolean."

    if func_name == "generate_analytics_chart":
        ctype = arguments.get("chart_type")
        if ctype not in VALID_CHART_TYPES:
            return False, f"Validation Error: 'chart_type' must be one of {sorted(VALID_CHART_TYPES)}."
        gby = arguments.get("group_by", "category")
        if gby not in VALID_GROUP_BY:
            return False, f"Validation Error: 'group_by' must be one of {sorted(VALID_GROUP_BY)}."

    return True, "Arguments validated successfully."


def query_sales_metrics(
    start_date: str = None,
    end_date: str = None,
    category: str = None,
    region: str = None,
    metric: str = "net_revenue_myr",
    check_boundary_coverage: bool = True,
) -> dict:
    """
    Aggregates e-commerce sales KPIs from ecommerce_daily_transactions.csv under specified filters.
    """
    df = load_dataset()
    data_min_date = df["date"].min()
    data_max_date = df["date"].max()

    coverage_warnings = []
    if check_boundary_coverage:
        if start_date and start_date < data_min_date:
            coverage_warnings.append(
                f"Requested start_date ({start_date}) precedes earliest available record ({data_min_date})."
            )
        if end_date and end_date > data_max_date:
            coverage_warnings.append(
                f"Requested end_date ({end_date}) extends beyond latest available record ({data_max_date})."
            )

    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["date"] <= end_date]
    if category:
        filtered = filtered[filtered["category"] == category]
    if region:
        filtered = filtered[filtered["region"] == region]

    if filtered.empty:
        return {
            "status": "NO_DATA",
            "message": f"No transaction records found for the requested filters. Available dataset window is {data_min_date} to {data_max_date}.",
            "records_matched": 0,
        }

    total_gmv = round(float(filtered["gmv_myr"].sum()), 2)
    total_net = round(float(filtered["net_revenue_myr"].sum()), 2)
    total_refunds = round(float(filtered["refund_amount_myr"].sum()), 2)
    total_orders = int(filtered["orders_count"].sum())
    total_units = int(filtered["units_sold"].sum())
    refund_rate_pct = round((total_refunds / total_gmv * 100.0) if total_gmv > 0 else 0.0, 2)

    breakdown_by_region = {}
    if region is None:
        reg_grp = filtered.groupby("region")[["gmv_myr", "net_revenue_myr", "orders_count", "refund_amount_myr"]].sum()
        for r_name, r_row in reg_grp.iterrows():
            breakdown_by_region[r_name] = {
                "gmv_myr": round(float(r_row["gmv_myr"]), 2),
                "net_revenue_myr": round(float(r_row["net_revenue_myr"]), 2),
                "orders_count": int(r_row["orders_count"]),
                "refund_amount_myr": round(float(r_row["refund_amount_myr"]), 2),
            }

    result = {
        "status": "SUCCESS",
        "observed_date_range": f"{filtered['date'].min()} to {filtered['date'].max()}",
        "filters": {"category": category or "ALL", "region": region or "ALL"},
        "records_matched": int(len(filtered)),
        "summary": {
            "gmv_myr": total_gmv,
            "net_revenue_myr": total_net,
            "orders_count": total_orders,
            "units_sold": total_units,
            "refund_amount_myr": total_refunds,
            "refund_rate_pct": refund_rate_pct,
        },
        "region_breakdown": breakdown_by_region,
    }
    if coverage_warnings:
        result["coverage_warning"] = " ".join(coverage_warnings)
    return result


def check_inventory_status(
    category: str = None,
    region: str = None,
    only_low_stock: bool = False,
) -> dict:
    """
    Checks the latest closing stock levels against reorder thresholds and estimates days of cover.
    """
    df = load_dataset()
    latest_date = df["date"].max()
    recent_dates = sorted(df["date"].unique())[-14:]
    vel_df = (
        df[df["date"].isin(recent_dates)]
        .groupby(["region", "category"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "avg_daily_units"})
    )

    snap = df[df["date"] == latest_date].merge(vel_df, on=["region", "category"], how="left")
    if category:
        snap = snap[snap["category"] == category]
    if region:
        snap = snap[snap["region"] == region]

    items = []
    for _, row in snap.iterrows():
        stock = int(row["closing_stock"])
        thresh = int(row["reorder_threshold"])
        vel = max(0.1, float(row["avg_daily_units"]))
        days_cover = round(stock / vel, 1)
        if stock == 0:
            health = "OUT_OF_STOCK"
        elif stock < thresh:
            health = "LOW_STOCK"
        else:
            health = "HEALTHY"

        if only_low_stock and health == "HEALTHY":
            continue

        items.append({
            "region": row["region"],
            "category": row["category"],
            "sku_id": row["sku_id"],
            "closing_stock": stock,
            "reorder_threshold": thresh,
            "est_days_cover": days_cover,
            "status": health,
        })

    return {
        "status": "SUCCESS",
        "snapshot_date": latest_date,
        "only_low_stock_filter": only_low_stock,
        "matched_items_count": len(items),
        "items": items,
    }


def generate_analytics_chart(
    chart_type: str,
    group_by: str = "category",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Generates a clean PNG analytics chart saved to outputs/ and returns chart metadata.
    """
    df = load_dataset()
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    if df.empty:
        return {"status": "NO_DATA", "message": "No records available for the selected chart window."}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)

    if chart_type == "sales_trend":
        daily = df.groupby("date")["net_revenue_myr"].sum().reset_index()
        ax.plot(daily["date"], daily["net_revenue_myr"], color="#1d4ed8", linewidth=2.0, marker="o", markersize=3.5)
        ax.fill_between(daily["date"], daily["net_revenue_myr"], color="#93c5fd", alpha=0.25)
        ax.set_title(f"Daily Net Revenue Trend ({daily['date'].min()} to {daily['date'].max()})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Net Revenue (MYR)", fontsize=9.5)
        ax.set_xlabel("Date", fontsize=9.5)
        step = max(1, len(daily) // 8)
        ax.set_xticks(range(0, len(daily), step))
        ax.set_xticklabels(daily["date"].iloc[::step], rotation=25, ha="right", fontsize=8)
        filename = f"chart_sales_trend_{daily['date'].min()}_{daily['date'].max()}.png"
        summary_note = f"Peak daily net revenue: MYR {daily['net_revenue_myr'].max():,.2f} on {daily.loc[daily['net_revenue_myr'].idxmax(), 'date']}."

    elif chart_type == "category_revenue":
        col = "region" if group_by == "region" else "category"
        grp = df.groupby(col)[["gmv_myr", "net_revenue_myr"]].sum().sort_values("net_revenue_myr", ascending=False)
        x = range(len(grp))
        ax.bar([i - 0.18 for i in x], grp["gmv_myr"], width=0.35, label="GMV (MYR)", color="#64748b")
        ax.bar([i + 0.18 for i in x], grp["net_revenue_myr"], width=0.35, label="Net Revenue (MYR)", color="#0f766e")
        ax.set_xticks(list(x))
        ax.set_xticklabels(grp.index, fontsize=9)
        ax.set_title(f"GMV vs. Net Revenue by {col.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amount (MYR)", fontsize=9.5)
        ax.legend(frameon=True, fontsize=8.5)
        filename = f"chart_revenue_by_{col}.png"
        summary_note = f"Top {col}: {grp.index[0]} (MYR {grp['net_revenue_myr'].iloc[0]:,.2f} net revenue)."

    elif chart_type == "inventory_health":
        latest_date = df["date"].max()
        snap = df[df["date"] == latest_date].groupby("category")[["closing_stock", "reorder_threshold"]].sum()
        x = range(len(snap))
        ax.bar([i - 0.18 for i in x], snap["closing_stock"], width=0.35, label="Closing Stock (Units)", color="#2563eb")
        ax.bar([i + 0.18 for i in x], snap["reorder_threshold"], width=0.35, label="Reorder Threshold (Units)", color="#dc2626")
        ax.set_xticks(list(x))
        ax.set_xticklabels(snap.index, fontsize=9)
        ax.set_title(f"Inventory Closing Stock vs. Reorder Threshold ({latest_date})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Units", fontsize=9.5)
        ax.legend(frameon=True, fontsize=8.5)
        filename = f"chart_inventory_health_{latest_date}.png"
        summary_note = f"Inventory snapshot rendered for {latest_date} across {len(snap)} product categories."

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path)
    plt.close(fig)

    return {
        "status": "SUCCESS",
        "chart_type": chart_type,
        "chart_path": out_path,
        "summary": summary_note,
    }
