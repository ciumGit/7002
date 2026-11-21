# ShopPulse Ops Agent (WQF7002 Assignment 1)

An intelligent e-commerce operations and inventory decision-support agent built with `Qwen/Qwen2.5-7B-Instruct` (Hugging Face), Python tool-calling with allowlist validation & safety guardrails, and a Gradio interface.

## Files
- `app.py`: Gradio web application displaying conversation responses, tool call traces (name, arguments, result), and generated charts.
- `agent_core.py`: Hugging Face `InferenceClient` tool-calling controller (`tools=[...]`, `tool_choice="auto"`) with offline JSON fallback.
- `agent_tools.py`: Guardrails, schema validation, and three domain tools (`query_sales_metrics`, `check_inventory_status`, `generate_analytics_chart`).
- `ecommerce_daily_transactions.csv`: 60-day daily e-commerce transaction and inventory dataset (`2026-09-01` to `2026-10-30`, 720 rows).
- `requirements.txt`: Python dependencies.

## Quickstart
```bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"  # Optional: falls back to local JSON planner if unset
python app.py
```
