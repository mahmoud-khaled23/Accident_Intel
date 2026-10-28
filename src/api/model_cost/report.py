"""CLI report — prints per-model token usage and USD cost.

Run from the repo root:
    uv run python -m src.api.model_cost.report
"""
from .analysis import get_cost_analysis


def main() -> None:
    """Print an aligned per-model cost table plus grand total and note."""
    data = get_cost_analysis()
    models = data["models"]
    grand = data["grand_total"]
    note = data["note"]

    col_widths = {
        "model": max(len("Model"), *(len(r["model_id"]) for r in models) if models else [5]),
        "source": max(len("Source"), *(len(r["source"]) for r in models) if models else [6]),
    }

    header = (
        f"{'Model':<{col_widths['model']}}  "
        f"{'Source':<{col_widths['source']}}  "
        f"{'Calls':>7}  "
        f"{'Prompt tok':>12}  "
        f"{'Completion tok':>14}  "
        f"{'Total tok':>10}  "
        f"{'Cost USD':>10}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    if not models:
        print("  (no model calls recorded yet)")
    else:
        for row in models:
            print(
                f"{row['model_id']:<{col_widths['model']}}  "
                f"{row['source']:<{col_widths['source']}}  "
                f"{row['calls']:>7}  "
                f"{row['prompt_tokens']:>12,}  "
                f"{row['completion_tokens']:>14,}  "
                f"{row['total_tokens']:>10,}  "
                f"{row['cost_usd']:>10.6f}"
            )

    print(separator)
    source_pad = col_widths["source"]
    print(
        f"{'TOTAL':<{col_widths['model']}}  "
        f"{'': <{source_pad}}  "
        f"{grand['calls']:>7}  "
        f"{grand['prompt_tokens']:>12,}  "
        f"{grand['completion_tokens']:>14,}  "
        f"{grand['total_tokens']:>10,}  "
        f"{grand['cost_usd']:>10.6f}"
    )
    print(separator)
    print()
    print(f"Note: {note}")


if __name__ == "__main__":
    main()
