from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

METRICS_PATH = OUTPUT_DIR / "baseline_metrics.csv"
CALIBRATION_PATH = OUTPUT_DIR / "calibration_buckets.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "confidence_buckets.csv"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def model_label(name: str) -> str:
    labels = {
        "logistic_regression_numpy": "NumPy logistic regression",
        "majority_class": "Majority class",
        "higher_elo_rule": "Higher Elo",
        "higher_win_pct_rule": "Higher win pct",
        "higher_experience_rule": "Higher experience",
        "better_recent_form_rule": "Better recent form",
    }
    return labels.get(name, name.replace("_", " "))


def bar_chart(metrics: pd.DataFrame) -> str:
    metrics = metrics.sort_values("accuracy", ascending=True).copy()
    width = 920
    row_height = 48
    top = 82
    left = 230
    chart_width = 590
    height = top + row_height * len(metrics) + 58
    max_accuracy = max(0.70, float(metrics["accuracy"].max()))

    bars = []
    for index, row in enumerate(metrics.itertuples(index=False)):
        y = top + index * row_height
        bar_width = int((float(row.accuracy) / max_accuracy) * chart_width)
        color = "#2563eb" if row.model == "logistic_regression_numpy" else "#94a3b8"
        bars.append(
            f"""
            <text x="24" y="{y + 22}" class="label">{escape(model_label(row.model))}</text>
            <rect x="{left}" y="{y}" width="{bar_width}" height="26" rx="5" fill="{color}" />
            <text x="{left + bar_width + 10}" y="{y + 20}" class="value">{pct(float(row.accuracy))}</text>
            """
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: #475569; }}
    .label {{ font: 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: #1e293b; }}
    .value {{ font: 700 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="24" y="34" class="title">Model Comparison</text>
  <text x="24" y="58" class="subtitle">Accuracy on fights from 2024-01-01 onward</text>
  {''.join(bars)}
</svg>
"""


def line_points(values: list[float], *, left: int, top: int, width: int, height: int) -> list[tuple[int, int]]:
    if len(values) == 1:
        return [(left + width // 2, top + height - int(values[0] * height))]
    return [
        (
            left + int((index / (len(values) - 1)) * width),
            top + height - int(value * height),
        )
        for index, value in enumerate(values)
    ]


def polyline(points: list[tuple[int, int]], color: str) -> str:
    point_text = " ".join(f"{x},{y}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" />'
        for x, y in points
    )
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3" />\n{circles}'


def calibration_chart(calibration: pd.DataFrame) -> str:
    calibration = calibration.dropna(subset=["avg_predicted_probability", "actual_fighter_a_win_rate"])
    labels = calibration["probability_bucket"].tolist()
    predicted = calibration["avg_predicted_probability"].astype(float).tolist()
    actual = calibration["actual_fighter_a_win_rate"].astype(float).tolist()

    width = 920
    height = 470
    left = 72
    top = 86
    chart_width = 790
    chart_height = 285
    predicted_points = line_points(predicted, left=left, top=top, width=chart_width, height=chart_height)
    actual_points = line_points(actual, left=left, top=top, width=chart_width, height=chart_height)

    x_labels = []
    for index, label in enumerate(labels):
        x = left + int((index / (len(labels) - 1)) * chart_width)
        x_labels.append(f'<text x="{x}" y="{top + chart_height + 35}" class="tick" text-anchor="middle">{escape(label)}</text>')

    y_labels = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + chart_height - int(value * chart_height)
        y_labels.append(f'<line x1="{left}" x2="{left + chart_width}" y1="{y}" y2="{y}" class="grid" />')
        y_labels.append(f'<text x="{left - 14}" y="{y + 5}" class="tick" text-anchor="end">{pct(value)}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: #475569; }}
    .tick {{ font: 12px system-ui, -apple-system, Segoe UI, sans-serif; fill: #475569; }}
    .legend {{ font: 14px system-ui, -apple-system, Segoe UI, sans-serif; fill: #1e293b; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1.5; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="24" y="34" class="title">Calibration Buckets</text>
  <text x="24" y="58" class="subtitle">Predicted probability compared with actual Fighter A win rate</text>
  {''.join(y_labels)}
  <line x1="{left}" x2="{left}" y1="{top}" y2="{top + chart_height}" class="axis" />
  <line x1="{left}" x2="{left + chart_width}" y1="{top + chart_height}" y2="{top + chart_height}" class="axis" />
  {polyline(predicted_points, "#2563eb")}
  {polyline(actual_points, "#f97316")}
  {''.join(x_labels)}
  <rect x="630" y="28" width="14" height="14" fill="#2563eb" rx="3" />
  <text x="652" y="40" class="legend">Avg predicted</text>
  <rect x="760" y="28" width="14" height="14" fill="#f97316" rx="3" />
  <text x="782" y="40" class="legend">Actual win rate</text>
</svg>
"""


def confidence_chart(confidence: pd.DataFrame) -> str:
    width = 920
    height = 430
    left = 80
    top = 74
    chart_width = 770
    chart_height = 265
    rows = confidence.copy()
    bar_width = int(chart_width / len(rows) * 0.58)

    grid = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + chart_height - int(value * chart_height)
        grid.append(f'<line x1="{left}" x2="{left + chart_width}" y1="{y}" y2="{y}" class="grid" />')
        grid.append(f'<text x="{left - 14}" y="{y + 5}" class="tick" text-anchor="end">{pct(value)}</text>')

    bars = []
    for index, row in enumerate(rows.itertuples(index=False)):
        x = left + int((index + 0.5) * chart_width / len(rows)) - bar_width // 2
        accuracy = float(row.accuracy)
        bar_height = int(accuracy * chart_height)
        y = top + chart_height - bar_height
        bars.append(
            f"""
            <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="6" fill="#16a34a" />
            <text x="{x + bar_width / 2}" y="{y - 8}" class="value" text-anchor="middle">{pct(accuracy)}</text>
            <text x="{x + bar_width / 2}" y="{top + chart_height + 32}" class="tick" text-anchor="middle">{escape(row.confidence_bucket)}</text>
            """
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: #475569; }}
    .tick {{ font: 12px system-ui, -apple-system, Segoe UI, sans-serif; fill: #475569; }}
    .value {{ font: 700 14px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1.5; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="24" y="34" class="title">Accuracy by Confidence</text>
  <text x="24" y="58" class="subtitle">Higher confidence predictions should generally be more accurate</text>
  {''.join(grid)}
  <line x1="{left}" x2="{left}" y1="{top}" y2="{top + chart_height}" class="axis" />
  <line x1="{left}" x2="{left + chart_width}" y1="{top + chart_height}" y2="{top + chart_height}" class="axis" />
  {''.join(bars)}
</svg>
"""


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    calibration = pd.read_csv(CALIBRATION_PATH)
    confidence = pd.read_csv(CONFIDENCE_PATH)

    write_svg(CHART_DIR / "model_comparison.svg", bar_chart(metrics))
    write_svg(CHART_DIR / "calibration_buckets.svg", calibration_chart(calibration))
    write_svg(CHART_DIR / "confidence_buckets.svg", confidence_chart(confidence))

    print("Saved charts")
    print(f"  {CHART_DIR / 'model_comparison.svg'}")
    print(f"  {CHART_DIR / 'calibration_buckets.svg'}")
    print(f"  {CHART_DIR / 'confidence_buckets.svg'}")


if __name__ == "__main__":
    main()
