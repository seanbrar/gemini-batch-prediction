"""
Visualization utilities for batch processing analysis
"""  # noqa: D200, D212, D415

from typing import Any, Dict, List, Tuple  # noqa: UP035

import matplotlib.pyplot as plt
import pandas as pd

from ..constants import (  # noqa: TID252
    TARGET_EFFICIENCY_RATIO,
    VIZ_ALPHA,
    VIZ_BAR_WIDTH,
    VIZ_COLORS,
    VIZ_FIGURE_SIZE,
    VIZ_SCALING_FIGURE_SIZE,
)
from ..response import calculate_quality_score  # noqa: TID252

# Constants for consistent styling
COLORS = VIZ_COLORS

PLOT_CONFIG = {
    "alpha": VIZ_ALPHA,
    "bar_width": VIZ_BAR_WIDTH,
    "figure_size": VIZ_FIGURE_SIZE,
    "scaling_figure_size": VIZ_SCALING_FIGURE_SIZE,
    "target_efficiency": TARGET_EFFICIENCY_RATIO,
}


def _format_integer(x):  # noqa: ANN001, ANN202
    """Format value as integer"""  # noqa: D415
    return f"{int(x)}"


def _format_integer_with_commas(x):  # noqa: ANN001, ANN202
    """Format value as integer with comma separators"""  # noqa: D415
    return f"{int(x):,}"


def _format_time_seconds(x):  # noqa: ANN001, ANN202
    """Format value as time in seconds with 2 decimal places"""  # noqa: D415
    return f"{x:.2f}s"


def _format_efficiency_multiplier(x):  # noqa: ANN001, ANN202
    """Format value as efficiency multiplier with 1 decimal place"""  # noqa: D415
    return f"{x:.1f}×"  # noqa: RUF001


def _add_bar_annotations(ax, bars, values: List, format_func=None):  # noqa: ANN001, ANN202, UP006
    """Add value annotations to bar charts"""  # noqa: D415
    if format_func is None:
        format_func = _format_integer

    max_val = max(values) if values else 1

    for bar, value in zip(bars, values):  # noqa: B905
        height = bar.get_height()
        offset = max_val * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            format_func(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def _create_comparison_subplot(  # noqa: PLR0913
    ax, title: str, ylabel: str, methods: List[str], values: List, format_func=None  # noqa: ANN001, COM812, UP006
) -> None:
    """Create a standardized comparison bar chart"""  # noqa: D415
    colors = [COLORS["individual"], COLORS["batch"]]
    bars = ax.bar(methods, values, color=colors, alpha=PLOT_CONFIG["alpha"])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)

    _add_bar_annotations(ax, bars, values, format_func)


def _validate_results_data(results: Dict[str, Any]) -> bool:  # noqa: UP006
    """Validate that results contain required data structure"""  # noqa: D415
    required_keys = ["metrics", "efficiency"]
    if not all(key in results for key in required_keys):
        return False

    metrics = results["metrics"]
    return all(key in metrics for key in ["individual", "batch"])


def create_efficiency_visualizations(results: Dict[str, Any]) -> None:  # noqa: UP006
    """Create comprehensive visualizations of efficiency gains"""  # noqa: D202, D415

    if not _validate_results_data(results):
        print("❌ Invalid results data structure")  # noqa: T201
        return

    individual_metrics = results["metrics"]["individual"]
    batch_metrics = results["metrics"]["batch"]
    efficiency = results["efficiency"]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=PLOT_CONFIG["figure_size"]  # noqa: COM812
    )
    fig.suptitle(
        "Gemini Batch Processing Efficiency Analysis", fontsize=16, fontweight="bold"  # noqa: COM812
    )

    methods = ["Individual", "Batch"]

    # 1. API Calls Comparison
    calls = [individual_metrics["calls"], batch_metrics["calls"]]
    _create_comparison_subplot(
        ax1, "API Calls Required", "Number of Calls", methods, calls  # noqa: COM812
    )

    # 2. Token Usage Comparison
    tokens = [individual_metrics["tokens"], batch_metrics["tokens"]]
    _create_comparison_subplot(
        ax2,
        "Total Token Usage",
        "Tokens",
        methods,
        tokens,
        format_func=_format_integer_with_commas,
    )

    # 3. Processing Time Comparison
    times = [individual_metrics["time"], batch_metrics["time"]]
    _create_comparison_subplot(
        ax3,
        "Processing Time",
        "Time (seconds)",
        methods,
        times,
        format_func=_format_time_seconds,
    )

    # 4. Efficiency Improvements
    improvements = [
        efficiency["token_efficiency_ratio"],
        efficiency["time_efficiency"],
        individual_metrics["calls"] / batch_metrics["calls"],
    ]
    improvement_labels = [
        "Token\nEfficiency",
        "Time\nEfficiency",
        "API Call\nReduction",
    ]

    bars4 = ax4.bar(
        improvement_labels,
        improvements,
        color=COLORS["improvements"],
        alpha=PLOT_CONFIG["alpha"],
    )
    ax4.set_title("Efficiency Improvements (×)", fontweight="bold")  # noqa: RUF001
    ax4.set_ylabel("Improvement Factor")
    ax4.axhline(
        y=PLOT_CONFIG["target_efficiency"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Target (3x)",
    )
    ax4.legend()

    _add_bar_annotations(ax4, bars4, improvements, _format_efficiency_multiplier)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    _print_efficiency_summary(individual_metrics, batch_metrics, efficiency)


def _print_efficiency_summary(
    individual_metrics: Dict, batch_metrics: Dict, efficiency: Dict  # noqa: COM812, UP006
) -> None:
    """Print formatted efficiency summary"""  # noqa: D415
    print("\n📊 EFFICIENCY SUMMARY:")  # noqa: T201
    print(  # noqa: T201
        f"🎯 Token efficiency improvement: "
        f"{efficiency['token_efficiency_ratio']:.1f}× "  # noqa: RUF001
        f"(Target: {PLOT_CONFIG['target_efficiency']}×+)"  # noqa: COM812, RUF001
    )
    print(f"⚡ Time efficiency improvement: {efficiency['time_efficiency']:.1f}×")  # noqa: RUF001, T201

    cost_reduction = (1 - 1 / efficiency["token_efficiency_ratio"]) * 100
    print(f"💰 Estimated cost reduction: {cost_reduction:.1f}%")  # noqa: T201

    tokens_saved = individual_metrics["tokens"] - batch_metrics["tokens"]
    if tokens_saved > 0:
        reduction_pct = tokens_saved / individual_metrics["tokens"] * 100
        print(f"🔢 Tokens saved: {tokens_saved:,} ({reduction_pct:.1f}% reduction)")  # noqa: T201


def visualize_scaling_results(scaling_data: List[Dict]) -> None:  # noqa: UP006
    """Visualize how efficiency scales with question count"""  # noqa: D202, D415

    if not scaling_data:
        print("❌ No scaling data available for visualization")  # noqa: T201
        return

    # Validate data structure
    required_fields = ["questions", "efficiency", "individual_tokens", "batch_tokens"]
    if not all(field in scaling_data[0] for field in required_fields):
        print("❌ Invalid scaling data structure")  # noqa: T201
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(scaling_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_CONFIG["scaling_figure_size"])
    fig.suptitle("Batch Processing Efficiency Scaling", fontsize=16, fontweight="bold")

    # 1. Efficiency vs Question Count
    ax1.plot(
        df["questions"],
        df["efficiency"],
        "o-",
        linewidth=3,
        markersize=8,
        color=COLORS["line"],
    )
    ax1.axhline(
        y=PLOT_CONFIG["target_efficiency"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Target (3×)",  # noqa: RUF001
    )
    ax1.set_xlabel("Number of Questions")
    ax1.set_ylabel("Efficiency Improvement (×)")  # noqa: RUF001
    ax1.set_title("Efficiency vs Question Count")
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    ax1.legend()

    # Add annotations for each point
    for _, row in df.iterrows():
        ax1.annotate(
            f"{row['efficiency']:.1f}×",  # noqa: RUF001
            (row["questions"], row["efficiency"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    # 2. Token Usage Comparison
    x_pos = range(len(df))
    width = PLOT_CONFIG["bar_width"]

    ax2.bar(
        [x - width / 2 for x in x_pos],
        df["individual_tokens"],
        width,
        label="Individual",
        color=COLORS["individual"],
        alpha=PLOT_CONFIG["alpha"],
    )
    ax2.bar(
        [x + width / 2 for x in x_pos],
        df["batch_tokens"],
        width,
        label="Batch",
        color=COLORS["batch"],
        alpha=PLOT_CONFIG["alpha"],
    )

    ax2.set_xlabel("Question Count")
    ax2.set_ylabel("Total Tokens")
    ax2.set_title("Token Usage: Individual vs Batch")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df["questions"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)  # noqa: FBT003

    plt.tight_layout()
    plt.show()

    # Summary insights
    _print_scaling_insights(df)


def _print_scaling_insights(df: pd.DataFrame) -> None:
    """Print formatted scaling insights"""  # noqa: D415
    print("\n🎯 SCALING INSIGHTS:")  # noqa: T201
    max_efficiency = df["efficiency"].max()
    best_q_count = df.loc[df["efficiency"].idxmax(), "questions"]

    print(  # noqa: T201
        f"📈 Maximum efficiency: {max_efficiency:.1f}× (with {best_q_count} questions)"  # noqa: COM812, RUF001
    )

    if "meets_target" in df.columns:
        target_count = sum(df["meets_target"])
        print(f"🎯 Questions meeting 3× target: {target_count}/{len(df)}")  # noqa: RUF001, T201

    if len(df) >= 2:  # noqa: PLR2004
        efficiency_trend = df["efficiency"].iloc[-1] - df["efficiency"].iloc[0]
        trend_direction = "+" if efficiency_trend > 0 else ""
        print(  # noqa: T201
            f"📊 Efficiency trend: {trend_direction}{efficiency_trend:.1f}× "  # noqa: RUF001
            f"from {df['questions'].min()} to {df['questions'].max()} questions"  # noqa: COM812
        )


def _format_metrics_table(metrics_data: List[Tuple[str, Any, Any, str]]) -> None:  # noqa: UP006
    """Format and print metrics comparison table"""  # noqa: D415
    print("\n📊 EFFICIENCY RESULTS:")  # noqa: T201
    print(f"{'Metric':<25} {'Individual':<12} {'Batch':<12} {'Improvement':<12}")  # noqa: T201
    print("-" * 65)  # noqa: T201

    for metric_name, individual_val, batch_val, improvement in metrics_data:
        print(f"{metric_name:<25} {individual_val:<12} {batch_val:<12} {improvement}")  # noqa: T201


def run_efficiency_experiment(
    processor, content: str, questions: List[str], name: str = "Demo"  # noqa: ANN001, COM812, UP006
) -> Dict[str, Any]:  # noqa: UP006
    """Run comprehensive efficiency experiment with visualizations"""  # noqa: D202, D415

    print(f"🔬 Running Experiment: {name}")  # noqa: T201
    print("=" * 50)  # noqa: T201

    # Process with comparison enabled
    results = processor.process_text_questions(content, questions, compare_methods=True)

    if not _validate_results_data(results):
        print("❌ Experiment failed - invalid results")  # noqa: T201
        return {}

    # Extract metrics
    individual_metrics = results["metrics"]["individual"]
    batch_metrics = results["metrics"]["batch"]
    efficiency = results["efficiency"]

    # Prepare metrics for table display
    metrics_data = [
        (
            "API Calls",
            individual_metrics["calls"],
            batch_metrics["calls"],
            f"{individual_metrics['calls'] / batch_metrics['calls']:.1f}x",
        ),
        (
            "Total Tokens",
            individual_metrics["tokens"],
            f"{batch_metrics['tokens']:,}",
            f"{efficiency['token_efficiency_ratio']:.1f}x",
        ),
        (
            "Processing Time",
            f"{individual_metrics['time']:.2f}s",
            f"{batch_metrics['time']:.2f}s",
            f"{efficiency['time_efficiency']:.1f}x",
        ),
    ]

    _format_metrics_table(metrics_data)

    # Target and quality analysis
    target_met = "✅ YES" if efficiency["meets_target"] else "❌ NO"
    print(f"{'Meets 3x Target':<25} {target_met}")  # noqa: T201

    # Quality analysis
    individual_answers = results.get("individual_answers", [])
    batch_answers = results.get("answers", [])

    quality_score = calculate_quality_score(individual_answers, batch_answers)
    if quality_score is not None:
        print(f"{'Quality Score':<25} {quality_score * 100:.1f}%")  # noqa: T201

    return results


def create_focused_efficiency_visualization(
    results: Dict[str, Any], show_summary: bool = False  # noqa: COM812, FBT001, FBT002, UP006
) -> None:
    """Create focused 2-chart visualization for notebook demonstrations"""  # noqa: D202, D415

    if not _validate_results_data(results):
        print("❌ Invalid results data structure")  # noqa: T201
        return

    individual_metrics = results["metrics"]["individual"]
    batch_metrics = results["metrics"]["batch"]
    efficiency = results["efficiency"]

    # Create side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Batch Processing Efficiency Analysis", fontsize=16, fontweight="bold")

    methods = ["Individual", "Batch"]

    # 1. Token Usage Comparison (main value proposition)
    tokens = [individual_metrics["tokens"], batch_metrics["tokens"]]
    _create_comparison_subplot(
        ax1,
        "Total Token Usage",
        "Tokens",
        methods,
        tokens,
        format_func=_format_integer_with_commas,
    )

    # Add more headroom by extending y-axis limits
    max_tokens = max(tokens)
    ax1.set_ylim(0, max_tokens * 1.3)  # 30% more headroom

    # 2. Processing Time Comparison (workflow benefit)
    times = [individual_metrics["time"], batch_metrics["time"]]
    _create_comparison_subplot(
        ax2,
        "Processing Time",
        "Time (seconds)",
        methods,
        times,
        format_func=_format_time_seconds,
    )

    # Add more headroom by extending y-axis limits
    max_times = max(times)
    ax2.set_ylim(0, max_times * 1.3)  # 30% more headroom

    # Add efficiency annotations to make impact clear
    token_efficiency = efficiency["token_efficiency_ratio"]
    time_efficiency = efficiency["time_efficiency"]

    # Add efficiency callout on token chart (moved higher up)
    ax1.text(
        0.5,
        0.9,
        f"{token_efficiency:.1f}× more efficient",  # noqa: RUF001
        ha="center",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.7},
    )

    # Add time savings callout (moved higher up)
    ax2.text(
        0.5,
        0.9,
        f"{time_efficiency:.1f}× faster",  # noqa: RUF001
        ha="center",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.show()

    # Optional summary (concise version)
    if show_summary:
        tokens_saved = individual_metrics["tokens"] - batch_metrics["tokens"]
        cost_reduction = (tokens_saved / individual_metrics["tokens"]) * 100
        print(f"\n💰 Cost reduction: {cost_reduction:.1f}%")  # noqa: T201
        print(f"⚡ Time savings: {time_efficiency:.1f}× faster")  # noqa: RUF001, T201
        print(f"🎯 Target met: {'Yes' if efficiency['meets_target'] else 'No'}")  # noqa: T201
