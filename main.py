"""
Clinical Trial Enrollment Forecaster
======================================
Analyzes and forecasts clinical trial enrollment trends across cancer types
using Google's TimesFM 2.5 and ARIMA, with Gemini-powered plain-language summaries.

Usage:
    python main.py                          # Run full demo (4 cancer types)
    python main.py --cancer "breast cancer" # Single cancer type
    python main.py --cancer "pancreatic cancer" # Any cancer type you want
    python main.py --cancer "melanoma" # Any cancer type you want
    python main.py --cancer "ovarian cancer" # Any cancer type you want
    python main.py --cancer "glioblastoma" # Any cancer type you want
    python main.py --horizon 24             # 24-month forecast
    python main.py --cancer "glioblastoma" --horizon 24 # Any cancer type you want + 24-month forecast

Requires: GEMINI_API_KEY environment variable (free at https://aistudio.google.com/apikey)

Author: Mira Kapoor Wadehra
"""

import os
import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from trial_fetcher import TrialFetcher
from analyzer import TrialAnalyzer
from forecaster import EnrollmentForecaster
from insights import InsightGenerator


CANCER_TYPES = [
    "breast cancer",
    "non-small cell lung cancer",
    "colorectal cancer",
    "acute lymphoblastic leukemia",
]


def run_single(cancer_type: str, horizon: int = 12, max_trials: int = 1000,
               insight_gen: InsightGenerator = None):
    """Analyze, forecast, and summarize a single cancer type."""
    print(f"\n{'═' * 60}")
    print(f"  {cancer_type.upper()}")
    print(f"{'═' * 60}")

    fetcher = TrialFetcher()

    # Fetch
    print(f"\n  Fetching trials from ClinicalTrials.gov...")
    trials = fetcher.fetch_trials(condition=cancer_type, max_results=max_trials)
    print(f"  ✓ Fetched {len(trials)} trials")

    if len(trials) < 20:
        print(f"  ⚠ Not enough data for analysis")
        return None, None

    # Analyze
    analyzer = TrialAnalyzer(trials, cancer_type=cancer_type)
    summary = analyzer.landscape_summary()
    analyzer.print_summary()

    timeline = analyzer.build_monthly_timeline()
    chart = analyzer.plot_timeline(timeline)
    if chart:
        print(f"  ✓ Timeline chart: {chart}")

    # Forecast
    forecast_results = {}
    if len(timeline) >= 12:
        forecaster = EnrollmentForecaster(timeline, cancer_type=cancer_type,
                                           forecast_months=horizon)
        forecast_results = forecaster.run()
        chart = forecaster.plot()
        if chart:
            print(f"  ✓ Forecast chart: {chart}")

    # Gemini summary
    if insight_gen:
        print(f"\n  🤖 Generating summary...")
        insight = insight_gen.summarize_cancer_type(cancer_type, summary, forecast_results)
        print(f"\n  {'─' * 50}")
        print(f"  📝 SUMMARY: {cancer_type.upper()}")
        print(f"  {'─' * 50}")
        for line in insight.split("\n"):
            print(f"  {line}")
        print()

        # Save summary
        os.makedirs("outputs", exist_ok=True)
        slug = cancer_type.replace(" ", "_").lower()
        with open(f"outputs/{slug}_summary.txt", "w") as f:
            f.write(insight)

    # Save data
    os.makedirs("outputs", exist_ok=True)
    slug = cancer_type.replace(" ", "_").lower()
    if len(timeline) > 0:
        timeline.to_csv(f"outputs/{slug}_timeline.csv", index=False)
    if forecast_results:
        with open(f"outputs/{slug}_forecast.json", "w") as f:
            json.dump(forecast_results, f, indent=2)

    return summary, forecast_results


def run_demo(horizon: int = 12):
    """Run analysis across all cancer types with comparison summary."""
    print("\n" + "█" * 60)
    print("  CLINICAL TRIAL ENROLLMENT FORECASTER")
    print(f"  Analyzing {len(CANCER_TYPES)} cancer types")
    print(f"  Forecast horizon: {horizon} months")
    print("█" * 60)

    # Initialize Gemini
    try:
        insight_gen = InsightGenerator()
        print("  ✓ Gemini connected for summaries\n")
    except ValueError as e:
        print(f"  ⚠ {e}")
        print("  Running without AI summaries.\n")
        insight_gen = None

    all_summaries = {}
    all_forecasts = {}

    for cancer_type in CANCER_TYPES:
        summary, forecast = run_single(cancer_type, horizon=horizon,
                                        insight_gen=insight_gen)
        if summary:
            all_summaries[cancer_type] = summary
        if forecast:
            all_forecasts[cancer_type] = forecast

    # Cross-cancer comparison
    if insight_gen and len(all_summaries) > 1:
        print(f"\n{'█' * 60}")
        print("  CROSS-CANCER COMPARISON")
        print(f"{'█' * 60}")
        print(f"\n  🤖 Generating comparison...")

        comparison = insight_gen.summarize_comparison(all_summaries, all_forecasts)
        print(f"\n  {'─' * 50}")
        print(f"  📝 COMPARISON ACROSS ALL CANCER TYPES")
        print(f"  {'─' * 50}")
        for line in comparison.split("\n"):
            print(f"  {line}")
        print()

        with open("outputs/cross_cancer_comparison.txt", "w") as f:
            f.write(comparison)

    print(f"\n  ✅ Done. All outputs saved to ./outputs/")


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Enrollment Forecaster — TimesFM + ARIMA + Gemini"
    )
    parser.add_argument("--cancer", "-c", type=str, help="Cancer type to analyze (any cancer works)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast months (default: 12)")
    parser.add_argument("--max-trials", type=int, default=1000, help="Max trials to fetch (default: 1000)")

    args = parser.parse_args()

    if args.cancer:
        try:
            insight_gen = InsightGenerator()
        except ValueError:
            insight_gen = None
        run_single(args.cancer, horizon=args.horizon, max_trials=args.max_trials,
                   insight_gen=insight_gen)
    else:
        run_demo(horizon=args.horizon)


if __name__ == "__main__":
    main()