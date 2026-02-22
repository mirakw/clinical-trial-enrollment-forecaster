"""
Clinical Trial Landscape Analyzer
===================================
Builds monthly time series from historical trial data and computes
landscape statistics: phase distribution, completion rates, sponsor
mix, intervention trends.
"""

import os
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TrialAnalyzer:
    """Analyze historical clinical trial data for a cancer type."""

    def __init__(self, studies: list[dict], cancer_type: str = ""):
        self.studies = studies
        self.cancer_type = cancer_type
        self.df = pd.DataFrame(studies)
        self._parse_dates()

    def _parse_dates(self):
        """Parse date columns."""
        for col in ["start_date", "completion_date", "study_first_post_date"]:
            if col in self.df.columns:
                self.df[col + "_parsed"] = pd.to_datetime(
                    self.df[col], errors="coerce", format="mixed"
                )

        if "start_date_parsed" in self.df.columns:
            self.df["start_year"] = self.df["start_date_parsed"].dt.year

        if "start_date_parsed" in self.df.columns and "completion_date_parsed" in self.df.columns:
            self.df["duration_days"] = (
                self.df["completion_date_parsed"] - self.df["start_date_parsed"]
            ).dt.days

    def build_monthly_timeline(self) -> pd.DataFrame:
        """
        Build a monthly time series of new trial starts.
        This is the core input for forecasting.
        """
        if "start_date_parsed" not in self.df.columns:
            return pd.DataFrame()

        df = self.df.dropna(subset=["start_date_parsed"]).copy()
        if len(df) < 10:
            return pd.DataFrame()

        df["month"] = df["start_date_parsed"].dt.to_period("M")

        monthly = (
            df.groupby("month")
            .agg(
                new_trials=("nct_id", "count"),
                total_enrollment=("enrollment_count", "sum"),
            )
            .reset_index()
        )
        monthly["month"] = monthly["month"].dt.to_timestamp()
        monthly = monthly.sort_values("month").reset_index(drop=True)

        # Fill gaps
        if len(monthly) > 2:
            full_range = pd.date_range(
                start=monthly["month"].min(),
                end=monthly["month"].max(),
                freq="MS"
            )
            monthly = (
                monthly.set_index("month")
                .reindex(full_range, fill_value=0)
                .reset_index()
                .rename(columns={"index": "month"})
            )

        monthly["cumulative_trials"] = monthly["new_trials"].cumsum()
        monthly["rolling_6m"] = monthly["new_trials"].rolling(6, min_periods=1).mean()
        monthly["rolling_12m"] = monthly["new_trials"].rolling(12, min_periods=1).mean()

        return monthly

    def landscape_summary(self) -> dict:
        """Compute landscape statistics."""
        summary = {
            "cancer_type": self.cancer_type,
            "total_trials": len(self.df),
        }

        if "overall_status" in self.df.columns:
            summary["status"] = self.df["overall_status"].value_counts().to_dict()

        if "phase" in self.df.columns:
            summary["phases"] = self.df["phase"].value_counts().to_dict()

        if "enrollment_count" in self.df.columns:
            e = self.df["enrollment_count"].dropna()
            if len(e) > 0:
                summary["enrollment"] = {
                    "mean": round(e.mean(), 1),
                    "median": round(e.median(), 1),
                    "total": int(e.sum()),
                }

        if "duration_days" in self.df.columns:
            d = self.df["duration_days"].dropna()
            d = d[d > 0]
            if len(d) > 0:
                summary["duration_months"] = {
                    "mean": round(d.mean() / 30.44, 1),
                    "median": round(d.median() / 30.44, 1),
                }

        if "sponsor_class" in self.df.columns:
            summary["sponsor_class"] = self.df["sponsor_class"].value_counts().to_dict()

        if "lead_sponsor" in self.df.columns:
            summary["top_sponsors"] = (
                self.df["lead_sponsor"].value_counts().head(10).to_dict()
            )

        if "intervention_types" in self.df.columns:
            all_types = []
            for types in self.df["intervention_types"].dropna():
                if isinstance(types, list):
                    all_types.extend(types)
            summary["intervention_types"] = dict(Counter(all_types).most_common(10))

        # Completion rates by phase
        phase_stats = {}
        for phase in ["PHASE1", "PHASE2", "PHASE3"]:
            phase_df = self.df[self.df["phase"].str.contains(phase, na=False)]
            if len(phase_df) > 5:
                completed = len(phase_df[phase_df["overall_status"] == "COMPLETED"])
                terminated = len(phase_df[phase_df["overall_status"].isin(
                    ["TERMINATED", "WITHDRAWN"]
                )])
                phase_stats[phase] = {
                    "total": len(phase_df),
                    "completed": completed,
                    "terminated_or_withdrawn": terminated,
                    "completion_rate": round(completed / len(phase_df) * 100, 1),
                }
        summary["phase_completion"] = phase_stats

        return summary

    def print_summary(self) -> str:
        """Formatted text summary."""
        s = self.landscape_summary()
        lines = [
            f"\n  {self.cancer_type.upper()} TRIAL LANDSCAPE",
            f"  {'─' * 40}",
            f"  Total trials: {s['total_trials']}",
        ]

        if "status" in s:
            lines.append("\n  Status:")
            for status, n in sorted(s["status"].items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {status}: {n}")

        if "phases" in s:
            lines.append("\n  Phases:")
            for phase, n in sorted(s["phases"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {phase}: {n}")

        if "enrollment" in s:
            e = s["enrollment"]
            lines.append(f"\n  Enrollment: mean {e['mean']}, median {e['median']}, total {e['total']:,}")

        if "duration_months" in s:
            d = s["duration_months"]
            lines.append(f"  Duration: mean {d['mean']} months, median {d['median']} months")

        if "phase_completion" in s:
            lines.append("\n  Completion rates:")
            for phase, data in s["phase_completion"].items():
                lines.append(f"    {phase}: {data['completion_rate']}% completed (n={data['total']})")

        text = "\n".join(lines)
        print(text)
        return text

    def plot_timeline(self, timeline: pd.DataFrame, save_path: Optional[str] = None) -> Optional[str]:
        """Plot the enrollment timeline."""
        if len(timeline) < 12:
            return None

        slug = self.cancer_type.replace(" ", "_").lower()
        if save_path is None:
            save_path = f"outputs/{slug}_timeline.png"
        os.makedirs(os.path.dirname(save_path) or "outputs", exist_ok=True)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(timeline["month"], timeline["new_trials"],
               width=25, alpha=0.4, color="#94a3b8", label="Monthly")
        ax.plot(timeline["month"], timeline["rolling_6m"],
                color="#2563eb", linewidth=2, label="6-month avg")
        ax.plot(timeline["month"], timeline["rolling_12m"],
                color="#dc2626", linewidth=1.5, linestyle="--", label="12-month avg")

        ax.set_title(f"New Clinical Trials Started — {self.cancer_type}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("New Trials / Month")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path


# Need this for the type hint
from typing import Optional
