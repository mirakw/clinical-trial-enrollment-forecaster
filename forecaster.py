"""
Enrollment Forecaster
======================
Forecasts clinical trial enrollment trends using:
  - TimesFM 2.5 (Google's 200M param foundation model) — primary
  - ARIMA — statistical baseline for comparison

TimesFM produces both point forecasts and quantile forecasts (uncertainty
bands), which is key: in clinical development, knowing the range of possible
outcomes matters as much as the point estimate. This is the same principle
behind Unlearn AI's PROCOVA method, which uses prognostic scores with
uncertainty to optimize trial design.
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import torch
    import timesfm
    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False


class EnrollmentForecaster:
    """Forecast monthly trial enrollment using TimesFM and ARIMA."""

    def __init__(self, timeline: pd.DataFrame, cancer_type: str = "",
                 forecast_months: int = 12):
        """
        Args:
            timeline: Monthly time series from TrialAnalyzer.build_monthly_timeline()
            cancer_type: Label for charts
            forecast_months: How far ahead to forecast
        """
        self.timeline = timeline
        self.cancer_type = cancer_type
        self.forecast_months = forecast_months
        self.results = {}
        self._timesfm_model = None

    def _load_timesfm(self):
        """Lazy-load TimesFM model."""
        if not HAS_TIMESFM:
            return None
        if self._timesfm_model is not None:
            return self._timesfm_model

        print("  ⏳ Loading TimesFM 2.5 (200M)...")
        try:
            torch.set_float32_matmul_precision("high")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            self._timesfm_model = model
            print("  ✓ TimesFM loaded")
            return model
        except Exception as e:
            print(f"  ⚠ TimesFM load failed: {e}")
            return None

    def forecast_timesfm(self) -> Optional[dict]:
        """
        Forecast using TimesFM 2.5.
        Returns point forecast + quantile forecasts (10th-90th percentile).
        """
        model = self._load_timesfm()
        if model is None or len(self.timeline) < 12:
            return None

        try:
            y = self.timeline["new_trials"].values.astype(np.float32)

            point_forecast, quantile_forecast = model.forecast(
                horizon=self.forecast_months,
                inputs=[y],
            )

            point = np.maximum(point_forecast[0], 0)
            quantiles = np.maximum(quantile_forecast[0], 0)

            # quantiles shape: (forecast_months, 11)
            # Index 0 = mean, 1-10 = 10th through 90th percentiles
            return {
                "model": "TimesFM 2.5",
                "forecast": point.tolist(),
                "forecast_mean": float(np.mean(point)),
                "quantile_10": quantiles[:, 1].tolist() if quantiles.shape[1] > 1 else None,
                "quantile_50": quantiles[:, 5].tolist() if quantiles.shape[1] > 5 else None,
                "quantile_90": quantiles[:, 9].tolist() if quantiles.shape[1] > 9 else None,
            }
        except Exception as e:
            print(f"  ⚠ TimesFM forecast failed: {e}")
            return None

    def forecast_arima(self) -> Optional[dict]:
        """ARIMA baseline forecast with confidence intervals."""
        if not HAS_STATSMODELS or len(self.timeline) < 24:
            return None

        try:
            y = self.timeline["new_trials"].values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=(2, 1, 2))
                fitted = model.fit()
                result = fitted.get_forecast(steps=self.forecast_months)

            mean = np.maximum(np.array(result.predicted_mean), 0)
            ci = np.array(result.conf_int(alpha=0.1))

            return {
                "model": "ARIMA(2,1,2)",
                "forecast": mean.tolist(),
                "forecast_mean": float(np.mean(mean)),
                "conf_lower": np.maximum(ci[:, 0], 0).tolist(),
                "conf_upper": ci[:, 1].tolist(),
                "aic": fitted.aic,
            }
        except Exception as e:
            print(f"  ⚠ ARIMA failed: {e}")
            return None

    def run(self) -> dict:
        """Run all available models and return results."""
        if len(self.timeline) < 12:
            print(f"  ⚠ Only {len(self.timeline)} months of data — need 12+ for forecasting")
            return {}

        print(f"\n  Forecasting: {self.cancer_type}")
        print(f"  Historical: {len(self.timeline)} months")
        print(f"  Horizon: {self.forecast_months} months")

        results = {}

        # TimesFM (primary)
        tfm = self.forecast_timesfm()
        if tfm:
            results["timesfm"] = tfm
            print(f"  ✓ TimesFM → avg {tfm['forecast_mean']:.1f} new trials/month")

        # ARIMA (baseline)
        arima = self.forecast_arima()
        if arima:
            results["arima"] = arima
            print(f"  ✓ ARIMA   → avg {arima['forecast_mean']:.1f} new trials/month")

        if not results:
            print("  ⚠ No models ran successfully")

        self.results = results
        return results

    def plot(self, save_path: Optional[str] = None) -> Optional[str]:
        """Generate forecast chart with historical data + predictions + uncertainty."""
        if not self.results or len(self.timeline) < 12:
            return None

        slug = self.cancer_type.replace(" ", "_").lower()
        if save_path is None:
            save_path = f"outputs/{slug}_forecast.png"
        os.makedirs(os.path.dirname(save_path) or "outputs", exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Historical
        ax.plot(self.timeline["month"], self.timeline["new_trials"],
                color="#94a3b8", alpha=0.4, linewidth=1, label="Observed")
        ax.plot(self.timeline["month"], self.timeline["rolling_6m"],
                color="#2563eb", linewidth=2, label="6-month rolling avg")

        # Future dates
        last_date = self.timeline["month"].max()
        future = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=self.forecast_months, freq="MS"
        )

        colors = {"timesfm": "#dc2626", "arima": "#059669"}

        for name, result in self.results.items():
            color = colors.get(name, "#6b7280")
            label = result.get("model", name)

            ax.plot(future, result["forecast"],
                    color=color, linewidth=2, linestyle="--",
                    marker="o", markersize=3, label=f"{label} forecast")

            # Uncertainty bands
            if "quantile_10" in result and result["quantile_10"]:
                ax.fill_between(future, result["quantile_10"], result["quantile_90"],
                                alpha=0.12, color=color, label=f"{label} 10-90th pctile")
            elif "conf_lower" in result:
                ax.fill_between(future, result["conf_lower"], result["conf_upper"],
                                alpha=0.12, color=color, label=f"{label} 90% CI")

        ax.axvline(x=last_date, color="#64748b", linestyle=":", alpha=0.5)
        ax.set_title(f"Clinical Trial Enrollment Forecast — {self.cancer_type}\n"
                     f"New trials/month + {self.forecast_months}-month forecast",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("New Trials / Month")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return save_path
