# 📈 Clinical Trial Enrollment Forecaster

**Forecasting cancer clinical trial enrollment trends using ARIMA and Google's TimesFM 2.5, with AI-generated plain-language summaries powered by Gemini.**

> *Companion project to [Clinical Trial Navigator](https://github.com/mirakw/clinical-trial-navigator). That project helps patients find the right trial today. This one looks ahead — forecasting where the cancer trial landscape is heading, so sponsors, researchers, and advocates can plan accordingly.*

---

## What It Does

You give it a cancer type. The tool:

1. **Fetches** historical trial data across all statuses (completed, recruiting, terminated) from ClinicalTrials.gov API v2
2. **Analyzes** the landscape — phase distribution, completion rates, sponsor mix, intervention trends
3. **Builds** a monthly time series of new trial starts going back ~40 years
4. **Forecasts** 12 months ahead using ARIMA (and optionally TimesFM 2.5 with quantile uncertainty bands)
5. **Summarizes** everything in plain language using Gemini AI — both a big-picture overview and a specific walkthrough of what each output chart and data file shows
6. **Compares** trends across cancer types with a cross-cancer summary

Works with **any cancer type** — breast cancer, melanoma, glioblastoma, pancreatic cancer, stomach cancer, etc.

### Example
```
Input: python main.py --cancer "stomach cancer"

Output:
  STOMACH CANCER TRIAL LANDSCAPE
  Total trials: 1,000
  Phase 2 dominates (196 trials), followed by Phase 1 (105) and Phase 3 (86)
  Completion rate: ~33% across Phase 1-3

  📝 SUMMARY:
  Right now, there are about 1,000 clinical trials focused on stomach cancer.
  Most are in Phase 2, although a significant number are in early stages.
  The forecast suggests about 1.7 new trials per month, but with a wide
  uncertainty range from 0 to 4.8 — the model isn't very confident, which
  makes sense given how volatile the monthly numbers have been recently.

  ✓ Timeline chart: outputs/stomach_cancer_timeline.png
  ✓ Forecast chart: outputs/stomach_cancer_forecast.png
  ✓ Summary: outputs/stomach_cancer_summary.txt
```

---

## Quick Start
```bash
git clone https://github.com/mirakw/clinical-trial-forecaster.git
cd clinical-trial-forecaster

pip install -r requirements.txt

# Set your Gemini API key (free at https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-key-here"

# Run full demo (4 cancer types + cross-cancer comparison)
python main.py

# Any cancer type you want
python main.py --cancer "stomach cancer"
python main.py --cancer "melanoma"
python main.py --cancer "pancreatic cancer"

# Combine flags
python main.py --cancer "glioblastoma" --horizon 24

# Optional: Install TimesFM for foundation model forecasting (~1GB download)
pip install timesfm torch
```

### Notebook
```bash
pip install jupyter
jupyter notebook demo_notebook.ipynb
```

---

## Architecture
```
Cancer Type (any cancer)
    │
    ▼
ClinicalTrials.gov API v2 → Fetch historical trials (all statuses)
    │
    ▼
Analyzer → Monthly time series + landscape stats
    │
    ▼
ARIMA / TimesFM 2.5 → 12-month forecast + uncertainty bands
    │
    ▼
Gemini AI → Plain-language summary of results
    │
    ▼
Outputs (charts, data, summary)
```

---

## Output Files

For each cancer type, the tool generates 5 outputs:

| File | What It Shows |
|------|---------------|
| `timeline.png` | Chart of new trials started per month over the past ~40 years, with 6-month and 12-month rolling averages |
| `timeline.csv` | Raw monthly data behind the chart, for your own analysis |
| `forecast.png` | Chart of the 12-month forecast with a shaded 90% confidence interval |
| `forecast.json` | Raw forecast numbers — predicted trials/month and the confidence range |
| `summary.txt` | Gemini-generated plain-language explanation of all outputs — what the trends mean, what the forecast predicts, and what to take away |

---

## Demo Scenarios

The default demo (`python main.py`) runs four cancer types:

| Cancer Type | Why |
|------------|-----|
| Breast cancer | Largest oncology trial volume |
| Non-small cell lung cancer | Fastest-growing immunotherapy pipeline |
| Colorectal cancer | Rising trial activity with MSI-targeted therapies |
| Acute lymphoblastic leukemia | Active CAR-T and immunotherapy pipeline |

Plus a cross-cancer comparison summary at the end.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) | Historical trial data (no auth required) |
| [ARIMA](https://www.statsmodels.org/) | Classical statistical time-series forecasting |
| [TimesFM 2.5](https://github.com/google-research/timesfm) | Google's 200M param foundation model for time series (optional) |
| [Gemini API](https://aistudio.google.com/apikey) | Plain-language summaries of results |

### Why TimesFM?

TimesFM is Google Research's pretrained foundation model for time-series forecasting (ICML 2024). It produces **quantile forecasts** — uncertainty bands (10th-90th percentile), not just point estimates. In clinical development, knowing the range of possible outcomes matters as much as the point estimate. This is the same principle behind approaches like prognostic covariate adjustment — using predictions **with quantified uncertainty** to make better decisions.

---

## Project Structure
```
clinical-trial-forecaster/
├── main.py              # CLI entry point
├── trial_fetcher.py     # ClinicalTrials.gov API v2 client
├── analyzer.py          # Landscape analysis + time series construction
├── forecaster.py        # TimesFM + ARIMA forecasting
├── insights.py          # Gemini-powered plain-language summaries
├── demo_notebook.ipynb  # Showcase notebook with inline charts
├── requirements.txt
└── outputs/             # Generated charts, summaries, data
```

---

## Limitations

- Forecasts are based on historical trial start patterns — they don't account for regulatory changes, funding shifts, or breakthrough approvals that could change the landscape
- ClinicalTrials.gov data may have reporting delays
- ARIMA works best when the underlying trend is stable — volatile cancer types will have wide confidence intervals
- TimesFM requires ~1GB model download and PyTorch installed

---

## Author

**Mira Kapoor Wadehra** — AI Product Manager
[LinkedIn](https://linkedin.com/in/mira-wadehra)

---

## License

MIT