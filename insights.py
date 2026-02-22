"""
Insight Generator
==================
Uses Google Gemini to translate raw analysis and forecast results
into plain-language summaries that anyone can understand.
"""

import os
import json
from typing import Optional

import google.generativeai as genai


class InsightGenerator:
    """Generate plain-language summaries of trial landscape analysis and forecasts."""

    def __init__(self, gemini_api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable, "
                "or pass gemini_api_key to constructor."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def summarize_cancer_type(self, cancer_type: str, summary: dict,
                               forecast_results: dict) -> str:
        """Generate a plain-language summary for a single cancer type."""
        prompt = f"""You are a clinical research analyst explaining results to someone who is NOT a data scientist.

Here is the data for {cancer_type} clinical trials:

LANDSCAPE STATS:
{json.dumps(summary, indent=2, default=str)}

FORECAST RESULTS (next 12 months):
{json.dumps(forecast_results, indent=2, default=str)}

Write a summary with TWO sections. Use plain text only — no markdown, no bullet points, no bold, no headers.

SECTION 1 — THE BIG PICTURE (4-6 sentences):
Give a plain-language overview of the {cancer_type} trial landscape. How many trials exist? What phases dominate? Is the landscape growing, stable, or shrinking? What does the forecast predict for the next 12 months? What's one notable insight (like completion rates or sponsor mix)? Be specific with numbers.

SECTION 2 — WHAT THE OUTPUTS SHOW:
Explain what each of the 4 output files specifically shows for {cancer_type}. Do NOT give generic definitions of the file types. Instead, describe the SPECIFIC content using the actual data provided above. Use real numbers, real trends, and real ranges from the data.

For the timeline chart and data: Describe the specific trend for {cancer_type}. Did trials start growing in a specific decade? What was the peak? Has activity dropped recently? Use the actual data to describe what someone will see when they open the chart.

For the forecast chart and data: Use the ACTUAL forecast numbers to explain what the model predicts. For example, say "the model predicts about X new trials per month, but the real number could be anywhere from Y to Z" using the real numbers from the forecast data above. If the confidence interval is wide (like 0 to 4), say that and explain what it means — that there's a lot of uncertainty. If the forecast is lower than recent historical activity, point that out.

Write conversationally. No jargon. No markdown formatting. Separate the two sections with a blank line."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Summary unavailable: {e}"

    def summarize_comparison(self, all_summaries: dict, all_forecasts: dict) -> str:
        """Generate a cross-cancer comparison summary."""
        prompt = f"""You are a clinical research analyst writing a cross-cancer comparison brief for a non-technical audience.

Here is data across multiple cancer types:

LANDSCAPE SUMMARIES:
{json.dumps(all_summaries, indent=2, default=str)}

FORECAST RESULTS:
{json.dumps(all_forecasts, indent=2, default=str)}

Write a 5-8 sentence comparison that covers:
1. Which cancer type has the most active trial landscape
2. Which is growing fastest / slowest
3. How completion rates compare
4. Any surprising findings across the cancers
5. What this means for the overall oncology trial landscape

Be specific with numbers. Write conversationally — no jargon, no bullet points, no markdown.
Just clear, plain text paragraphs."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Comparison summary unavailable: {e}"