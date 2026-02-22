"""
ClinicalTrials.gov Historical Data Fetcher
============================================
Fetches clinical trial data across all statuses for time-series analysis.
Optimized for building enrollment timelines, not patient matching.

API: https://clinicaltrials.gov/data-api/api
No auth required. Rate limit: ~10 req/sec.
"""

import requests
import time
from typing import Optional


class TrialFetcher:
    """Fetch historical trial data from ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, delay: float = 0.12):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClinicalTrialForecaster/1.0"
        })

    def fetch_trials(
        self,
        condition: str,
        max_results: int = 1000,
        phase: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch trials across ALL statuses for a condition.
        Returns the fields needed for enrollment time-series analysis.
        """
        all_studies = []
        page_token = None
        page_size = min(max_results, 100)

        while len(all_studies) < max_results:
            params = {
                "query.cond": condition,
                "pageSize": page_size,
                "format": "json",
                "countTotal": "true",
            }
            if phase:
                params["filter.phase"] = phase
            if page_token:
                params["pageToken"] = page_token

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                print(f"  ⚠ API error: {e}")
                break

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                parsed = self._parse_study(study)
                if parsed:
                    all_studies.append(parsed)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(self.delay)

        return all_studies[:max_results]

    def _parse_study(self, raw: dict) -> Optional[dict]:
        """Parse API response — only the fields needed for time-series analysis."""
        try:
            proto = raw.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})

            enrollment_info = design_mod.get("enrollmentInfo", {})
            lead_sponsor = sponsor_mod.get("leadSponsor", {})
            phases = design_mod.get("phases", [])
            start_struct = status_mod.get("startDateStruct", {})
            completion_struct = status_mod.get("completionDateStruct", {})
            interventions = arms_mod.get("interventions", [])

            return {
                "nct_id": id_mod.get("nctId", ""),
                "brief_title": id_mod.get("briefTitle", ""),
                "overall_status": status_mod.get("overallStatus", ""),
                "phase": ", ".join(phases) if phases else "N/A",
                "study_type": design_mod.get("studyType", ""),
                "enrollment_count": enrollment_info.get("count"),
                "enrollment_type": enrollment_info.get("type", ""),
                "start_date": start_struct.get("date", ""),
                "completion_date": completion_struct.get("date", ""),
                "study_first_post_date": status_mod.get(
                    "studyFirstPostDateStruct", {}
                ).get("date", ""),
                "lead_sponsor": lead_sponsor.get("name", ""),
                "sponsor_class": lead_sponsor.get("class", ""),
                "conditions": cond_mod.get("conditions", []),
                "intervention_types": list(set(
                    i.get("type", "") for i in interventions
                )),
                "intervention_names": [i.get("name", "") for i in interventions],
            }
        except Exception:
            return None
