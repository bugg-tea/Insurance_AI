from typing import List, Dict
from collections import defaultdict
import re


class PolicyBuilder:
    """
    Converts raw chunks into structured insurance policy representation.
    """

    def build(self, chunks: List[Dict]) -> Dict[str, Dict]:

        policies = defaultdict(lambda: {
            "coverage_limit": "not mentioned",
            "room_rent": "not mentioned",
            "icu_charges": "not mentioned",
            "ped_waiting": "not mentioned",
            "specific_waiting": "not mentioned",
            "co_payment": "not mentioned",
            "network_hospitals": "not mentioned",
            "day_care": "not mentioned",
            "maternity": "not mentioned",
            "no_claim_bonus": "not mentioned",
            "exclusions": []
        })

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            company = chunk.get("company", "unknown")

            # -------- COVERAGE --------
            if "sum insured" in text or "coverage" in text:
                policies[company]["coverage_limit"] = chunk["text"]

            # -------- ROOM RENT --------
            if "room rent" in text:
                policies[company]["room_rent"] = chunk["text"]

            # -------- ICU --------
            if "icu" in text:
                policies[company]["icu_charges"] = chunk["text"]

            # -------- PED --------
            if "waiting period" in text or "ped" in text:
                if "pre-existing" in text:
                    policies[company]["ped_waiting"] = chunk["text"]
                else:
                    policies[company]["specific_waiting"] = chunk["text"]

            # -------- CO-PAYMENT --------
            if "co-payment" in text or "copayment" in text:
                policies[company]["co_payment"] = chunk["text"]

            # -------- NETWORK --------
            if "hospital" in text:
                policies[company]["network_hospitals"] = chunk["text"]

            # -------- DAY CARE --------
            if "day care" in text:
                policies[company]["day_care"] = chunk["text"]

            # -------- MATERNITY --------
            if "maternity" in text:
                policies[company]["maternity"] = chunk["text"]

            # -------- NO CLAIM BONUS --------
            if "no-claim" in text or "bonus" in text:
                policies[company]["no_claim_bonus"] = chunk["text"]

            # -------- EXCLUSIONS --------
            if "exclusion" in text or "not payable" in text:
                policies[company]["exclusions"].append(chunk["text"])

        return dict(policies)