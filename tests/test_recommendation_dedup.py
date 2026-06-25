import unittest

import numpy as np
import pandas as pd

from backend.geo import GeoIndex
from backend.itinerary_llm import (
    assemble_itinerary_plan,
    filter_to_recommendation_pool,
    recommendations_for_prompt,
)
from backend.recommender import _unique_place_positions


class RecommendationDedupTests(unittest.TestCase):
    def test_ranked_positions_dedupe_visible_place_names(self):
        locations = pd.DataFrame([
            {"location_id": "a", "location_name": "Garrett Popcorn Shops"},
            {"location_id": "b", "location_name": "Garrett Popcorn Shops"},
            {"location_id": "c", "location_name": "City Cruises Chicago"},
            {"location_id": "d", "location_name": "Shoreline Sightseeing"},
        ])

        picked = _unique_place_positions(locations, np.array([0, 1, 2, 3]), top_k=3)

        self.assertEqual(picked.tolist(), [0, 2, 3])

    def test_itinerary_prompt_pool_dedupes_visible_place_names(self):
        recs = [
            _rec("a", "Garrett Popcorn Shops"),
            _rec("b", "Garrett Popcorn Shops"),
            _rec("c", "City Cruises Chicago"),
        ]

        filtered = filter_to_recommendation_pool(recs, ["a", "b", "c"])
        prompt = recommendations_for_prompt(recs, geo=GeoIndex({}))

        self.assertEqual([r["location_id"] for r in filtered], ["a", "c"])
        self.assertEqual([r["location_id"] for r in prompt], ["a", "c"])

    def test_itinerary_assembly_dedupes_same_place_across_days(self):
        rec_by_id = {
            "a": _rec("a", "Garrett Popcorn Shops"),
            "b": _rec("b", "Garrett Popcorn Shops"),
            "c": _rec("c", "City Cruises Chicago"),
            "d": _rec("d", "Shoreline Sightseeing"),
        }
        llm_payload = {
            "summary": "test",
            "days": [
                {"day_number": 1, "theme": "Day 1", "stops": [
                    {"location_id": "a", "slot": "morning"},
                    {"location_id": "c", "slot": "afternoon"},
                ]},
                {"day_number": 2, "theme": "Day 2", "stops": [
                    {"location_id": "b", "slot": "morning"},
                    {"location_id": "d", "slot": "afternoon"},
                ]},
            ],
        }

        plan = assemble_itinerary_plan(llm_payload, rec_by_id, GeoIndex({}), trip_days=2)
        names = [
            stop["location_name"]
            for day in plan["days"]
            for stop in day["stops"]
        ]

        self.assertEqual(names.count("Garrett Popcorn Shops"), 1)


def _rec(location_id, name):
    return {
        "location_id": location_id,
        "location_name": name,
        "primary_category": "Attractions",
        "categories": ["Attractions"],
        "reason": "",
        "final_score": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
