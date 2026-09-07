from __future__ import annotations

import copy
import unittest

from benchmark.validate_pyg_scenario import validate_scenario


def valid_scenario() -> dict:
    return {
        "grid": {
            "nodes": {
                "bus": [[230.0, 3.0, 0.9, 1.1]],
                "generator": [[100.0, 0.0, 0.0, 2.0, 0.0, -1.0, 1.0, 1.0, 0.1, 1.0, 0.0]],
            },
            "edges": {
                "ac_line": {"senders": [0], "receivers": [0], "features": [[-1.0, 1.0, 0.0, 0.0, 0.01, 0.1, 2.0, 2.0, 2.0]]},
                "transformer": {"senders": [], "receivers": [], "features": []},
            },
        },
        "solution": {
            "nodes": {"bus": [[0.0, 1.0]], "generator": [[1.0, 0.0]]},
            "edges": {
                "ac_line": {"senders": [0], "receivers": [0], "features": [[0.0, 0.0, 0.0, 0.0]]},
                "transformer": {"senders": [], "receivers": [], "features": []},
            },
        },
        "metadata": {
            "termination_status": "LOCALLY_SOLVED",
            "objective": 10.0,
            "solve_time_seconds": 0.5,
        },
    }


class ValidatePygScenarioTest(unittest.TestCase):
    def test_accepts_complete_solved_scenario(self) -> None:
        summary = validate_scenario(valid_scenario())
        self.assertEqual(summary["n_buses"], 1)
        self.assertEqual(summary["n_generators"], 1)
        self.assertEqual(summary["n_ac_lines"], 1)

    def test_rejects_unsolved_status(self) -> None:
        scenario = valid_scenario()
        scenario["metadata"]["termination_status"] = "INFEASIBLE"
        with self.assertRaisesRegex(ValueError, "not solved"):
            validate_scenario(scenario)

    def test_rejects_incomplete_bus_solution(self) -> None:
        scenario = valid_scenario()
        scenario["solution"]["nodes"]["bus"] = []
        with self.assertRaisesRegex(ValueError, "bus input and solution counts differ"):
            validate_scenario(scenario)

    def test_rejects_branch_shape_mismatch(self) -> None:
        scenario = copy.deepcopy(valid_scenario())
        scenario["solution"]["edges"]["ac_line"]["features"] = []
        with self.assertRaisesRegex(ValueError, "sender/receiver/feature counts differ"):
            validate_scenario(scenario)

    def test_rejects_non_finite_solution(self) -> None:
        scenario = valid_scenario()
        scenario["solution"]["nodes"]["bus"][0][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
