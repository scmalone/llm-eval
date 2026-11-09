import unittest

from llm_eval import htmlreport
from llm_eval.workload import pi_config
from llm_eval.workload.inference import thinking_cap_hits


def turn(budget, generated, reasoning_tokens, answer=""):
    return {"effective_body": {"thinking_budget_tokens": budget},
            "response": {"usage": {"completion_tokens": generated}, "reasoning": "x" * (4 * reasoning_tokens),
                         "answer": answer}}


class ThinkingCapTests(unittest.TestCase):
    def test_a_turn_forced_closed_at_the_budget_counts(self):
        # captured Qwen3.6 loop turn: 16,758 generated, ~17,155 reasoning tokens by the chars/4 estimate
        self.assertEqual(thinking_cap_hits([turn(16384, 16758, 17155)]), 1)

    def test_a_long_turn_that_spent_its_tokens_on_the_answer_does_not(self):
        self.assertEqual(thinking_cap_hits([turn(16384, 18000, 6000, answer="y" * 40000)]), 0)

    def test_a_turn_below_the_budget_cannot_have_hit_it(self):
        # captured: 15,792 generated with ~15,455 reasoning, under 16,384
        self.assertEqual(thinking_cap_hits([turn(16384, 15792, 15455)]), 0)

    def test_no_budget_means_no_cap(self):
        self.assertEqual(thinking_cap_hits([turn(None, 40000, 40000), {"response": {}}]), 0)


class ThinkingReportTests(unittest.TestCase):
    def workload(self, profile):
        return {"production_entry": pi_config.production_entry(profile), "thinking": "max",
                "mode": {"output_limit": 32768}}

    def test_each_model_shows_what_pi_sends_at_max(self):
        self.assertEqual(htmlreport._thinking_sent(self.workload("qwen3.8-27b-unsloth-q4-mtp")),
                         "enable_thinking=True, reasoning_effort=xhigh, thinking_budget_tokens=16384")
        self.assertEqual(htmlreport._thinking_sent(self.workload("muse-glimmer-30b")),
                         "reasoning_strength=high, thinking_budget_tokens=16384")
        self.assertEqual(htmlreport._thinking_sent(self.workload("north-mini-code-1.0")),
                         "reasoning=True, thinking_budget_tokens=16384")

    def test_no_pi_workload_shows_nothing(self):
        self.assertEqual(htmlreport._thinking_sent({}), "")

    def test_cap_turns_are_unknown_for_runs_recorded_before_the_count(self):
        self.assertIsNone(htmlreport._cap_turns([{"reasoning_from_capture": {"blocks": 3}}, {}]))
        self.assertEqual(htmlreport._cap_turns([{"reasoning_from_capture": {"cap_hits": 2}},
                                                {"reasoning_from_capture": {"cap_hits": 0}}, {}]), 2)


if __name__ == "__main__":
    unittest.main()
