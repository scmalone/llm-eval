import unittest

from llm_eval.haystack import build, records_for, render


def words(text):
    return len(text.split())


class HaystackTests(unittest.TestCase):
    spec = {"records": 12, "seed": 5}

    def test_prompt_is_deterministic_and_lists_every_record_in_order(self):
        first = render(self.spec, 200)
        self.assertEqual(first, render(self.spec, 200))
        prompt, expected = first
        lines = [line for line in prompt.splitlines() if line.startswith("RECORD:")]
        self.assertEqual(len(lines), 12)
        self.assertEqual([f"{s}:{c}" for s, c in records_for(self.spec)], expected)
        self.assertEqual(expected, [l.replace("RECORD: site=", "").replace(" code=", ":") for l in lines])

    def test_different_seeds_change_records_and_filler(self):
        self.assertNotEqual(render(self.spec, 50), render({"records": 12, "seed": 6}, 50))

    def test_build_lands_just_under_the_target(self):
        for target in (2000, 30000, 128000):
            built = build(self.spec, target, words)
            self.assertLessEqual(built["prompt_tokens"], target)
            self.assertGreater(built["prompt_tokens"], target * 0.99)
            self.assertEqual(built["prompt_tokens"], words(built["prompt"]))

    def test_build_works_with_a_nonlinear_counter(self):
        counter = lambda text: int(words(text) * 1.37) + 11
        built = build(self.spec, 90000, counter)
        self.assertLessEqual(built["prompt_tokens"], 90000)
        self.assertGreater(built["prompt_tokens"], 90000 * 0.99)

    def test_rejects_impossible_requests(self):
        with self.assertRaises(ValueError):
            build(self.spec, 5, words)
        with self.assertRaises(ValueError):
            records_for({"records": 99, "seed": 1})


if __name__ == "__main__":
    unittest.main()
