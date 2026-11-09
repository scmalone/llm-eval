"""The benchmark's pi is the owner's pi (2026-10-30): their model entry, their AGENTS.md, pi's own compaction.

Before this, every pi run sent reasoning_effort and the developer role (a reasoning entry the owner's models.json
does not declare), capped output at 16,384 where the owner's pi allows 32,768 (impl-llm-eval-haystack hit that cap
in three of six runs before writing a file), and loaded no AGENTS.md at all.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_workload_runner as fixtures
from llm_eval import paired
from llm_eval.plan import fingerprint
from llm_eval.workload import pi_config, prepare as workload
from llm_eval.workload.pi_driver import PiDriver

REQUEST = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "high"}


class ProductionEntryTests(unittest.TestCase):
    def test_strata_s_profile_is_pi_s_strata_entry(self):
        self.assertEqual(pi_config.production_entry("strata-iq3s"), {"contextWindow": 262144, "maxTokens": 32768})
        self.assertEqual(pi_config.production_entry("north-mini-code-1.0")["maxTokens"], 32768)
        self.assertIsNone(pi_config.production_entry("not-a-model"))

    def test_the_entry_declares_what_production_declares_and_compacts_at_the_owner_s_reserve(self):
        production = {"contextWindow": 65536, "maxTokens": 12288, "input": ["text", "image"]}
        models, settings, _ = pi_config.build("muse-glimmer-30b", 65536, 12288, REQUEST, production=production)
        entry = models["providers"]["local-llama"]["models"][0]
        # no reasoning flag: pi sends no reasoning_effort and uses the system role, as it does for the owner
        self.assertNotIn("reasoning", entry)
        self.assertNotIn("thinkingLevelMap", entry)
        self.assertEqual((entry["input"], entry["maxTokens"]), (["text", "image"], 12288))
        # the owner's settings.json reserve: one full 32,768 answer plus pi's 4,096 context margin
        self.assertEqual(settings["compaction"]["reserveTokens"], pi_config.production_compaction()[0])
        self.assertEqual(pi_config.production_compaction(), (36864, 20000))
        _, settings, _ = pi_config.build("m", 262144, 32768, REQUEST, compaction_at=200000, production=production)
        self.assertEqual(settings["compaction"]["reserveTokens"], 62144)

    def test_a_paired_seed_reaches_the_capture(self):
        # pi never sends a seed, so the capture must supply it, or the three seeds are three unseeded repeats
        expected = pi_config.expected_request("m", 131072, 16384, {**REQUEST, "seed": 29})
        self.assertEqual(expected["seed"], 29)
        self.assertNotIn("seed", pi_config.expected_request("m", 131072, 16384, REQUEST))


class ThinkingTests(unittest.TestCase):
    """pi sends thinking itself (2026-10-31). Before, no local entry declared reasoning, so pi sent nothing and the
    capture supplied reasoning_effort "high": llama.cpp passes that to the template, Swift's template raises on
    "high", Qwen3.8's ran at a level the owner's pi never asks for, and Gemma never thought at all."""

    # Values each template accepts (tokenizer.chat_template in the GGUFs, read 2026-10-31)
    TEMPLATE_VALUES = {"reasoning_effort": {"low", "medium", "xhigh"}, "reasoning_strength": {"low", "medium", "high", "xhigh"}}

    def entry(self, name):
        return pi_config.production_entry(name)

    def test_swift_never_gets_a_value_its_template_rejects(self):
        for level in pi_config.THINKING_LEVELS:
            sent = pi_config.thinking_request(self.entry("swift-qwen3.8-27b-vision"), level, 32768)
            effort = sent.get("chat_template_kwargs", {}).get("reasoning_effort")
            self.assertTrue(effort is None or effort in self.TEMPLATE_VALUES["reasoning_effort"], (level, effort))
        expected = pi_config.expected_request("swift-qwen3.8-27b-vision", 131072, 32768, REQUEST, "high",
                                              self.entry("swift-qwen3.8-27b-vision"))
        self.assertNotIn("reasoning_effort", expected)
        self.assertEqual(expected["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "xhigh"})
        self.assertEqual(expected["thinking_budget_tokens"], 16384)

    def test_every_local_entry_maps_every_level_to_values_its_template_takes(self):
        frozen = json.loads(pi_config.PRODUCTION.read_text())
        for name, entry in frozen["models"].items():
            for level in pi_config.THINKING_LEVELS:
                sent = pi_config.thinking_request(entry, level, entry["maxTokens"])
                for key, allowed in self.TEMPLATE_VALUES.items():
                    value = sent.get("chat_template_kwargs", {}).get(key)
                    self.assertTrue(value is None or value in allowed, (name, level, key, value))

    def test_the_owner_s_max_reaches_each_model_s_highest_level(self):
        self.assertEqual(pi_config.production_thinking_level(), "max")
        cases = {"qwen3.8-27b-q4": {"enable_thinking": True, "reasoning_effort": "xhigh"},
                 "muse-glimmer-30b-dflash": {"reasoning_strength": "high"},  # vendor default; xhigh unmapped in the benchmark copy (2026-11-04)
                 "gemma-4-26b-a4b-it": {"enable_thinking": True},
                 "north-mini-code-1.0": {"reasoning": True},
                 "occamy-1.0": {"enable_thinking": True, "preserve_thinking": True}}
        for name, kwargs in cases.items():
            sent = pi_config.thinking_request(self.entry(name), "max", 32768)
            self.assertEqual(sent, {"chat_template_kwargs": kwargs, "thinking_budget_tokens": 16384}, name)

    def test_off_turns_thinking_off_where_the_model_allows_it(self):
        self.assertEqual(pi_config.thinking_request(self.entry("qwen3.8-27b-q4"), "off", 32768),
                         {"chat_template_kwargs": {"enable_thinking": False}})
        # Glimmer cannot stop thinking: off is unsupported and clamps up to minimal, its lowest strength
        self.assertEqual(pi_config.clamp_level(self.entry("muse-glimmer-30b"), "off"), "minimal")
        self.assertEqual(pi_config.thinking_request(self.entry("muse-glimmer-30b"), "off", 32768),
                         {"chat_template_kwargs": {"reasoning_strength": "low"}, "thinking_budget_tokens": 1024})

    def test_the_budget_leaves_room_for_an_answer(self):
        sent = pi_config.thinking_request(self.entry("gemma-4-31b-it"), "high", 8192)
        self.assertEqual(sent["thinking_budget_tokens"], 8192 - pi_config.MIN_ANSWER_TOKENS)

    def test_the_sandbox_entry_carries_production_s_thinking_fields(self):
        production = self.entry("qwen3.8-27b-q4")
        models, _, _ = pi_config.build("qwen3.8-27b-q4", 131072, 32768, REQUEST, production=production)
        entry = models["providers"]["local-llama"]["models"][0]
        for field in pi_config.THINKING_FIELDS:
            self.assertEqual(entry[field], production[field])

    def test_an_entry_without_reasoning_still_gets_the_pinned_effort_from_the_capture(self):
        expected = pi_config.expected_request("strata", 262144, 32768, REQUEST, "high", self.entry("strata-iq3s"))
        self.assertEqual(expected["reasoning_effort"], "high")


class BlockTests(unittest.TestCase):
    def block(self, model_id, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        agents = Path(tmp.name) / "AGENTS.md"
        agents.write_text("# Working agreement\n\nOptions: lettered, one line each, exactly one recommended.\n")
        model = {"id": model_id, "contexts": [131072, 262144], "request": REQUEST}
        with mock.patch.object(workload, "pi_version", return_value="1.4.0"), \
                mock.patch.object(workload.shutil, "which", return_value="/usr/bin/bwrap"):
            block = workload.pi_workload_block(Path(tmp.name) / "plan", model, [], context=262144,
                                               agents_md=kwargs.pop("agents_md", agents), **kwargs)
        return block, agents

    def test_output_is_the_owner_s_max_tokens_unless_given(self):
        block, _ = self.block("strata-iq3s")
        self.assertEqual(block["mode"]["output_limit"], 32768)
        self.assertFalse(block["mode"]["output_overridden"])
        block, _ = self.block("strata-iq3s", output=16384)
        self.assertEqual((block["mode"]["output_limit"], block["mode"]["output_overridden"]), (16384, True))

    def test_the_owner_s_agents_md_is_pinned_where_pi_reads_it(self):
        block, agents = self.block("strata-iq3s")
        pinned = block["home_files"][(pi_config.AGENT_DIR / "AGENTS.md").as_posix()]
        self.assertEqual(Path(pinned).read_text(), agents.read_text())
        self.assertEqual(block["agents_md"], str(agents))

    def test_the_default_agents_md_is_the_frozen_bench_copy(self):
        import hashlib
        self.assertEqual(workload.PI_AGENTS_MD.name, "pi-agents-md-bench.md")
        self.assertEqual(hashlib.sha256(workload.PI_AGENTS_MD.read_bytes()).hexdigest()[:8], "69cbbcde")

    def test_the_compaction_point_is_pi_s_own(self):
        block, _ = self.block("strata-iq3s")
        self.assertEqual(block["mode"]["compaction_point"], 262144 - pi_config.production_compaction()[0])

    def test_without_agents_md_the_synthetic_entry_remains(self):
        block, _ = self.block("strata-iq3s", agents_md=None)
        self.assertEqual(block["mode"]["output_limit"], workload.SCREEN_OUTPUT)
        self.assertTrue(block["config"]["providers"]["local-llama"]["models"][0]["reasoning"])


class DriverTests(unittest.TestCase):
    def test_the_driver_writes_the_plan_s_pinned_files(self):
        # it used to rebuild them from its arguments, so --compaction-at never reached a run
        models, settings, ref = pi_config.build("m", 131072, 16384, REQUEST, compaction_at=100000)
        with tempfile.TemporaryDirectory() as tmp:
            PiDriver(Path(tmp), "m", ref, 8391, REQUEST, config=models, settings=settings).install()
            written = json.loads((Path(tmp) / pi_config.AGENT_DIR / "settings.json").read_text())
        self.assertEqual(written["compaction"]["reserveTokens"], 31072)


def seal(plan):
    plan["plan_hash"] = fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


class LlamaPairedTests(unittest.TestCase):
    def setUp(self):
        helper = fixtures.WorkloadCampaignTests()
        helper.setUp()
        self.addCleanup(helper.tearDown)
        self.root = helper.root / "paired"
        plan = helper.plan({"workload": {"impl-1": {"turns": [[{"src/app.py": fixtures.FIX}, "Fixed."]]}}})
        model = plan["models"][0]
        model["fixtures"] = [f for f in model["fixtures"] if f["id"] == "impl-1"]
        model.update(contexts=[65536], anchor=65536, request={"temperature": 0.6},
                     settings={"model": "/m.gguf", "spec-type": "draft-mtp", "cache-type-k": "q4_0"})
        self.a = seal(plan)

    def arm(self, change):
        b = copy.deepcopy(self.a)
        change(b["models"][0])
        return seal(b)

    def test_a_llama_knob_freezes_and_its_seed_goes_to_the_request_not_the_preset(self):
        b = self.arm(lambda m: m["settings"].pop("spec-type"))
        manifest = paired.freeze({"A": self.a, "B": b}, self.root, ["settings.spec-type"])
        plan = paired.attempt_plan(manifest, manifest["attempts"][0])
        self.assertEqual(plan["models"][0]["request"], {"temperature": 0.6, "seed": 17})
        self.assertNotIn("request", plan["models"][0]["settings"])  # settings become server flags

    def test_a_llama_knob_cannot_change_the_request(self):
        b = self.arm(lambda m: m["request"].update(temperature=1.0))
        with self.assertRaisesRegex(ValueError, "undeclared arm changes"):
            paired.freeze({"A": self.a, "B": b}, self.root, ["settings.spec-type"])
        with self.assertRaisesRegex(ValueError, "only explicit settings"):
            paired.freeze({"A": self.a, "B": b}, self.root, ["request.temperature"])

    def test_a_model_comparison_records_every_difference_but_keeps_tasks_and_context(self):
        def other_model(m):
            m.update(id="other-model", request={"temperature": 1.0})
            m["settings"]["model"] = "/other.gguf"
        manifest = paired.freeze({"A": self.a, "B": self.arm(other_model)}, self.root, kind="models")
        self.assertEqual(manifest["kind"], "models")
        self.assertIn("model", manifest["changes"])
        self.assertIn("request.temperature", manifest["changes"])
        moved = self.arm(lambda m: (other_model(m), m.update(contexts=[131072], anchor=131072),
                                    m["workload"].update(context=131072)))
        with self.assertRaisesRegex(ValueError, "keep tasks, context and harness"):
            paired.freeze({"A": self.a, "B": moved}, self.root / "moved", kind="models")

    def test_a_model_comparison_needs_two_models(self):
        with self.assertRaisesRegex(ValueError, "two different models"):
            paired.freeze({"A": self.a, "B": self.arm(lambda m: m["settings"].pop("spec-type"))}, self.root,
                          kind="models")


if __name__ == "__main__":
    unittest.main()
