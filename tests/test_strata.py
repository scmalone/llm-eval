import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import acceptance, strata
from llm_eval.adapters import CapacityError, HostMemoryError, InfrastructureError
from llm_eval.plan import compile_plan, digest, verify_plan
from llm_eval.runtime import adapter_class
from llm_eval.runner import execute
from test_live import ScriptedHost


class StrataHost(ScriptedHost):
    mem_available_mib = 200000

    def read_text(self, path):
        if str(path) == "/proc/meminfo":
            return f"MemTotal:       64668744 kB\nMemAvailable:   {self.mem_available_mib * 1024} kB\n"
        return super().read_text(path)

    def _launch(self, argv):
        command = argv[argv.index('--') + 1:]
        cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
        context = cfg['args'][cfg['args'].index('--max-context') + 1]
        translated = argv[:argv.index('--') + 1] + ['engine', '--ctx-size', context, '--port', str(cfg['port'])]
        return super()._launch(translated)

    def http(self, method, port, path, body=None, timeout=30):
        if port != 8080 and self.server:
            if path == '/health':
                return 200, {'loaded': True, 'service': 'strata'}
            if path == '/v1/messages/count_tokens':
                return 200, {'input_tokens': len(body['messages'][0]['content'].split()) + 5}
            if path == '/metrics':
                return 200, {'live': {'state': 'generating', 'generated': 7, 'elapsed_s': self.clock}}
        return super().http(method, port, path, body, timeout)


class StrataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('engine', 'server.py', 'model.gguf', 'ple.gguf', 'mtp/rt', 'tok/vocab.json', 'pack/data.bin'):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')
        self.config = self.root / 'strata.json'
        self.raw = {'exe': 'engine', 'cwd': str(self.root), 'tokenizer': 'tok', 'args': [
            '--pack', 'pack', '--native', 'model.gguf', '--ple-gguf', 'ple.gguf', '--mtp', 'mtp/rt',
            '--max-context', '262144', '--kv', 'int8'], 'log': '/original.log'}
        self.config.write_text(json.dumps(self.raw))
        self.fixture = self.root / 'smoke.json'
        self.fixture.write_text(json.dumps({'id': 'smoke', 'kind': 'smoke',
            'prompt': 'Reply with the single word OK', 'max_output_tokens': 512, 'expect': {'exact': 'OK'}}))
        self.manifest = {'schema_version': 1, 'backend': 'live', 'runtime': 'strata',
            'production': {'unit': 'llama-router.service', 'restore_model': 'tiny'},
            'models': [{'id': 'strata-iq3s', 'anchor': 131072, 'contexts': [131072],
                'settings': {'config': str(self.config), 'python': sys.executable, 'server': str(self.root/'server.py')},
                'artifacts': [{'role': 'launch_config', 'path': str(self.config), 'sha256': digest(self.config)}],
                'fixtures': [str(self.fixture)]}]}
        self.path = self.root/'manifest.json'

    def plan(self):
        self.path.write_text(json.dumps(self.manifest))
        return compile_plan(self.path)

    def adapter(self, **kwargs):
        plan = self.plan()
        host = StrataHost(**kwargs)
        adapter = strata.StrataAdapter(plan, self.root/'logs', host, run_id='strata-test')
        return plan, host, adapter

    def test_pins_config_model_ple_pack_tokenizer_engine_and_code(self):
        plan = self.plan()
        for name in ('strata.json','engine','server.py','model.gguf','ple.gguf','mtp/rt','tok/vocab.json','pack/data.bin'):
            self.assertIn(str(self.root/name), plan['files'])
        self.assertIs(adapter_class(plan), strata.StrataAdapter)
        (self.root/'model.gguf').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'planned input changed'):
            verify_plan(plan)

    def test_config_tamper_rejected(self):
        self.config.write_text('{}')
        with self.assertRaises((ValueError, KeyError)):
            self.plan()

    def test_rejects_unknown_runtime(self):
        self.manifest['runtime'] = 'other'
        with self.assertRaisesRegex(ValueError,'runtime'):
            self.plan()

    def test_rejects_config_execution_hooks(self):
        self.raw['before_load'] = 'touch /tmp/no'
        self.config.write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, 'unsupported keys'):
            self.plan()

    def test_refuses_missing_shard(self):
        first = self.root/'model-00001-of-00002.gguf'
        first.write_text('shard')
        self.raw['args'][3] = str(first)
        self.config.write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, 'dependency missing'):
            self.plan()

    def test_ladder_offers_the_rungs_below_the_anchor(self):
        self.assertEqual(strata.ladder(131072),
                         [65536, 86016, 98304, 102400, 114688, 131072])

    def test_ladder_never_offers_a_rung_above_the_anchor(self):
        self.assertNotIn(262144, strata.ladder(131072))

    def test_the_native_window_can_be_the_anchor(self):
        self.assertEqual(strata.ladder(262144)[-2:], [131072, 262144])

    def test_a_262k_strata_plan_compiles(self):
        self.manifest['models'][0]['anchor'] = 262144
        self.manifest['models'][0]['contexts'] = strata.ladder(262144)
        plan = self.plan()
        self.assertEqual(plan['models'][0]['anchor'], 262144)
        self.assertIn(131072, plan['models'][0]['contexts'])

    def test_a_single_rung_strata_plan_still_compiles(self):
        plan = self.plan()          # the fixture declares contexts [131072] only
        self.assertEqual(plan['models'][0]['contexts'], [131072])

    def test_run_config_is_local_and_does_not_mutate_original(self):
        settings = self.plan()['models'][0]['settings']
        original = self.config.read_bytes()
        dest = strata.run_config(settings, 131072, 41234, self.root/'run')
        cfg = json.loads(dest.read_text())
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual(cfg['host'], '127.0.0.1')
        self.assertEqual(cfg['port'], 41234)
        self.assertEqual(cfg['args'].count('--max-context'), 1)
        self.assertEqual(cfg['args'][cfg['args'].index('--max-context')+1], '131072')
        self.assertEqual(cfg['sampling']['temperature'], 1.0)
        self.assertTrue(cfg['log'].startswith(str(self.root/'run')))

    def test_readiness_requires_engine_loaded(self):
        _, _, adapter = self.adapter()
        self.assertFalse(adapter.is_ready(200, {'status': 'ok', 'loaded': False}))
        self.assertFalse(adapter.is_ready(200, 'ok'))
        self.assertTrue(adapter.is_ready(200, {'loaded': True}))

    def test_load_count_and_restore_share_lifecycle(self):
        plan, host, adapter = self.adapter()
        snapshot = adapter.snapshot()
        adapter.quiesce()
        result = adapter.load(plan['models'][0], 131072)
        # The KV share now comes from the measured slope (1.9 GiB per 131,072 tokens), so a lower rung can be
        # forecast instead of loaded blind; the engine's own cache line supplies the rest when the log has it.
        self.assertEqual(result['kv_mib'], 1946)
        self.assertEqual(adapter.tokenize({'prompt': 'one two'}), 7)
        self.assertEqual(adapter.progress_signal(), adapter.progress_signal()) # elapsed alone is not progress
        adapter.stop()
        adapter.restore(snapshot)
        self.assertTrue(host.unit_active)
        self.assertEqual(host.probed, 1)

    def test_oom_cleanup_and_memory_obligation(self):
        plan, host, adapter = self.adapter(oom_above=100000, stuck_after_failed_load=True)
        adapter.snapshot(); adapter.quiesce()
        with self.assertRaises(InfrastructureError):
            adapter.load(plan['models'][0],131072)
        self.assertTrue(adapter.owes_memory)
        with self.assertRaises(InfrastructureError):
            adapter.load(plan['models'][0],65536)

    def test_unknown_gpu_client_blocks_before_quiesce(self):
        plan, host, adapter = self.adapter(procs=[{'pid': 99,'name':'other','mib':2000}])
        state = execute(plan,self.root/'runs','strata-test',adapter=adapter,lock_path=self.root/'lock')
        self.assertEqual(state['status'],'blocked')
        self.assertTrue(host.unit_active)

    def test_complete_classic_run_restores_production(self):
        plan, host, adapter = self.adapter()
        state = execute(plan,self.root/'runs','strata-test',adapter=adapter,lock_path=self.root/'lock')
        self.assertEqual(state['status'],'complete',state)
        self.assertFalse(state['restoration_required'])
        self.assertTrue(host.unit_active)
        self.assertTrue(any(s.get('status') == 'pass' for s in state['sections'].values()))

    def test_low_ram_is_refused_before_the_scratch_server_starts(self):
        # 2026-11-01 22:05: a load begun with 47,400 MiB available for a 52,278 MiB model stalled until
        # systemd-oomd killed it twice, and the owner's terminal with it.
        plan, host, adapter = self.adapter()
        host.mem_available_mib = 1000  # the fixture model is a few bytes, so the gate needs the 57,014 MiB floor
        adapter.snapshot(); adapter.quiesce()
        with self.assertRaisesRegex(HostMemoryError, r'^refused: 1,000 MiB RAM available, the load needs 57,014 MiB'):
            adapter.load(plan['models'][0], 131072)
        self.assertFalse([c for c in host.calls if c[0] == 'systemd-run'])
        self.assertFalse(adapter.owes_memory)

    def test_a_ram_refusal_is_not_retried_and_production_comes_back(self):
        plan, host, adapter = self.adapter()
        host.mem_available_mib = 1000
        state = execute(plan, self.root/'runs', 'strata-test', adapter=adapter, lock_path=self.root/'lock')
        failure = state['sections']['strata-iq3s__failure']
        self.assertEqual(failure['status'], 'infra_fail')
        self.assertTrue(failure['reason'].startswith('refused: '), failure)
        self.assertFalse([c for c in host.calls if c[0] == 'systemd-run'])
        self.assertEqual(state['sections']['strata-iq3s__context_qualification']['status'], 'skipped')
        self.assertFalse(state['restoration_required'])
        self.assertTrue(host.unit_active)

    def test_enough_ram_passes_the_gate(self):
        plan, host, adapter = self.adapter()
        host.mem_available_mib = 57014
        adapter.snapshot(); adapter.quiesce()
        adapter.load(plan['models'][0], 131072)
        self.assertEqual(len([c for c in host.calls if c[0] == 'systemd-run']), 1)

    def test_one_mib_under_the_floor_is_refused(self):
        plan, host, adapter = self.adapter()
        host.mem_available_mib = 57013
        adapter.snapshot(); adapter.quiesce()
        with self.assertRaisesRegex(HostMemoryError, r'^refused: 57,013 MiB RAM available, the load needs 57,014 MiB'):
            adapter.load(plan['models'][0], 131072)

    def test_load_timeout_cleans_scratch(self):
        plan, host, adapter = self.adapter()
        adapter.is_ready = lambda *_: False
        adapter.limits['load_seconds'] = 2
        adapter.snapshot(); adapter.quiesce()
        with self.assertRaisesRegex(InfrastructureError,'load exceeded'):
            adapter.load(plan['models'][0],131072)
        self.assertIsNone(host.server)
        self.assertFalse(adapter.owes_memory)

    def test_inactive_production_stays_inactive(self):
        plan, host, adapter = self.adapter()
        host.unit_active=False
        state=execute(plan,self.root/'runs','strata-test',adapter=adapter,lock_path=self.root/'lock')
        self.assertFalse(state['restoration_required'])
        self.assertFalse(host.unit_active)

    def test_cancel_during_load_and_generation_restores_and_fresh_adapter_resumes(self):
        for phase in ('load', 'generation'):
            with self.subTest(phase=phase):
                plan, host, adapter = self.adapter()
                root = self.root/phase
                target = 'is_ready' if phase == 'load' else 'request'
                with patch.object(adapter, target, side_effect=KeyboardInterrupt('cancel')):
                    state = execute(plan, root, 'strata-test', adapter=adapter, lock_path=self.root/'lock')
                self.assertEqual(state['status'], 'partial')
                self.assertFalse(state['restoration_required'])
                self.assertTrue(host.unit_active)
                self.assertIsNone(host.server)
                fresh = strata.StrataAdapter(plan, root/'logs', host, run_id='strata-test')
                resumed = execute(plan, root, 'strata-test', resume=True, adapter=fresh, lock_path=self.root/'lock')
                self.assertEqual(resumed['status'], 'complete')
                self.assertFalse(resumed['restoration_required'])

    def test_failed_restore_is_retained_as_stop(self):
        plan, host, adapter = self.adapter(refuse_probe=True)
        state = execute(plan, self.root/'runs', 'strata-test', adapter=adapter, lock_path=self.root/'lock')
        self.assertEqual(state['status'], 'restore_failed')
        self.assertTrue(state['restoration_required'])

    def test_missing_sleep_lock_refuses_before_production_changes(self):
        from llm_eval.inhibit import InhibitUnavailable
        plan, host, adapter = self.adapter()
        adapter.holds_machine = True
        with patch('llm_eval.runner.inhibit.sleep_inhibit', side_effect=InhibitUnavailable('missing')):
            with self.assertRaises(InhibitUnavailable):
                execute(plan, self.root/'runs', 'strata-test', adapter=adapter, lock_path=self.root/'lock')
        self.assertTrue(host.unit_active)
        self.assertFalse(host.stops)
        self.assertIsNone(host.server)

    def test_fake_backend_strata_plan_compiles_runs_and_resumes(self):
        self.manifest['backend'] = 'fake'
        plan = self.plan()
        first = execute(plan, self.root/'fake-runs', 'offline')
        resumed = execute(plan, self.root/'fake-runs', 'offline', resume=True)
        self.assertEqual(first['status'], 'complete')
        self.assertEqual(resumed['status'], 'complete')
        self.assertFalse(resumed['restoration_required'])


class StrataClassicContextTests(unittest.TestCase):
    """Phase 1 of the 2026-10-30 plan asked for the classic suite at the 262,144 target and was refused:
    `--context apply to workload suites only`. For a Strata manifest `--context` is what anchors the
    ladder, for every suite, so the refusal blocked a phase the harness can run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('engine', 'server.py', 'model.gguf', 'ple.gguf', 'mtp/rt', 'tok/vocab.json', 'pack/data.bin'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
        self.config = self.root / 'strata.json'
        self.config.write_text(json.dumps({'exe': 'engine', 'cwd': str(self.root), 'tokenizer': 'tok', 'args': [
            '--pack', 'pack', '--native', 'model.gguf', '--ple-gguf', 'ple.gguf', '--mtp', 'mtp/rt',
            '--max-context', '262144', '--kv', 'int8'], 'log': '/original.log'}))
        self.preset = self.root / 'models-preset.ini'
        self.preset.write_text(f"[tiny]\nmodel = {self.root / 'model.gguf'}\nctx-size = 131072\n")
        self.dir = self.root / 'acceptance'

    def prepare(self, **kwargs):
        return acceptance.prepare(self.dir, 'tiny', 'tiny', self.preset, ScriptedHost(), runtime='strata',
                                  strata_config=self.config, strata_python=sys.executable,
                                  strata_server=self.root / 'server.py', **kwargs)

    def test_a_context_override_anchors_the_classic_suite(self):
        card, code = self.prepare(suite='classic', context=262144)
        self.assertEqual(code, acceptance.OK, card)
        self.assertEqual(card['anchor'], 262144)
        self.assertEqual(card['context_ladder'], strata.ladder(262144))
        self.assertIn(131072, card['context_ladder'])  # the rungs below the anchor are still offered
        plan = json.loads((self.dir / 'plan.json').read_text())
        self.assertEqual(plan['models'][0]['anchor'], 262144)
        self.assertEqual(plan['models'][0]['contexts'], strata.ladder(262144))

    def test_without_the_override_the_classic_suite_anchors_at_the_default(self):
        card, code = self.prepare(suite='classic')
        self.assertEqual(code, acceptance.OK, card)
        self.assertEqual(card['anchor'], 131072)

    def test_workload_only_flags_are_still_refused_on_the_classic_suite(self):
        with self.assertRaisesRegex(ValueError, '--runs applies to workload suites only'):
            self.prepare(suite='classic', context=262144, runs=1)
        with self.assertRaisesRegex(ValueError, '--keep-going applies to workload suites only'):
            self.prepare(suite='classic', context=262144, keep_going=True)
