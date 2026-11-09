"""The warehouse schema. Additive migrations only: a new version appends to ``MIGRATIONS``."""
import sqlite3

TABLES = ("runs", "run_models", "load_attempts", "trials", "task_runs", "checks", "server_requests",
          "gpu_samples", "events", "sections", "judge_scores", "inference_requests")

MIGRATIONS = [("""
CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, run_dir TEXT, plan_hash TEXT, backend TEXT, suite_kind TEXT, suite_label TEXT,
  status TEXT, restoration_required INTEGER, restore_verified INTEGER, started_at REAL, ended_at REAL,
  wall_seconds REAL, harness_sha TEXT, harness_dirty INTEGER, opencode_version TEXT, llama_server_path TEXT,
  llama_server_version TEXT, gpu_name TEXT, gpu_driver TEXT, gpu_memory_mib REAL, kernel TEXT,
  valid_for_analysis INTEGER NOT NULL, invalid_reason TEXT, limits_json TEXT, environment_json TEXT,
  ingested_at REAL);
CREATE TABLE run_models (
  run_id TEXT, profile TEXT, covers_json TEXT, anchor INTEGER, contexts_json TEXT, model_path TEXT, quant TEXT,
  artifacts_json TEXT, settings_json TEXT, preset_sha256 TEXT, restore_model TEXT, mode_entry TEXT,
  mode_context INTEGER, mode_output_limit INTEGER, mode_compaction_point INTEGER, mode_entry_explicit INTEGER,
  mode_context_overridden INTEGER, selected_context INTEGER, verified_recall_context INTEGER,
  PRIMARY KEY (run_id, profile));
CREATE TABLE load_attempts (
  run_id TEXT, profile TEXT, context INTEGER, status TEXT, free_mib REAL, fixed_mib REAL, kv_mib REAL,
  load_seconds REAL, predicted_free_mib REAL, buffers_source TEXT);
CREATE TABLE trials (
  run_id TEXT, profile TEXT, section TEXT, fixture_id TEXT, fixture_kind TEXT, fixture_hash TEXT, context INTEGER,
  status TEXT, prompt_tokens INTEGER, output_tokens INTEGER, max_output_tokens INTEGER, finish_reason TEXT,
  fits INTEGER, request_seconds REAL, prompt_n INTEGER, cache_n INTEGER, prompt_ms REAL, predicted_n INTEGER,
  predicted_ms REAL, prompt_tps REAL, predicted_tps REAL, usage_json TEXT, grading_json TEXT, recorded_at REAL,
  PRIMARY KEY (run_id, section));
CREATE TABLE task_runs (
  run_id TEXT, profile TEXT, section TEXT, task_id TEXT, task_hash TEXT, category TEXT, group_name TEXT,
  run_n INTEGER, context INTEGER, status TEXT, passed INTEGER, failure_class TEXT, stop_reason TEXT, turns INTEGER,
  clarifications INTEGER, tests_pasted INTEGER, steps INTEGER, tool_calls INTEGER, tools_json TEXT,
  invalid_calls INTEGER, tool_errors INTEGER, compactions INTEGER, peak_context_tokens INTEGER,
  output_tokens INTEGER, reasoning_tokens INTEGER, model_seconds REAL, tool_seconds REAL, wall_seconds REAL,
  user_turns INTEGER, ended_with_question INTEGER, harness_restarts INTEGER, harness_lingered_turns INTEGER,
  web_hits INTEGER, web_misses INTEGER, answer_chars INTEGER, command_count INTEGER, export_path TEXT,
  export_sha256 TEXT, metrics_json TEXT, recorded_at REAL, PRIMARY KEY (run_id, section));
CREATE TABLE checks (
  run_id TEXT, section TEXT, idx INTEGER, type TEXT, passed INTEGER, detail_json TEXT,
  PRIMARY KEY (run_id, section, idx));
CREATE TABLE server_requests (
  run_id TEXT, log_file TEXT, load_context INTEGER, task INTEGER, slot_similarity REAL, prompt_n INTEGER,
  prompt_ms REAL, prompt_tps REAL, predicted_n INTEGER, predicted_ms REAL, predicted_tps REAL, total_ms REAL,
  total_tokens INTEGER, ctx_tokens_after INTEGER, cache_n_est INTEGER, graphs_reused INTEGER, truncated INTEGER,
  t_epoch_est REAL, PRIMARY KEY (run_id, log_file, task));
CREATE TABLE gpu_samples (
  run_id TEXT, t REAL, tag TEXT, mem_used_mib REAL, mem_free_mib REAL, util_gpu_pct REAL, util_mem_pct REAL,
  temp_c REAL, power_w REAL, clock_sm_mhz REAL, clock_mem_mhz REAL, error TEXT);
CREATE INDEX gpu_samples_run_tag ON gpu_samples (run_id, tag);
CREATE TABLE events (
  run_id TEXT, sequence INTEGER, time REAL, event TEXT, section TEXT, status TEXT, payload_json TEXT,
  PRIMARY KEY (run_id, sequence));
CREATE TABLE sections (
  run_id TEXT, key TEXT, status TEXT, time REAL, reason TEXT, PRIMARY KEY (run_id, key));
CREATE TABLE judge_scores (
  run_id TEXT, task_id TEXT, mode TEXT, criterion TEXT, value TEXT, overall TEXT, reason TEXT, judge_model TEXT,
  judge_base_url TEXT, against_run_id TEXT, rubric_sha TEXT, judged_at REAL);
CREATE INDEX task_runs_task ON task_runs (task_id, task_hash);
CREATE INDEX trials_fixture ON trials (fixture_id, fixture_hash);
""", None), ("""
ALTER TABLE task_runs ADD COLUMN task_spec_sha TEXT;
""", None), ("""
ALTER TABLE task_runs ADD COLUMN peak_step_output_tokens INTEGER;
ALTER TABLE task_runs ADD COLUMN length_finishes INTEGER;
ALTER TABLE task_runs ADD COLUMN target_context INTEGER;
ALTER TABLE task_runs ADD COLUMN compaction_point INTEGER;
""", None), ("""
ALTER TABLE task_runs ADD COLUMN task_identity_sha TEXT;
ALTER TABLE run_models ADD COLUMN workload_plan_json TEXT;
""", None), ("""
ALTER TABLE runs ADD COLUMN runtime_name TEXT;
ALTER TABLE runs ADD COLUMN runtime_json TEXT;
CREATE TABLE inference_requests (
  run_id TEXT, file TEXT, request_n INTEGER, turn_n INTEGER, status INTEGER,
  started REAL, ended REAL, ttft_seconds REAL, first_answer_seconds REAL, first_tool_seconds REAL,
  finish_reason TEXT, usage_json TEXT, timings_json TEXT, contract_json TEXT,
  PRIMARY KEY (run_id, file));
""", None), ("""
ALTER TABLE task_runs ADD COLUMN graders_sha TEXT;
ALTER TABLE runs ADD COLUMN harness_name TEXT;
ALTER TABLE runs ADD COLUMN harness_version TEXT;
ALTER TABLE run_models ADD COLUMN effort TEXT;
""", None)]

# Read-only conveniences. Scorecards only ever pool runs that are valid for analysis, within one entry, the
# context actually run at, output limit and workload group (core vs depth stay apart), and one configuration
# version (B2): harness code, harness client and its version, engine and its build, preset settings, model files
# (template included), thinking effort and grader code each split a row; a timeout is a failure
# shown in its own column. Model selection reads `acceptance screen`, not these views. A category pools several tasks, so `task_versions` is the number of distinct
# (task, hash) pairs behind it: more than its task count means a task changed mid-corpus and the rate
# mixes versions (see v_check_discrimination and task_runs.task_hash for the per-task view).
VIEWS = """
DROP VIEW IF EXISTS v_runs_valid;
CREATE VIEW v_runs_valid AS SELECT * FROM runs WHERE valid_for_analysis = 1;
DROP VIEW IF EXISTS v_scorecard;
CREATE VIEW v_scorecard AS
  SELECT m.profile, m.mode_entry, COALESCE(t.context, m.mode_context) AS mode_context, m.mode_output_limit, t.category,
         t.group_name, r.harness_sha, r.harness_name, r.harness_version, r.runtime_name, r.llama_server_version,
         r.runtime_json, m.settings_json, m.artifacts_json, m.effort, t.graders_sha, COUNT(*) AS runs, SUM(t.passed) AS passes, SUM(t.failure_class = 'timeout') AS timeouts,
         SUM(t.failure_class = 'incomplete') AS incomplete,
         COUNT(DISTINCT t.task_id || ':' || COALESCE(t.task_hash, '')) AS task_versions, COUNT(DISTINCT t.task_id) AS tasks,
         COUNT(DISTINCT t.task_id || ':' || COALESCE(t.task_hash, '')) > COUNT(DISTINCT t.task_id) AS mixed_task_versions,
         ROUND(1.0 * SUM(t.passed) / COUNT(*), 3) AS pass_rate, ROUND(AVG(t.steps), 1) AS avg_steps,
         MAX(t.peak_context_tokens) AS max_peak_context, SUM(t.compactions) AS compactions,
         ROUND(AVG(t.wall_seconds), 1) AS avg_wall_seconds
  FROM task_runs t JOIN v_runs_valid r USING (run_id) JOIN run_models m ON m.run_id = t.run_id AND m.profile = t.profile
  WHERE t.failure_class IS NULL OR t.failure_class IN ('model', 'timeout', 'incomplete')
  GROUP BY m.profile, m.mode_entry, COALESCE(t.context, m.mode_context), m.mode_output_limit, t.category, t.group_name,
           r.harness_sha, r.harness_name, r.harness_version, r.runtime_name, r.llama_server_version, r.runtime_json,
           m.settings_json, m.artifacts_json, m.effort, t.graders_sha;
DROP VIEW IF EXISTS v_recall_ladder;
CREATE VIEW v_recall_ladder AS
  SELECT t.profile, t.context, COUNT(*) AS trials, SUM(t.status = 'pass') AS passes
  FROM trials t JOIN v_runs_valid r USING (run_id) WHERE t.fixture_kind = 'recall'
  GROUP BY t.profile, t.context;
DROP VIEW IF EXISTS v_speed_by_load_context;
CREATE VIEW v_speed_by_load_context AS
  SELECT run_id, load_context, COUNT(*) AS requests, ROUND(AVG(prompt_tps), 1) AS avg_prefill_tps,
         ROUND(AVG(predicted_tps), 1) AS avg_decode_tps, MAX(ctx_tokens_after) AS max_ctx_tokens
  FROM server_requests WHERE predicted_n IS NOT NULL GROUP BY run_id, load_context;
DROP VIEW IF EXISTS v_gpu_by_tag;
CREATE VIEW v_gpu_by_tag AS
  SELECT run_id, tag, COUNT(*) AS samples, MIN(mem_free_mib) AS min_free_mib, MAX(mem_used_mib) AS max_used_mib,
         MAX(util_gpu_pct) AS max_util_pct, MAX(temp_c) AS max_temp_c, MAX(power_w) AS max_power_w
  FROM gpu_samples WHERE error IS NULL AND tag IS NOT NULL GROUP BY run_id, tag;
DROP VIEW IF EXISTS v_check_discrimination;
CREATE VIEW v_check_discrimination AS
  SELECT t.task_id, t.task_hash, c.type, c.idx, m.profile, COUNT(*) AS runs, SUM(c.passed) AS passes
  FROM checks c JOIN task_runs t ON t.run_id = c.run_id AND t.section = c.section
  JOIN v_runs_valid r ON r.run_id = t.run_id JOIN run_models m ON m.run_id = t.run_id AND m.profile = t.profile
  GROUP BY t.task_id, t.task_hash, c.type, c.idx, m.profile;
"""


def connect(path):
    """Open (creating or migrating) the warehouse."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    version = 0
    if "schema_meta" in have:
        row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
        version = int(row[0]) if row else 0
    for number, (script, _) in enumerate(MIGRATIONS, start=1):
        if number > version:
            conn.executescript(script)
            conn.execute("INSERT OR REPLACE INTO schema_meta VALUES ('schema_version', ?)", (str(number),))
    conn.executescript(VIEWS)
    conn.commit()
    return conn
