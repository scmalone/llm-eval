# 2026-11-01: The accuracy suite's tasks: mined fixes, game, vision and documents for every model

## What happened

The owner asked for the accuracy suite's missing parts to be built and red-teamed: 20 fixes from their own repos,
the Blender and flight-test tasks, a vision set and long documents every model can read.

- 20 mined fixes (`fixtures/workload/mined/`) over eight repos, at most four per repo, Strata two: Node, Python
  (pytest and unittest), bash suites and one PASS/FAIL script. Packets are hand-written in `suites/mined-specs.json`;
  `tools/build_mined_tasks.py` takes the rest from git and refuses a packet that quotes the fix.
- Game (`fixtures/workload/game/`): three flight-test vitest fixes (ground field, humerus hinge, wall bands) and three
  authored Blender tasks over a new `blender-kit` repo in `<data>/sources/` (rock cluster, wing rig, a 2.79 script to
  port), graded by a headless inspector.
- Vision (`fixtures/workload/vision/`): ten synthetic 1024 x 768 images, six OCR and four grounding tasks; a new `box`
  check scores overlap in pixels or the 0 to 1000 grid; profiles without image input skip them.
- Documents: suites `documents` and `accuracy` size the six document tasks with each profile's own tokenizer
  (`llama-tokenize` on the GGUF vocabulary, about 4 s for all six).
- Environments: two venvs, two `node_modules` and Blender under `<data>/envs/`, mounted read-only into both sandboxes,
  with scratch tmpfs folders where a tool writes a cache (vite).

## Back and forth

- Proofs in the real sandbox found what unit tests could not: vite writes into `node_modules` (read-only), a
  `pytest.ini` turned `-v` off, unittest's `ok` can land lines after the test name, a consult test lived in a third
  script, and a failing first attempt showed up as "reference diff did not apply" (the follow-up turn re-applied it).
- job-tracker's store-locks fix was flaky at its own answer commit (two of four runs failed); replaced with the
  usage-dashboard keychain fix. Every answer state then ran three times.
- The top 3 are not known before testing: the screen ranks every profile first (now the check suite's tasks).

## Red team and fixes

A blind Codex red team (gpt-6.1-sol, high) returned not ready with five findings; the owner accepted option A.

- RT-001: editable source could mark every unittest test skipped and still print `OK (skipped=28)` with exit 0.
  unittest and node tasks now fail on any skip (`skipped` in the tests check); the attack is replayed through
  `run_task` in the real sandbox and fails.
- RT-002: mounted environments were unpinned. Each `ENV.json` and task `env` entry now carries a content digest,
  re-hashed at prepare and before every task-run (`llm_eval/workload/envs.py`).
- RT-003: evidence-wipe's test never exercised the dialog half of its fix. Replaced with job-tracker's batch
  promotion contract (`mined-job-tracker-batch-promote`, 8 tests that cover the whole packet).
- RT-004: ground-field's whole-route test skipped without the gitignored heightfield. A `files` env mount now
  binds that one asset read-only, and the route test is a required, passing test.
- RT-005: an OCR answer could list several candidates. Each OCR task's field must hold one value (`one_value`).
- Documents: per-profile counts are relabeled `profile_tokenizer_raw_user_text` (Strata's stay templated).

## Learned

- A mined task is only as stable as its own tests at the fix commit; run them repeatedly before trusting a grade.
- A read-only environment needs to declare where its tools write.

## Sources

- Commits: llm-eval `f7626bc` and the docs commit after it; blender-kit `ac9bde4`, `678f473`; vision-set `ab5bea8`.
