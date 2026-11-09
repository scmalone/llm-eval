# Spec and proposed plan: longitudinal scoring and task coverage

Date: 2026-10-05
Status: Scoring item 4 implemented and validated 2026-10-17; expansion and reports remain pending.
Resume: Complete the scoring step below. No evaluation run is authorized here.

## Purpose and scope

Measure whether a local model or configuration gets more reliable over time on
the owner's desktop workload, and whether improvements transfer to unfamiliar
work. Use interpretable outcomes and uncertainty rather than A-F grades or an
arbitrary composite score. Keep speed and resource use separate from quality.

The agreed direction has five items:
1. Preserve the existing 35 core tasks as the longitudinal baseline.
2. Repeat full-suite runs to measure consistency and compare configurations.
3. Add targeted task diversity relevant to actual work.
4. Report success rates, baseline changes, and appropriate uncertainty.
5. Keep baseline and expanded-suite results separately comparable.

Items 1 and 2 require no implementation in this workstream. Existing run
permissions still apply; this document does not change the locked screen/top-3
execution plan or authorize automatic repeated campaigns.

## Starting evidence

The suite has 35 core tasks across seven equally sized categories, plus four
depth and five diagnostic tasks. It has broad workflow coverage, but 24 core
tasks use llm-eval snapshots. All core implementation and review tasks come
from that project. Implementation prompts are generally detailed; all five
review tasks contain expected defects. Repeated attempts add reliability
evidence, not new task families.

Existing building blocks: `llm_eval/metrics/stats.py` provides Wilson intervals
and a paired bootstrap over per-task scores. `metrics/schema.py` records task
versions, run validity, environment, configuration, outcomes, and timing. Its
scorecard currently pools attempts; that is insufficient for equal-task
weighting when repeat counts differ. Reuse these components where suitable.

## Item 3: targeted diversity

Proposed first expansion: eight tasks, separate from core.

| Family | Tasks | New evidence |
|---|---:|---|
| Unfamiliar project | 2 | Implementation and lookup in a small synthetic Python utility with different structure and conventions |
| Diagnosis-led implementation | 2 | Reproduce a user-visible failure, identify the cause, repair it, and demonstrate recovery from a terse request |
| Review specificity | 2 | One clean module and one defective counterpart; measure false alarms as well as defect detection |
| Operations diagnosis | 2 | The same visible symptom caused by different faults, requiring evidence before repair |

Prefer relevant Python/Linux work over adding languages solely for breadth.
Use synthetic fixtures only, with no live-home or external-repository dependency.
Record source project, task family, skill, request shape, and expected outcome.
Renaming files or changing seeds does not make a new family.

Acceptance: each task has objective checks and a demonstrated reference outcome;
an incomplete or incorrect response fails. For clean review, the correct
no-change/no-defect response must pass, and a seeded false accusation must fail.
Test forbidden edits and shortcuts. Verify every task through the real sandbox
integration path without GPU inference. Keep sibling variants grouped in
analysis and disclose shared-source concentration.

## Item 4: scoring and uncertainty

Primary quality measure: compute each task's success fraction over eligible
attempts, then average those fractions equally across the declared suite.
Show percentage, passed/attempted counts, task count, repeat distribution,
category breakdown, and coverage. A heavily repeated task gets no extra weight.
One decimal place is display precision, not a claim about measurement precision.

For comparisons, explicitly select a baseline and use matching task identities,
fixture/grader versions, and a declared configuration cohort. Show candidate
minus baseline in percentage points and the task-level changes. A changed
model/configuration is the comparison target; unintended harness, hardware,
budget, or grader differences are visible confounders, not silent pooling.

Distinguish two uncertainty questions:
- Fixed-suite repeatability: uncertainty from stochastic attempts on these exact
  tasks. Assess within-task repeated outcomes; preserve shared campaign blocks
  where failures or conditions couple attempts.
- Transfer to other tasks: variation across tasks/families. A paired task or
  family bootstrap can support a sensitivity analysis, but this authored suite
  is not a random sample of all future work. Label that limitation explicitly.

Do not use a single Wilson interval over all pooled attempts as though repeats
were independent new tasks. Proposed nominal level is 95%; finalize the
estimator and pairing assumptions before coding. Sparse data, all-identical
outcomes, or too few independent families must not produce a misleading claim
of certainty. Show insufficient evidence where appropriate. An interval crossing
zero means the direction is unresolved, not that configurations are equivalent.

Keep infrastructure/capacity failures and missing tasks visible outside the
model-quality denominator; count model failures and task timeouts as failures.
An incomplete suite cannot receive a complete-suite headline. Report clean-review
false alarms separately. Timing, output caps, tool errors, and resource use are
secondary diagnostics; do not blend them into the quality percentage.

Acceptance: synthetic analysis cases cover unequal repeats, paired improvement
and regression, unchanged outcomes, missing tasks, invalid runs, mixed versions,
shared families, and degenerate intervals. Results are reproducible from raw
records, with method/version and resampling seed recorded. Validate interval
behavior using simulated known probabilities, not just expected arithmetic.

## Item 5: reporting and versioning

Present fixed core, expansion, depth, and diagnostics separately. Each panel
shows its suite manifest/version, task coverage, eligible attempts, configuration,
and uncertainty scope. Do not turn a harder expansion into an apparent core
regression. Do not pool the seven-task screen with complete-core results.

Retain immutable historical observations. Changed task or grader versions start
a new comparison series; an explicitly labelled common-task comparison may
bridge versions. Save machine-readable scores and a local self-contained report
using the existing reporting path. No new dashboard service is needed.

Acceptance: adding expansion results leaves core results unchanged; incompatible
versions cannot silently pool; incomplete panels remain visibly incomplete;
reports expose baseline identity and percentage-point deltas. Verify rendered
report behavior with representative synthetic complete and incomplete datasets.

## Decisions for acceptance

1. Recommended expansion scope: the eight tasks above, all synthetic and relevant
   to Python/Linux work. Owner acceptance pending.
2. Recommended scoring: equal-task success rate, explicit paired baseline delta,
   95% uncertainty with repeatability and transfer distinguished. Exact estimator
   remains a methodology decision before implementation.
3. Recommended delivery order: scoring/data contract, expansion fixtures, then
   separate versioned reports. Existing baseline tasks and live-run controls stay
   in place. Owner acceptance pending.

## Resumable implementation plan

- [x] Owner authorized scoring first and chose A, conservative campaign bounds,
  2026-10-05. Method and cohort contract below finalized before implementation.
- [x] Implement item 4 in the existing metrics layer, with focused synthetic
  validation and simulation of interval behavior; preserve raw-record replay.
  Completed 2026-10-17; 33 focused scoring tests, 346 unit tests, and all eight
  integration tests passed.
- [ ] Implement item 3's accepted task families and metadata; prove correct,
  incorrect, and forbidden-action outcomes through sandbox integration.
- [ ] Implement item 5's manifests, comparison export, and separate report panels;
  verify version boundaries, incomplete data, and rendered report behavior.
- [x] Run directly relevant regression/integration checks. The owner explicitly
  waived another redteam for these repairs; the review recommendation is tracked
  separately. Live evaluation requires separate authority.

Current status: owner authorized repairs for all four findings and explicitly
waived another redteam. Scoring item 4 is implemented and validated; items 3 and
5 remain pending acceptance. No live evaluation was performed.
Checkpoint authority: `../STATE.md`. Keep detailed progress here and reference it
from STATE rather than duplicating this plan.

## Accepted scoring method, 2026-10-05

Use equal-task means and two-sided weighted Hoeffding bounds at 95%. For T tasks,
an eligible attempt on task t has weight 1/(T*n_t). Sum those weights within each
declared campaign block to obtain range width r_b. The half-width is
sqrt(log(2/alpha) * sum(r_b^2) / 2). For candidate-minus-baseline, combine both
arms in each shared block before squaring; negative baseline contributions still
add to range width. This permits arbitrary dependence inside blocks, including
between tasks and arms. Independent blocks and a fixed outcome-independent
sampling plan are required. Repeating attempts inside a block cannot manufacture
independent evidence. Clip score bounds to [0,1] and delta bounds to [-1,1].

This follows the bounded independent-sum result in [John Duchi's Stanford
notes](https://cs229.stanford.edu/extra-notes/hoeffding.pdf), Lemma 5 and its
independent-sum derivation, applied to each campaign's bounded weighted
contribution. It is conservative and does not use observed
zero variance to assert certainty. The legacy [Wilson
helper](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm) remains for
individual binomial summaries; it is not the pooled suite interval.

The input contract pins suite group and task spec/snapshot hashes, explicit
run/profile selections, campaign block mapping, and declared differences between
comparison cohorts. The engine reads existing warehouse records without schema
mutation. Mixed task/grader/config versions fail closed. Missing provenance,
dirty harnesses, incomplete suites, filtered attempts, or unconfirmed independent
fixed-design blocks withhold inferential bounds while retaining visible counts
and descriptive outcomes where versions are known. At least two blocks per task
and arm are required by policy. Cohort comparisons pair task identities, not
invented seed pairs. Family/source concentration is disclosed; transfer intervals
are unavailable because population sampling assumptions have not been established.

Bounds cover fixed-design repeatability on the specified tasks under stable
conditions, not future-work generalization. They are pointwise for a preselected
comparison, not simultaneous guarantees over repeatedly inspected comparisons.
Adaptive stopping, selecting the best result, or outcome-dependent repeats
invalidate this contract. No bootstrap or resampling seed is used by this method.

## Scoring validation progress

- 333 unit tests pass, including 20 focused scoring tests and a real subprocess
  CLI replay from synthetic raw run folders through warehouse ingestion.
- Fixed-seed simulations, 1,500 draws per case: true probability covered in
  1,500/1,500 at p=.02, 1,485/1,500 at p=.5, and 1,500/1,500 at p=.98, with
  perfectly dependent task outcomes inside independent campaigns. Paired,
  anticorrelated arms with unequal repeats: 1,494/1,500. These demonstrate tested
  behavior, not a proof of coverage for every possible data-generating process.
- Integration first failed on sandbox network-namespace permissions. The first
  bypass approval could not run because automatic review hit a usage limit;
  owner said continue, and the retry was approved. All eight integration tests
  pass in 123.126 seconds; source stamp current.
- Blind committed-code review: Codex `gpt-6.1-sol`, high, same-family reduced
  independence; verdict not ready. All four findings verified with synthetic
  probes; owner subsequently authorized fixing all four:
  - LS-001 (High): producer uses category groups for core and `chat_sweep` for
    diagnostics; scorer's exact group match excludes their real records.
  - LS-002 (High): expected results missing from individual selected campaigns
    can disappear while pooled suite coverage and bounds remain complete.
  - LS-003 (Medium): empty-array or scalar JSON provenance can permit bounds.
  - LS-004 (Medium): legacy task hash includes `runs`, so changing repeat counts
    is falsely rejected as a task-version change despite unchanged spec/snapshot.
- Independent review checked the original committed scoring diff and synthetic
  evidence only; no real runs, records, models, or services. Owner-authorized
  repairs followed; no second review, live evaluation, or remote mutation.

## Repair progress, 2026-10-05

- LS-001: normalize producer categories to core and `chat_sweep` to diagnostic.
  Actual fake runner output now passes through saved-run ingestion and the real
  scoring CLI; diagnostic and depth flags also have ingestion/CLI coverage.
- LS-002: ingestion records each profile's planned tasks and repeat counts in
  `workload_plan_json`. Every selection exposes coverage, missing task/repeat
  rows, and profile-level failures. Missing or unknown planned coverage cannot
  support a complete headline or bounds. Partial plans may combine only when
  each recorded plan's required attempts are present. Explicit counts cannot
  contradict saved plans.
- LS-003: validate producer JSON shapes and necessary contents for artifacts,
  model settings, limits, and planned tasks. Malformed types/entries are rejected;
  missing or empty provenance withholds bounds.
- LS-004: additive `task_identity_sha` is a versioned digest of normalized spec
  and snapshot contents. `runs` and transient `_path` do not change this identity;
  changed spec or snapshot does. Original `task_hash` remains intact. Request v2
  uses the new identity. Re-ingestion from saved plans backfills metadata without
  rerunning models; request v1 retains legacy descriptive support and explicitly
  withholds inference. Scoring itself remains read-only, including old warehouses.
- Four regression cases first reproduced the original failures. All 33 focused
  scoring tests now pass, including unequal repeats across actual ingested runs,
  additive migration, old read-only warehouse handling, and producer-shaped CLI
  replay. Final validation completed 2026-10-17: all 346 unit tests and all eight
  integration tests passed (116.734 seconds). The restricted integration attempt
  failed during bwrap namespace setup; the approved rerun passed. No GPU or live
  evaluation was performed.
