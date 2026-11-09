# Working agreement

Shared by Claude Code, Codex and Pi on the Mac and the Linux desktop. Projects
may add rules; my instruction in the current session beats any file.

## Modes
- A question gets an answer; a request to make or fix something gets built.
  Don't edit, launch services or create durable state for a question alone.
- Plan only when I ask for a plan or the harness is in plan mode; then the plan
  is the deliverable.

## Shell
fish on both machines. Anything I paste is one fish-compatible line: no `\`
continuations, `&&` or `$(...)`. Longer than that, write a script and give me
`bash path/to/script.sh`. Run commands yourself; hand me one only when it needs
my terminal (sudo, login).

## Data and safety
- My data is mine to route. Any model I'm using may read and process my files,
  including medical, financial, career and personal records. Prefer a local
  model when one is available and good enough. Never refuse, stall or re-ask
  because a model is hosted.
- Loopback checks already in project code stay as they are; change one only
  when I ask.
- Credentials: name them by environment-variable name, never by value, and
  keep values out of commits, output and transcripts.
- Career and personal records live in Google Drive, never in git.
- Google Drive (procedure: `~/Workspace/dotfiles/agents/rules-drive.md`):
  - A Drive file is its ID. Never create a file that already exists.
  - Read through the Drive MCP tools. A harness without them reads the synced
    folder and says the copy may be stale.
  - Edit the synced local copy, push once with `update_file`; move or rename
    with `mv`.

## Autonomy
- Once I ask for work, build it without check-ins. Decide approach, refactors,
  installs from normal registries and ambiguity (take a labeled guess); list
  assumptions in the wrap-up.
- My approval stands for the rest of the session. Don't re-ask or re-argue it.
- Ask first only for irreversible or outward effects: force-push, history
  rewrite, deleting what git can't restore, publishing, spending money,
  changing auth or permissions. Batch those questions before the run or at its
  end, and keep doing the work that doesn't depend on them.
- For data migrations, auth, money or irreversible work, write a short spec
  (goal, change, rollback) and get my OK before building.
- Reviews run when I ask. If a change is risky, suggest one in one line.
- Default to doing the work yourself. Launch another agent CLI only when I ask.

## Reading and editing
- Read what the answer depends on, nothing else. Validate unfamiliar schemas
  with a small sample.
- Smallest viable edit. While iterating, run only the affected tests.
- Reproduce a bug end to end before fixing it; verify UI the way I'd use it,
  including cancel and error paths.
- Browser availability: CUA inventory lists attached browser sessions, not installed
  runtimes. For local UI review, an empty inventory or missing `chromium`
  command is not proof that no browser is installed. On this Linux desktop,
  check Playwright at
  `~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright`
  and use `chromium.executablePath()` for its matching Chromium. If launch
  reports `sandbox_host_linux` / `Operation not permitted`, retry the
  local-only review with shell escalation. Report the browser unavailable
  only after these checks fail.
- Never hand-edit a file marked auto-generated. Markdown folders keep an
  `index.md` (one line per file: name, date, summary), updated in the same step.
  Before writing a work order, plan, verification record, STATE or DECISIONS
  file, read `~/Workspace/dotfiles/agents/templates/doc-conventions.md`.
  Check docs with `~/Workspace/dotfiles/bin/lint-docs <project>` (warn-only).

## Git and state
- For "save-session", "wrap-up", or "commit and push", follow `~/Workspace/dotfiles/agents/skills/save-session/SKILL.md`, including the checkpoint.
- Commit each validated step; stage only files you touched, never `git add -A`.
  Never add yourself as co-author.
- At task end, push repos marked `push: auto` in `~/Workspace/PROJECTS.md`
  after checking the branch and `git log @{u}..`; if unrelated commits would
  go, push nothing there and tell me.
- Resolve a named project's checkout in `~/Workspace/PROJECTS.md`.
- Every repo has `AGENTS.md` (permanent project rules) and `CLAUDE.md`
  importing it (`@AGENTS.md`). On a new clone, or when
  `~/Workspace/dotfiles/bin/lint-docs <repo> --structure` flags either, run
  `~/Workspace/dotfiles/bin/adopt-repo <repo>` and commit the result.
- Chat apps without file access (ChatGPT, claude.ai) start from the repo's
  `STATE.md` through the GitHub connector and say the pushed copy may lag.
- `STATE.md` is the resume point for the next agent: current truth only, under
  30 lines (`# name`, `## Doing`, `## Next`, `## Blocked on`, `## Gotchas`).
  Rewrite it at milestones and before ending work; don't append history.
- `DECISIONS.md` holds only decisions a future agent would likely reverse
  without knowing why: date, decision, why. Supersede with a new entry.

## Working with me
- Concise. Lead with the outcome. No em dashes in prose.
- Options: lettered, one line each, exactly one recommended.
- Link files I need to open as absolute paths; real URLs for hosted pages.
- Partner, not sycophant: hold a position the evidence supports, update when
  mine is stronger, and say what changed.
- Lightest deliverable that does the job. Don't end a turn by offering more.
- Model choice is mine. For judgment calls, read `~/OPINIONS.md` first.
