# 2026-11-04: Chat transcripts were about to expire, so the owner had a devlog tool built

- Covers: 2026-11-04 16:44 to 18:08 UTC, sessions 885e1c2d

## What happened
- The owner asked three things at once: strip the co-author tag from local commits, whether to start the benchmark over after
  the day's settings changes, and how the chat discussions reach a devlog. The answers: only the Qwen3.6 screen was
  invalidated, so the data was kept; the Qwen3.6 re-run was queued, not started.
- The agent found Claude Code deletes raw transcripts after 30 days by default, and the October sessions were already gone.
  The owner had them copied into the repo, then kept out of GitHub (`352a0cd` ignores `transcripts/`), then made a local-only
  git repo, then condensed (14 MB of raw files became 412 KB of Markdown).
- The owner asked for a workflow behind a short command. It became a global `/devlog` skill and a `save-transcripts` script in
  dotfiles, usable in any repo. Each entry carries a `Covers:` line so a later run writes up only what is new.
- Asked whether the record captured how the owner worked, the agent listed the gaps: other tools, the owner's own view,
  process signals, hindsight, cross-project patterns. The owner accepted optional notes, Codex capture, process signals
  and a periodic `/reflect`, and declined Pi and Perplexity capture for now.
- Capture for this repo went from 6 to 38 condensed sessions (12 Codex, back to 2026-10-01).
- The owner raised running cost. The agent proposed a shorter "brief" view; the owner said the condensed files are not that
  big and left it as is.
- Owner notes became on by default, with Skip first so Enter bypasses (`0dd0439` in dotfiles).

## What the owner challenged
- "Is this new condensed version suitable for creating new dev diaries" and "am I capturing all the information needed ...
  Not just on what work was done, but how I went about it, challenges, and insights?"
- "It'd probably make sense to strip down the transcripts instead of taking them full sail."
- "This seems like it'll be costly to run given the amount of sessions I run on a daily basis." Measured, the condensed files
  were 680K tokens across 23 sessions and 14 active days (per session), which the owner judged acceptable.
- "Is this skill purely for llm-eval? I think it makes sense to have it work with any given repo."

## What the agent got wrong
- It said 178 Codex sessions were about llm-eval. That count included every session that merely mentioned the repo in loaded
  instructions; 12 actually worked in it. It corrected this the next turn.
- It first recommended Opus at high effort for `/devlog`, then Sonnet at high after sizing the reading cost.
- It had left two older commits carrying a co-author trailer, one already on GitHub. Those were later stripped before the
  2026-11-06 push (see that entry).

## Changed
- `352a0cd` llm-eval ignores `transcripts/`; `df966ce` Covers line on the 2026-11-04 entry.
- Dotfiles: `40db80f` skill and `save-transcripts`, `a59dcfc` window per session, `518158c` Codex capture and `/reflect`,
  `0dd0439` notes by default.
- `transcripts/` is its own local git repo with no remote, with a secret scan that redacts before commit.

## Learned
- Raw agent transcripts are perishable, and the repo's own records hold the numbers. The transcript supplies the story and the
  owner's challenges, so each claim in it is checked against the repo.
- A transcript cannot hold the owner's own view; the optional notes step is the only place that comes from.

## Data
- No benchmark data changed. Which screens stay valid after the settings changes is in [the settings entry](2026-11-04-like-for-like-settings.md).
