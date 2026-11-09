# 2026-11-02: The SIGINT drill killed by systemd-oomd, and a RAM gate for Strata loads

## What happened

The owner started `plan_ctl launch -- --only D` at 2026-11-01 21:48. After the integration tier, the SIGINT drill
quiesced the router at 22:03 and started Strata. The engine reads the whole `--native` GGUF (52,278 MiB) into RAM,
but only 47,400 MiB was available (Steam, opened after the afternoon runs, plus Brave, Slack and agent terminals).
The load stalled under memory pressure. systemd-oomd killed the owner's Ghostty tab (Claude and Codex sessions)
at 22:05:41, the Strata load at 22:05:45, and the harness's retry at 22:07:17. Production came back healthy and
the drill was recorded failed ("the campaign ended before the signal"). Every afternoon load began with 57,014 MiB
or more available and finished in about 27 s.

## Changed

- `StrataAdapter.ram_gate`: after quiesce and before the scratch server starts, refuse when `MemAvailable` is below
  the `--native` file size plus 2,048 MiB (54,326 MiB for IQ3_S). It raises `HostMemoryError`, an
  `InfrastructureError` that the runner does not retry, so the run tears down and restores production as for any
  infrastructure failure. The failure section's reason begins `refused:` and names the numbers.
- `tools/run_test_plan.py`: a phase whose failure reason is a `refused:` stops with `refused` and no re-prepare;
  a drill that ends before its signal now logs the campaign's failure reason instead of "see its sections".

## Learned

- systemd-oomd acts on memory pressure, not exhaustion: 61 GiB of zram swap is no headroom, and a slow load
  under pressure is killed even when swap is mostly free.
