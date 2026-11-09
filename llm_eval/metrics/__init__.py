"""The metrics corpus: a derived, rebuildable SQLite warehouse over the run folders.

Run folders stay the source of truth (append-only, plan-hashed). Everything here is read from
them, so an ingest bug never loses data: fix it and ingest again. Nothing in this package
touches production, the GPU, systemd or the network.
"""
