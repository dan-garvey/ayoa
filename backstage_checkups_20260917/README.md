# Periodic backstage checkups

Nine fresh Terra/max coding-agent calls, all first results retained. No direct API
calls, automatic editing, private reasoning inspection or source-biography changes.

- `covenant/`: a copy of the current playtest through passage 19; one checkup before
  message 20 and three following author passages. Interval 5. Messages 20–21 replay
  the original inputs; message 22 explicitly requests reading to the chapter's end.
- `breakwater/`: a fresh opening, then one checkup and response to an introduction
  asking what is urgent before a screening. Interval 2, to exercise another setting.
- `breakwater_verified/`: the exact same opening and player input replayed with
  the final checkup prompt. Only the checkup instruction changes. Two new calls;
  the copied opening is not counted twice.

The first Breakwater checkup incorrectly called the source dates contradictory.
The final prompt adds verification of statements and temporal references before
flagging a contradiction. The repeat correctly distinguishes booking entry on
Friday from room use on Monday. This is one targeted repeat, not a reliable
improvement estimate.

`TRANSCRIPTS.md` contains every new checkup and passage in full. Session attempts
retain exact JSON/text requests, metadata, start markers, final outputs and exposed
reasoning summaries. `validation.json` records exact reconstruction and projection
checks. `review.md` contains the human-readable assessment, including weaknesses.
The two runtime source files used for the model loop are in `runtime_source/`.

`run.py` and `refine.py` record the live invocation procedures; `audit.py` makes no
model calls. Each session's snapshot, rather than today's working sources, is the
authority for the prompts actually used. `protocol.json` pins inputs and starting
hashes. The Covenant transcript retains its original 19-passage prefix; copied
historical attempts are evidence, not new calls in this experiment.

`source_unchanged.json` verifies that the live user's original session was unchanged
through the initial trial. Afterward, `upgrade_live.py` explicitly converted all
seven local sessions to format 3, including the two nested foundation playtests.
The conversion was resumed from verified backups after its initial inventory and
validation assumptions needed correction. `live_upgrade.json` records all before
and after hashes: existing changes were limited to manifest, state metadata and
the author prompt's backstage output contract, plus the new checkup snapshot.
Every original passage, player input, response version and raw attempt remained
unchanged. Full local backups live outside the checkout at
`/home/dan/ayoa-evaluations/backstage_checkups_20260917/live_before`.

All 94 non-browser and 20 Chromium tests passed. The restarted chat served all
seven preserved sessions and retained the configured localhost/LAN host and
origin checks. New checkups are scheduled from existing ordinary turn counts;
none were generated retroactively for the live user's stories.
