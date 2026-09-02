# Live Premium-Lobby Playtest Through Floor 2

- Date: 2026-09-01 CDT / 2026-09-02 UTC
- Session: `codex_premium_lobby_floor2_live_20260902T040622Z`
- Story: `one_star_ascension_s1`
- Player binding: Master only
- Runtime work began at: `6f071fc`
- Last behavioral fix used in the session: `cd2bcc5`
- Stop condition: Floor 2 cleared and the run returned to a stable management choice
- Final checkpoint: `ckpt_0010.json`

## Outcome

The requested path completed. Floor 1 opened directly on the goblin ambush and
cleared after one combat `/defer`. The Master then bought one 20-Gem pack and
performed five premium summons. Renna Holt and Mirelle Voss were deployed to
Floor 2 while all five new Heroes remained in Niflheim. On the next mission
`/defer`, Liora Fen privately began ward practice in the lobby before the exact
owed Renna floor turn resumed. Floor 2 cleared in that same command.

The final checkpoint is stable: there is no active mission or pending
operation; Floor 2 is cleared and Floor 3 is unlocked; the account has 49 Gold,
0 Gems, 5 Building Resources, $100 discretionary funds, and 3/5 stamina. Renna
and Mirelle are level 3 and back in Niflheim. Edren Marr remains permanently
culled. The five premium-pull Heroes are alive in the lobby, and Liora's private
training commitment remains open.

This was an implementation playtest rather than one pristine run from a single
revision. It deliberately continued from intact checkpoints while five defects
found during play were fixed. The final code has a full offline pass, and the
inline-deployment liveness case that was discovered after Floor 2 selection has
a focused regression. A clean end-to-end replay from final HEAD was not run.

## Exact Master sequence

1. `/begin` — the first attempt rejected bad live-observer visual metadata;
   after the adapter fix, retry opened directly in Floor 1.
2. `/defer` — completed Floor 1.
3. `Buy one 20-Gem pack, then spend 25 Gems on five premium summons.` — the
   first two attempts rolled back at 2,000- and 8,000-token character-authoring
   caps; after increasing the Luna authoring headroom, retry committed.
4. `/defer` — before the lobby-defer routing fix, this settled without giving a
   Hero an opportunity.
5. `Watch the five new arrivals in the lobby.` — produced only a gate cue.
6. `/defer` — after the routing fix, Renna and Mirelle discussed Edren's death
   and bandaged Mirelle's arm.
7. `Deploy Renna Holt and Mirelle Voss to Floor 2.` — selected and deployed the
   pair, leaving all five premium summons in the lobby; the mission then made
   autonomous progress.
8. `/defer` — after the preserved-frontier fix, ran Liora's private lobby
   activity, resumed Renna on Floor 2, and completed the floor.
9. `/defer` — produced Iselle's stable post-clear management prompt.

The malformed setup-only `/join the Master` probe is retained in the raw logs
but is not part of the playtest sequence.

## Implemented runtime changes validated here

### One mission-result path

The separate One-Star mission-report projector, report schema, report delivery,
and report-specific tests were removed. Mission end remains deterministic for
state, rewards, deaths, progression, and exact System facts, but the ordinary
narrator now renders the same canonical event stream used everywhere else.

Both live clears produced ordinary VN decks. Floor 2 no longer raised
`OneStarMissionReportError`, no result was lost after checkpoint commit, and no
source-shaped location id appeared in any of the 59 presented card texts.

### Lobby liveness without a second fiction system

One-Star now contributes a bounded private lobby affordance through the existing
router contract. Eligible nonparty Heroes remain ordinary CharacterAgents; the
router decides what they do, canonical history persists it, and the active
mission receives none of the private fact. The turn loop preserves the exact
foreground frontier, runs at most one due lobby opportunity, and then resumes
that frontier without spending the foreground cascade cap.

The live evidence is the exact sequence:

1. `event_0030_liora_training_camp_affordance` privately exposed the built
   Training Camp practice lane to Liora.
2. Liora was called in `background` frame and authored solo ward practice.
3. `event_0031_liora_begins_conductor_ward_practice` committed that action and
   opened `commit_event_0031_liora_begins_conductor_ward_practice_liora_fen`.
4. Renna was then called in `foreground` frame at the previously owed Floor 2
   frontier.
5. The floor continued from the lantern ascent and cleared normally.

The final follow-on fix also admits this liveness opportunity when deployment
selection opens and resolves its reversible Cat II operation inline. That exact
selection-to-background-to-floor sequence is covered offline; the live
selection in this session occurred just before that fix and exposed the bug.

### Agent configuration and summon authoring

All CharacterAgent tiers and the CharacterManager used `gpt-5.6-luna` with max
reasoning. The successful five-pull authoring calls used 14,950 and 16,803
output tokens, confirming that the old 8,000-token ceiling was not viable for a
batched premium cast. Casting-plan and character-authoring requests now have
32,000-token output headroom.

## Behavioral findings

### Passed

- The first-floor tutorial is the attack itself. There was no lobby briefing,
  gate negotiation, mission acceptance, or opening-refusal scaffold.
- Edren's response was literal terror: `Goblins—I can’t fight them!` He fled and
  was cut down from behind. The canonical stream included the exact fact
  `System: Edren Marr died.`, and Iselle later said he was dead and permanent.
- Renna and Mirelle acted through Floor 1 without combat chatter. Floor 2 speech
  stayed short and attached to rescue, movement, or the beacon task.
- No presented Hero or narrator card used the forbidden standalone word
  `line`. One internal router rationale used `spear line`; it never reached a
  Hero intention or player-facing deck.
- The normal narrator successfully compressed both clear sequences while
  retaining rewards, level gains, return, and unlock state.
- The five premium pulls resolved atomically as Odelia Marr (3-star), Perrin
  Vale (2-star), Wren (3-star), Liora Fen (2-star), and Rowan Kest (2-star).
- Before Floor 2 selection, a bare `/defer` produced real Hero-to-Hero lobby
  material: Renna named Edren's death, Mirelle accepted help, and Renna held the
  bandage. Iselle was not required to monopolize the turn.
- While Floor 2 was active, Liora independently used an available facility.
  Her private action did not enter the Master's mission deck or influence the
  deployed party.
- After Floor 2, another `/defer` neither looped nor soft-locked. Iselle offered
  a concrete next management choice.

### Remaining limitations and observations

- **Model-call cadence remains slow.** Floor 2 selection/progress used seven
  Hero calls, eleven router calls, one narrator call, and four image-director
  calls over about four minutes. The final mission `/defer` used five Hero
  calls, nine router calls, one narrator call, and ten image-director calls over
  about three minutes. Player-facing narration was compact, but the underlying
  autonomous cascade is still expensive and granular.
- **Router correction remains routine.** Live corrections included forbidden
  generic lifecycle/location drafts and a deployment resolve missing its paired
  `mission_start`. They recovered, but they still add latency and fragility.
- **Two generated summon panels lack portraits in this artifact.** A stale
  prewarm lookup read retired `ImageGenerationRequest.character_id`; the reader
  was fixed in `d207369`, but the already-rendered Odelia and Perrin panels were
  not regenerated.
- **Visual staging remains opportunistic.** Several stage requests exceeded the
  presentation budget and used neutral fallback stages. One Floor 2 clear image
  request received a nonfatal capacity rejection.
- **Generic world flags remain stale.** The authoritative One-Star account state
  correctly records Floor 2 cleared and Floor 3 unlocked, but the final
  checkpoint's generic `world_state.global_flags` still says
  `phase=floor_one_goblin_ambush`. No observed runtime consumer used that stale
  flag during this path, but the duplicate state deserves removal or
  reconciliation rather than another compatibility path.

## Presentation artifacts

Eight source decks were copied byte-for-byte with their content-addressed ids
and manifests under `artifacts/presentation/`:

1. `cd8133172ef8a8596bce8ac3af10d3ce4224638fca110a39e1d0b308ed7c99f5`
2. `03730fc20971e6f9453167d24a2c8848b513dc5d33d4f8794414b31c770c6fd3`
3. `a2d839bc5903356e6f6305972d6b862ca7d420a730b10151a25aecc191bfd157`
4. `caaabee4a9eccb45a5f62f8fd43da2df2be3e9c94b9ddff109d30b42c4109c68`
5. `19adeef20affa399cc9e65648b704789c37499f98b83329f777617cdddf33f65`
6. `426bd91efa9f22971afe5b224c05dfeebd4596275c5099506bccb1859abb86cc`
7. `c47b6de702adbc02f3791f75c849cf54c762aa08c4df22c5d93f63b293373208`
8. `cee7aa112fd9c9b61be137fdad2238607999733ee0276a173e04c6cd6f321148`

`artifacts/vn-slideshow/` is a hash-validated 59-card chronological export:

- cards 1-13: opening and Floor 1 combat
- cards 14-20: Floor 1 clear and Iselle debrief
- cards 21-35: five premium pulls and survival induction
- cards 36-37: the unsuccessful quiet-watch probe
- cards 38-45: Renna and Mirelle's preselection lobby exchange
- cards 46-53: Floor 2 deployment and first autonomous progress
- cards 54-56: Floor 2 clear
- cards 57-59: final Iselle management prompt

Liora's ward practice is intentionally absent from the Master-view slideshow;
its private canonical event, character observation, and open commitment are in
`ckpt_0009.json` and `ckpt_0010.json`.

## Verification

- Every copied deck card matched its manifest SHA-256.
- The flat slideshow contains all 59 cards and its `index.json` records each
  original source path and digest.
- Presented card text contains no raw `niflheim_lobby`, report exception,
  source-shaped identifier, or forbidden standalone `line`.
- Full offline suite: `2195 passed, 2 skipped` in 122.02 seconds.

## Raw evidence

- `ckpt_0000.json` through `ckpt_0010.json`
- every `playtest_turn_*.log`, including the transactional failure probes
- `PLAYTEST_REPORT.md`
- `artifacts/presentation/`
- `artifacts/vn-slideshow/index.json`
