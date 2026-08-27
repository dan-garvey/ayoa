# One-Star VN runtime-sprite playtest

Date: 2026-08-27 (America/Chicago)

Implementation branch: `codex/vn-runtime-sprites`

Primary session: `one-star-vn-sprites-luna-playtest-post-matte-osa-20260827`

Acceptance session: `one-star-vn-sprites-luna-playtest-acceptance-20260827`

## Outcome

The runtime sprite approach is viable. The production compositor now places
zero, one, or two immutable transparent sprites over one immutable reviewed or
generated environment, then draws the ADV interface. The live One-Star replay
showed Iselle and stable masculine/feminine veiled one-star recruits, including
neutral and tense pose-expression changes, inward-facing two-character layouts,
centered solo layouts, and exact background reuse. Character sprites were not
painted into the background and no runtime model inspected an image.

The playtest also found four failures before acceptance: visual bindings were
lost during One-Star transaction copying, seeded Heroes were incorrectly queued
for generated prewarm, gender-unspecified veil selection was too opaque to use
when explicit descriptor evidence existed, and enclosed magenta-screen islands
survived the first runtime matte. All four were fixed with offline regressions.

The final stage audit found one more generic VN issue. Action-plus-dialogue facts
were too broad for the persistent first-meeting classifier, so an accepted lobby
pass temporarily lost its authored location options. VN location projection now
uses the broader transient visual-staging classifier while the durable
first-meeting ledger remains strict. The acceptance deck uses the approved 1F
courtyard from its first card.

Two non-sprite issues remain. One fresh `/begin` was safely rejected because the
router put `birth_stars` in an unsupported summon-state detail, and the accepted
fallback prose described a summoning circle and faceless idol that are absent
from the general courtyard plate. They are tracked as `ayoa-z95j` and
`ayoa-fzta` rather than being hidden by this feature.

## Configuration and coverage

The Master was the only human seat. Every embodied character and all character
generation roles were pinned to Luna; router and narrator retained their normal
Terra models.

| Role | Model |
| --- | --- |
| `agent` | `openai:gpt-5.6-luna` |
| `agent_standard` | `openai:gpt-5.6-luna` |
| `agent_convenience` | `openai:gpt-5.6-luna` |
| `character_manager` | `openai:gpt-5.6-luna` |
| `event_router` | `openai:gpt-5.6-terra` |
| `narrator` | `openai:gpt-5.6-terra` |
| `image_director` | `openai:gpt-5-mini` |

The real CLI and shared `EngineBridge` executed:

1. `/join 2`
2. `/settings presentation_mode visual_novel`
3. `/begin --confirm`
4. `/defer`

The primary replay committed three generated one-star Heroes, preserved all 13
seeded sprite bindings, 188 reviewed references, and 15 reviewed sprite sets.
The three new Heroes remained at one star, so their exact generated packs were
correctly not warmed or revealed. The Master saw only stable veiled variants.

A setup run that used numeric story alias `1` resolved to a different story and
was aborted during its first continuation. It is excluded from all evidence;
the reported runs use the exact `one_star_ascension_s1` id.

## Deck ledger

| Session / purpose | Cards | Deck id | Stage | Sprite result |
| --- | ---: | --- | --- | --- |
| Primary opening before location fix | 4 | `c3d39e9f0be1f501f09c9fed07983ea76817b21fe384b28e5b0ad17175f2aad4` | neutral | Iselle happy plus feminine veil |
| Primary autonomous exchange | 13 | `bb73fd1cba3d82261c96c76d22fe2af3254d0e90a7e834d1da0932a53053b9f4` | approved 1F courtyard | 13 sprite-bearing sections, one/two-character and neutral/tense/happy variants |
| Post-fix acceptance fallback | 3 | `f655b7fe2c409011d851425da7dee0a23db6e6974b151a318d363ceb36c5761f` | approved 1F courtyard | background-only establishing card, then centered Iselle neutral |

The two accepted reviewed-stage decks bind exact courtyard SHA-256
`04ac3c7d7926fb783e40b979254ac77d4956f450153b37f13bb61bd10d8f228f`.
The acceptance manifest reports `used_neutral_stage=false` for all three cards.

## On-the-fly generation evidence

Generated Heroes use one durable pack id. The runtime composes and hash-locks a
neutral full-body identity first, then independently composes the seven other
pose-expression variants with only that neutral PNG as the identity reference.
It does not use Qwen edit: the reviewed asset experiment showed that small
seam/prop edits repeatedly muddied clean pixels without repairing the prop.
Missing or failed expressions fall back to the ready neutral and never block
the story. Candidate-prewarm errors are logged and omitted; they cannot suppress
an otherwise renderable text-and-reviewed-sprite deck.

The offline end-to-end coordinator test completes all eight variants and proves
restart-safe durable resolution. A discarded live prewarm exposed six actual
FLUX neutral outputs before the transaction was cancelled. Those raw outputs
were retained only as diagnostics and were reprocessed without reviving their
cancelled jobs:

A final prompt-conflict scan also found and removed an instruction that asked
the neutral candidate to use both a neutral stance and a pose distinct from
neutral. Neutral now establishes the resting baseline; only the seven derived
variants are required to change both pose and expression. The generated-request
test asserts both sides of that contract.

| Candidate | Opaque hot-key pixels before | After corrected matte |
| --- | ---: | ---: |
| Renna | 5,245 | 0 |
| Halcyon | 0 | 0 |
| Castor | 16,042 | 1 |
| Wren | 5,342 | 1 |
| Rowan | 5,549 | 0 |
| Liora | 819 | 0 |

The corrected matte seeds every near-exact magenta region, including screen
islands enclosed by arms, capes, hair, or weapons, grows only through a bounded
near-key color region, physically unmixes partial edges, and applies a final
normalized-image residue gate. Native-resolution review preserved skin, red
hair, dark clothing, lantern glow and lattice, bows, swords, hands, feet, and
thin silhouette geometry.

## Feedback by source layer

### Sprite and compositor path: pass

- Reviewed bytes, opaque identity/variant handles, placement transforms, stage
  hash, sprite hashes, and final cards are all manifest-bound.
- One-character pages use a centered 98% layout. Two-character pages use 92%
  left/right layouts and deterministic inward facing.
- Sprite composition occurs after stage cover-crop and before the ADV box.
- Restart serves manifest-verified immutable card bytes; CLI and Discord share
  the same deck contract.
- Narrator sprite cues are transient presentation metadata. They are excluded
  from narrator history and source-id validation prevents them from leaking raw
  character ids.

### One-Star reveal policy: pass

- The Master sees birth-one/current-one Heroes through stable masculine or
  feminine veils; other viewpoints retain ordinary appearance knowledge.
- Explicit player-safe gender descriptors choose the corresponding veil.
  Genuinely unspecified or conflicting descriptions use a stable hash fallback.
- Seeded birth-one Heroes reveal their exact authored set at two stars.
- Generated birth-one Heroes begin neutral-first prewarm at two stars and reveal
  the generated pack at three stars.
- Seeded characters never enter generated prewarm merely because a binding is
  absent, and private sprite/reference metadata survives transaction copying.

### Narrator pacing: usable, still long

The primary `/defer` produced thirteen cards. The exchange was coherent, kept
central participants on adjacent action/dialogue pages, used tense only where
the visible action supported it, and did not interrupt the disembodied Master
for meaningless tactical input. Thirteen clicks for one defer is still a useful
tuning warning. Existing narrator rhythm work remains tracked by `ayoa-fhm6`;
this run does not support restoring a raw page-character limit.

### Stage selection: runtime contract fixed, semantic fit needs tuning

The accepted-render location bug is fixed: generic VN stage projection now uses
transient visually staged subjects for current-location inference, without
weakening physical first-meeting persistence. The acceptance deck proves direct
reviewed-stage selection survives an unavailable diffusion worker.

The selected courtyard is tier-appropriate and visually excellent, but the
fallback event's prominent summoning circle and faceless idol are not present.
The director already receives text-only applicability hints and explicit
permission to generate when none fits. `ayoa-fzta` tracks a focused director
harness and prompt tuning for landmark, facility, lighting, tier, and
indoor/outdoor mismatches.

### Router opening: safe failure, follow-up required

One fresh acceptance `/begin` exhausted state repair and routing correction on
an unsupported `birth_stars` detail. No invalid checkpoint committed. The next
defer safely produced a normal Summoning Hall state, but the intended initial
summon was lost. `ayoa-z95j` tracks the exact raw-message investigation and
adapter-local correction.

## Preserved artifacts

Raw local evidence is intentionally ignored by Git:

- `app/storage/sessions/one-star-vn-sprites-luna-playtest-post-matte-osa-20260827/`
- `app/storage/sessions/one-star-vn-sprites-luna-playtest-acceptance-20260827/`
- the three deck directories in the ledger
- `app/storage/runtime/image_generation/jobs.sqlite`
- `app/storage/playtest_reports/one_star_vn_sprites_runtime_20260827/matte_after_runtime/`

The selected decks, report, and matte diagnostics are also copied to the Windows
Pictures review folder reported in the implementation handoff.

## Verification

- Cross-cutting VN, image, narrator, One-Star, compositor, and delivery gate:
  370 passed.
- Full offline repository suite: 1,847 passed, 2 skipped.
- Changed-file Ruff, source/test/script compilation, and `git diff --check`:
  passed.
- Story asset validation: 188 reviewed references, including 120 immutable
  sprite PNGs in 15 sets with 13 seeded-character bindings; all declared hashes,
  byte counts, dimensions, ownership, and on-disk files validated.
- Both copied Windows deck directories and the copied report are byte-identical
  to their repo/runtime sources.
