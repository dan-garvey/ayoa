# D&D Module Privacy Boundary Decision

Status: decided for the protected module import path.

This decision applies to protected or privately owned adventure modules,
including scanned PDFs, extracted text/images, manual import notes, compiled
packs, runtime lookup records, projection/application profiles, checkpoints,
logs, Discord, CLI, review exports, and project task tracking.

## Current Grounding

This policy follows the current module direction in `DND_MODULE_IMPORT.md`:

- module import is a compiler/review workflow, not turn-time raw OCR RAG
- modules are reviewed content packs with projection/application profiles
- runtime lookup reads compiled records and profile projections, not raw pages
- checkpoints store refs, hashes, reveal state, projections, and mutations
- images are private table assets; players receive only reviewed player-safe
  reveals
- narrator prompts receive visible facts and safe captions only
- character agents receive text observations only

It also matches the current code surfaces:

- `app/engine/content_lookup.py` does bounded deterministic lookup and fails
  loudly with `MissingContentError` when required content is unavailable.
- `app/engine/content_resolver.py` formats compact router history records:
  `content_known`, `location_card`, and `front_signal`.
- `app/engine/content_assets.py` writes asset catalogs with forbidden metadata
  keys removed, rejects absolute-path delivery refs, and emits
  `SafeAssetRevealPayload` only for approved player-safe assets.
- `app/schemas/content_pack.py` separates provenance, page inventory, compiled
  cards, aliases, coverage manifests, private image assets, asset reveal
  requests, and safe reveal payloads.
- `app/schemas/responses.py` keeps `TurnResponse` presentation-shaped; it does
  not include a debug payload or raw prompt/source records.
- `app/bot/commands.py` sends `TurnResponse` prose, dice, XP, loot, and reaction
  affordances through Discord; future asset delivery must use the same POV
  filtering and no-public-fallback rule.
- `tests/test_prompt_hygiene.py` is negative prompt hygiene only and should be
  extended with concrete forbidden module leakage patterns, not prose-freezing
  prompt assertions.

## Decision

Protected module data has three runtime boundaries:

1. `Local-only raw artifacts`: original PDFs, rendered pages, extracted image
   streams, OCR text, layout JSON, raw source paths, DM map labels, review
   screenshots, crop masks, and raw review notes. These are private source
   derivatives. They may exist only in ignored local storage such as
   `private_extractions/raw/`, `private_extractions/review/`, or a user-owned
   path outside the repo.
2. `Router-only adjudication context`: reviewed authored or compiled module
   records selected for the event router, D&D combat manager, or another
   router-equivalent adjudicator. This context may include hidden module facts
   needed to DM properly: keyed-area summaries, secret-feature state, trap or
   hazard summaries, front dossiers, unrevealed reveal candidates, reviewed
   tactical geometry summaries, refs, hashes, visibility gates, and safe asset
   candidates. These records are not observable facts by themselves and are not
   safe for narrator prompts, character-agent prompts, player output, ordinary
   logs, or default checkpoint exports.
3. `POV-safe projection surface`: narrator prompts, character-agent prompts,
   Discord output, CLI output, player-facing responses, ordinary logs, default
   checkpoint exports, Beads/GitHub text, and normal test reports. These
   surfaces receive only canonical visible facts, text observations, refs,
   hashes, reveal state, player-safe asset payloads, aggregate diagnostics, and
   other allowlisted fields below.

The private compiled pack is the storage substrate for the second boundary. It
is reviewed, redacted, runtime-readable SQLite data and media addressed by
stable ids and content hashes. The current default path is
`private_extractions/compiled/`, which is gitignored. A compiled pack may contain
provenance ids, page ids, span ids, image ids, hashes, confidence, review state,
redacted summaries/bodies, reveal triggers, safe captions, and safe delivery
refs. It must not contain raw PDF text/images or protected excerpts in runtime
card text.

No runtime path may fall back to raw OCR JSONL, raw PDF pages, extracted images,
or unreviewed review artifacts. If a compiled ref, pack hash, safe asset, or
review gate is missing, the runtime must fail loudly with a non-spoiling error
instead of asking the router to improvise from raw source material.

## Boundary Matrix

| Destination | Allowed source-derived fields | Forbidden fields/material |
| --- | --- | --- |
| Hosted LLM prompts: event router or router-equivalent adjudicator | Reviewed router-only compiled records needed for adjudication: `content_known`, `location_card`, `front_signal`, keyed-area summaries, secret-feature refs, trap/hazard summaries, front dossiers, unrevealed reveal candidates, reviewed tactical geometry summaries, visibility gates, refs, hashes, pack ids, and safe asset candidate ids/captions. The current compact-record projection is listed below; any larger projection must name exact fields and tests before implementation. | Raw PDF text, rendered page images, extracted images, OCR/layout JSON, source paths, page screenshots, raw provenance JSON, unredacted card `body`, unredacted card `metadata`, page inventory notes, raw DM-only map labels, unsafe `delivery_ref`, protected excerpts, raw review notes, image bytes or image-understanding payloads. |
| Hosted LLM prompts: narrator | Canonical observable facts already visible to that POV. After a router-owned asset reveal, a player-safe caption may be converted into visible prose. | `pack_id`, `asset_id`, `source_ref`, `delivery_ref`, hidden captions, hidden image refs, map labels, DM notes, raw source text/images, protected excerpts, compact hidden router records. |
| Hosted LLM prompts: character agents | Text observations the character can perceive, plus ordinary character state. If an image matters to an NPC, the router must canonicalize the perceivable content as text first. | Asset reveal structs, `SafeAssetRevealPayload`, `delivery_ref`, source refs, hidden content records, raw source material, protected excerpts, DM-only labels. |
| Hosted LLM prompts: import helpers | Only redacted candidate fields whose text has already passed the same `protected_terms` and absolute-path checks as compiled card text. | Raw PDF text/images, OCR dumps, source page images, source paths, protected excerpts, unattended OCR-RAG prompts. Hosted import over raw protected source remains blocked by this decision. |
| Checkpoints | `session.content_state[pack_id]`; `ContentPackState.pack_id`; `introduced_refs` fields `pack_id`, `ref_id`, `content_hash`, `label`, `kind`, `source_event_id`, `introduced_at_s`; `pending_signals` fields `signal_id`, `pack_id`, `ref_id`, `content_hash`, `reason`, `source_event_id`, `status`, `priority`, `created_at_s`, `requested_fields`; `fronts` fields `front_id`, `label`, `status`, `clock`, `max_clock`, `villain_ids`, `introduced_ref_keys`; `villains` fields `villain_id`, `label`, `status`, `front_ids`, `goals`, `introduced_ref_keys`; campaign overlay fields such as visited refs, reveal state, fog/masks by ids, consumed refs, spawned refs, and pack/version/content hashes. | Raw images, page scans, OCR text, layout JSON, full source cards, protected excerpts, raw review notes, absolute source paths, raw local asset paths, unsafe delivery refs, hidden DM labels not transformed into refs or redacted summaries. |
| Normal logs and playtest reports | Counts, timing, model/cache metrics, route names, session ids, turn indexes, event ids, failure codes, sanitized `pack_id`, `ref_id`, `content_hash`, review/gate statuses, and non-source error categories. | Raw source text/images, protected excerpts, source paths, local pack paths, DM-only labels, card `body`, card `metadata`, raw asset `delivery_ref`, `SafeAssetRevealPayload.delivery_ref` when it is not a safe logical ref, imported aliases/reasons that contain protected source wording. |
| Beads, GitHub issues, PR text, commit messages | Code paths, schema field names, test names, synthetic fixture ids, aggregate counts, generic failure categories, and redacted examples. | Real protected module titles, real private `pack_id` values when identifying the source, real refs/aliases/room names from the module, source fingerprints, source paths, OCR text, page images, excerpts, screenshots, DM labels, private asset ids that reveal spoilers. |
| Gitignored private compiled packs | `CoverageManifest`, `PageInventoryRecord`, `ContentProvenance`, `CompiledContentCard`, `ContentAliasRecord`, `ContentImageAsset`, asset catalog rows, safe media files, and reviewed derivative media. Compiled card `title`, `summary`, `body`, and `reveal_trigger` must be redacted/paraphrased runtime text. Provenance may store `source_asset_id`, `page_id`, `span_id`, `image_id`, `bbox`, `section_id`, `method`, `confidence`, `importer_version`, and `human_review_status`. | Original PDFs, rendered source pages, raw OCR dumps, raw layout JSON, raw extracted source images, absolute source paths in persisted metadata, protected excerpts in card text, unsafe asset delivery refs. Those belong only to local raw/review artifacts. |
| Private review exports | If stored under ignored local review storage and marked private: rendered page refs, page thumbnails/crops, OCR text, normalized spans, source paths, DM labels, raw review notes, protected excerpts, blocking issues, coverage warnings, and proposed card diffs. Redacted shareable review exports may contain only compiled-pack fields plus synthetic excerpts. | Any review export containing raw/protected source material is forbidden from git, Beads, hosted runtime prompts, normal logs, normal checkpoints, Discord, CLI, and CI artifacts. |
| CLI output | `TurnResponse.output_text`; joined-character `per_player_renders`; dice, XP, loot, reaction, and commitment affordances; safe asset captions; safe non-path local refs such as `asset://pack/asset` after POV filtering; generated safe cache/export paths only behind an explicit local CLI flag after the same POV authorization. | Raw PDF text/images, source paths, raw local pack or extraction paths, default local filesystem path output, unrevealed filenames, DM labels, hidden refs, `source_ref`, unsafe `delivery_ref`, protected excerpts, prompts, compact hidden router records. |
| Discord output | Same prose and affordances as CLI, delivered per POV. Future asset delivery may attach only `SafeAssetRevealPayload` fields: `pack_id`, `asset_id`, `kind`, `title`, `mime_type`, `width`, `height`, `sha256`, safe `delivery_ref`, `presentation`, `caption`, `alt_text`, after approval and POV filtering. | Public fallback for private asset delivery, raw source paths, absolute file paths, unreviewed or unsafe assets, DM/source images, hidden map labels, source refs, protected excerpts, raw OCR/page text, raw prompt/debug dumps. |
| Local-only raw artifacts | Original PDF bytes, rendered pages, extracted images, OCR text, generic text extraction, layout JSON, thumbnails, crop masks, raw source paths, source catalog SQLite rows, review screenshots, source hashes, page labels, bboxes, confidence, extraction methods, and raw review notes. | These artifacts are forbidden from git, Beads, normal logs, normal checkpoints, hosted runtime prompts, CLI output, Discord output, and ordinary test artifacts. |

## Explicit Forbidden-Destination Rules

These rules are testable as deny lists using synthetic sentinel values.

| Material | Allowed only in | Forbidden everywhere else, specifically |
| --- | --- | --- |
| Raw PDF bytes, rendered page images, extracted source images, raw page thumbnails, crop masks | Local-only raw artifacts and private review exports under ignored storage | Hosted LLM prompts including router prompts, checkpoints, logs, Beads/GitHub, git-tracked files, compiled runtime card text, CLI, Discord, normal test reports |
| Raw OCR/generic PDF text, raw layout JSON, raw text spans | Local-only raw artifacts and private review exports under ignored storage | Hosted LLM prompts including router prompts, checkpoints, logs, Beads/GitHub, git-tracked files, compiled runtime card text, CLI, Discord, normal test reports |
| Absolute source paths and raw local paths | Local-only raw artifacts and private review exports under ignored storage | Hosted LLM prompts including router prompts, checkpoints, logs, Beads/GitHub, git-tracked files, compiled packs, asset catalogs, `SafeAssetRevealPayload`, CLI, Discord |
| DM-only labels, hidden map labels, unrevealed room names, secret labels | Local-only raw artifacts, private review exports, or compiled pack internals only after transformation into redacted refs/summaries with explicit visibility | Narrator prompts, character-agent prompts, player CLI/Discord output, Beads/GitHub, normal logs, public review exports, safe asset captions/alt text |
| Protected excerpts from module prose | Local-only raw artifacts and private review exports under ignored storage | Hosted LLM prompts including router prompts, checkpoints, logs, Beads/GitHub, git-tracked files, compiled card `title`/`summary`/`body`/`reveal_trigger`, CLI, Discord, normal test reports |
| Unsafe asset refs: absolute paths, unreviewed `delivery_ref`, DM/source image refs, hidden asset ids, non-player-safe assets | Local-only raw artifacts, private asset catalog internals before sanitization, private review exports, and router-only adjudication context after sanitization to reviewed ids/captions | `SafeAssetRevealPayload`, CLI, Discord, narrator/agent prompts, POV-safe logs, Beads/GitHub, checkpoints unless represented only as hidden reveal state ids |

## Field-Level Projection Contracts

### Runtime Content Records

The first router projection is `format_compact_record()`. It and callers may
project only these fields into compact router history:

```text
content_known ref scope visibility hash kind pack summary
location_card ref exits hazards clues visibility hash pack summary
front_signal ref actor knows pressure visibility hash pack summary
```

The router may use these records, and any later reviewed router-only compiled
record projection, as hidden adjudication context. They are not observable facts
by themselves. To become player-visible, the router must emit a canonical
observable fact or an `AssetReveal` with normal visibility.

`CompiledContentCard.body`, `metadata`, `provenance`, page inventory rows, and
asset catalog rows are not part of the current compact-record prompt contract.
A later router-only projection may add reviewed fields from those records, but
it must name the exact fields, prove they are not raw source/protected excerpts,
and add prompt hygiene tests before implementation.

### Checkpoint Content State

Checkpoints store content memory and mutable campaign overlay, not source
material. `ContentPackState.metadata` is currently a scaffold escape hatch used
by lookup tests for `db_path`; real protected packs must move to a pack registry
or logical locator before release-candidate use. Until that decision lands:

- no checkpoint exported for review or sharing may include absolute `db_path`,
  `pack_path`, `sqlite_path`, or `content_db_path`
- no checkpoint may include raw source paths
- code that needs a local pack path should resolve it from a local-only registry
  rather than from portable checkpoint content

This is a blocker for release-quality private packs, not permission to ship
absolute paths in checkpoints.

### Asset Reveals

`AssetReveal` is router-owned intent:

```text
pack_id asset_id audience visible_to_character_ids visible_to_user_ids
presentation caption
```

It is not directly player-safe. The only player/LLM-safe asset payload is
`SafeAssetRevealPayload`:

```text
pack_id asset_id kind title mime_type width height sha256 delivery_ref
presentation caption alt_text
```

`safe_asset_reveals_for_viewer()` must remain the filter boundary. A payload is
deliverable only when the underlying `ContentImageAsset` is
`safe_for_players=true`, has `review_status in {"reviewed", "approved"}`, and
has a non-path safe `delivery_ref`.

`ContentImageAsset.source_ref`, `review_status`, `spoiler_class`,
`safe_for_players`, `safe_for_llm`, and sanitized metadata may live in the
private asset catalog. They must not be emitted to player output. `safe_for_llm`
does not by itself authorize image bytes or source refs in hosted prompts; it
only allows a future prompt projection if that projection is separately
specified and tested.

### Logs And Errors

`MissingContentError` and other lookup failures may identify a missing
`pack_id`, `ref_id`, and generic reason code. For protected content, aliases and
reasons must be sanitized before logging because aliases can be source-derived
room names, secret labels, or module prose. Player-facing errors must say the
content is unavailable without naming hidden refs or source labels.

## Enforcement And Test Implications

Downstream tasks should turn this policy into tests before wiring real protected
packs into runtime play.

Required tests:

- Prompt hygiene: scan rendered router, narrator, character-agent, and D&D Cat
  II prompts with sentinel values for raw OCR, source path, protected excerpt,
  unsafe asset ref, private `delivery_ref`, and image payloads. Also assert that
  reviewed hidden router records stay out of narrator, character-agent, player
  output, ordinary logs, and default checkpoint exports.
- Router compact-record allowlist: assert only the named fields in
  `content_known`, `location_card`, and `front_signal` survive formatting.
- Lookup failure rollback: missing refs must leave `session_conversation` and
  `content_state` unchanged and must not call the hosted router.
- Checkpoint hygiene: checkpoint dumps with content packs must contain only
  refs, hashes, review/reveal state, and campaign overlay fields; sentinel raw
  source fields must be absent.
- Pack compiler hygiene: compiled card `title`, `summary`, `body`, and
  `reveal_trigger` must reject protected excerpts and absolute paths; metadata
  must drop forbidden keys recursively.
- Asset catalog hygiene: asset metadata must drop forbidden keys; unsafe
  `delivery_ref` values must not persist as deliverable refs; unreviewed or
  non-player-safe assets must not produce `SafeAssetRevealPayload`.
- Discord delivery: private/single-POV asset reveals must never fall back to a
  public channel. If private delivery fails, strip the asset or fail with a
  non-spoiling error.
- CLI delivery: safe captions and safe logical refs may print; generated safe
  cache/export paths may print only behind an explicit local flag after POV
  filtering; source paths, unrevealed filenames, hidden labels, and unsafe
  delivery refs must not print.
- Log/report hygiene: normal logs and playtest reports must omit protected
  sentinel strings, absolute paths, raw card bodies, raw OCR, and unsafe asset
  refs.
- Beads/Git hygiene: committed fixtures, issue text, and docs may use synthetic
  sentinel refs only; tests must not require private source files.
- Review export split: private review exports containing raw/protected material
  must be written only under ignored storage and marked private; redacted
  shareable exports must pass the same deny-list scan as logs.

Tests should use synthetic packs and sentinel strings. Do not write tests that
depend on proprietary module excerpts, real page images, or
`stories/curse_of_strahd.pdf`.

## Blocked Or Deferred Decisions

These items are intentionally not decided here:

- Pack locator/registry: current lookup scaffolding accepts local paths such as
  `metadata.db_path`. Release-quality private packs need a local-only registry
  so portable checkpoints do not persist absolute pack paths.
- Larger router-only card projection: current router prompts receive compact
  record strings, not full `CompiledContentCard.body` or provenance. A future
  projection may provide more reviewed compiled fields to the router, but must
  name exact fields and tests first and must not widen narrator, agent, player,
  log, or checkpoint surfaces.
- Asset delivery integration: `SafeAssetRevealPayload` exists, but Discord and
  CLI asset transport still need concrete wiring and no-public-fallback tests.
- Local private model policy: raw OCR/page-image assistance is allowed only as a
  local-only import aid. Sending raw protected module pages or OCR to hosted
  import prompts remains blocked by this decision.
- Shareable review export schema: private review exports may contain raw source
  material under ignored storage. Any shareable/exportable format must be
  redacted and separately specified.
