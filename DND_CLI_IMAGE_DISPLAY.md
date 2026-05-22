# D&D CLI Image Display Target

Status: accepted decision for `ayoa-v1n`; initial CLI implementation landed for
`ayoa-0f0`.

This document chooses the CLI image display target for reviewed, player-safe
maps and handouts. The current CLI now resolves per-POV safe payloads, writes an
Ayoa-managed cache copy, and uses the iTerm2 inline image protocol when a
supported iTerm2 or WezTerm terminal is detected. Unsupported terminals remain
an explicit degraded fallback and must not be described as successful image
display.

## Grounding

This decision sits downstream of:

- `DND_ASSET_REVEAL_CONTRACT.md`, which makes image reveals router-owned event
  side effects filtered into `SafeAssetRevealPayload` records per POV.
- `DND_ASSET_BACKING_AND_DELIVERY.md`, which makes
  `asset://<pack_id>/<asset_id>` the canonical logical delivery ref and says
  CLI image delivery is complete only after resolving safe bytes to a real local
  display/export path.
- `DND_MODULE_PRIVACY_BOUNDARY.md`, which forbids source paths, raw protected
  material, hidden labels, source refs, and unsafe delivery refs from CLI output
  and normal logs.
- `DND_CONTENT_PACK_AUTHORITY.md`, which keeps runtime authority in reviewed
  compiled packs plus checkpoint overlay state.
- `DND_TACTICAL_MAP_GEOMETRY.md`, which treats map images as player display
  artifacts, not strict geometry or rules authority.

## Decision

The first supported CLI inline image backend is the iTerm2 inline image protocol,
emitted through an Ayoa-owned renderer compatible with WezTerm and iTerm2. The
implementation may initially shell out to `wezterm imgcat` when a compatible
WezTerm binary and terminal session are detected, but the runtime contract is
the protocol/rendering behavior, not the presence of a raw source file on disk.

This backend is chosen because it has a simple file/byte display model, is
supported by iTerm2, is implemented by WezTerm through `wezterm imgcat`, and can
be reached after Ayoa writes a reviewed player-safe cache copy. It is narrower
than "any terminal image support"; unsupported terminals are explicit fallback
or blocker cases, not silent text-only success.

The supported CLI flow is:

1. Receive already filtered `SafeAssetRevealPayload` records from the same
   per-POV response surface Discord will use.
2. Resolve `asset://<pack_id>/<asset_id>` through the reviewed pack asset
   resolver, validating pack identity, asset review status, MIME type, size,
   hash, audience, and presentation.
3. Write a player-safe content-addressed copy under an Ayoa-managed runtime
   cache/export root using a generated filename that reveals no source path,
   source filename, hidden map label, or protected title.
4. Render the safe copy inline through the iTerm2 protocol backend when the CLI
   can prove a supported terminal/session path.
5. If inline rendering is unavailable, report a non-spoiling "image could not be
   displayed here" notice and, only when explicitly configured, expose the safe
   export path. This is degraded fallback, not completed inline image display.

Kitty graphics protocol, Sixel, Unicode half-block renderers, web previews, and
OS-specific viewer launching are deferred. They can be added later as separate
backend adapters behind the same safe cache and per-POV filtering contract, but
the first implementation should not pretend to support them by printing an
asset ref.

## Per-POV Visibility

CLI image delivery uses `per_player_asset_reveals` as authoritative. A joined
multi-character CLI session must render assets separately for each bound
character POV rather than merging all reveals into one table-wide feed.

Visibility rules:

- `audience="all_observers"` displays only to CLI-controlled characters listed
  as observers on the owning canonical event.
- `audience="only"` displays only to the listed visible character ids, narrowed
  by visible user ids when the reveal is user-bound.
- If two locally controlled characters are allowed to see different crops,
  fog states, handouts, or map overlays from the same turn, the CLI must display
  them as separate POV sections and must not reuse one character's cache record
  as another character's reveal.
- The CLI must not infer visibility from the session, party, current channel,
  current combat, or map name. It consumes the filtered response payload.

No image bytes, source refs, local paths, hidden labels, or private catalog
metadata are sent to narrator prompts, character-agent prompts, or router prose.
If the image content changes what a character can perceive, that perception must
also be canonicalized as visible text facts.

## Local Artifact Handling

The renderer works only from resolved safe bytes, never from raw module files,
private extraction files, source PDFs, source images, or unchecked delivery refs.

The cache/export root should be local-only and outside portable checkpoint
state, for example:

```text
app/storage/runtime/asset_cache/<session_id>/<sha256>.<safe_ext>
```

The exact root can move to a user cache directory later, but the semantics are:

- filenames are generated from content hashes or opaque reveal ids, not source
  filenames, room names, map labels, handout titles, or asset ids that may spoil
  content;
- the cache contains only reviewed player-safe derivative bytes after resolver
  validation;
- the cache path is allowed to appear in the local CLI output only after the
  player was eligible for that reveal and only when safe export fallback is
  enabled;
- normal logs record sanitized ids, hashes, sizes, MIME types, backend names,
  and failure codes, not cache absolute paths by default;
- debug logs may include local cache paths only behind an explicit local debug
  flag and must never include raw source paths or protected material.

`asset://` remains the canonical checkpoint and response ref. Cache paths are
ephemeral delivery artifacts. They are not router input, narrator input,
checkpoint authority, pack authority, or source provenance.

## Rewind Implications

Rewind restores checkpoint reveal state, map fog/crops, and per-POV visibility.
It does not need to delete content-addressed safe cache files on every rewind,
because those files are derived player-safe artifacts. It must, however, make
the external transcript match the rewound turn state:

- a rewound turn's CLI transcript should no longer present that turn's image
  reveal as active;
- replaying after rewind must re-resolve or re-associate reveal state from the
  checkpoint rather than trusting stale in-memory display history;
- cache hits are allowed only after the current checkpoint state still authorizes
  the same `pack_id`, `asset_id`, hash, audience, and POV;
- fog/crop/overlay derivatives are distinct cache entries by derivative id or
  hash so a previously revealed wider map cannot satisfy a later narrower POV;
- logs for rewind cleanup mention sanitized reveal ids and cache hashes, not
  source paths or spoiler titles.

Inline terminal pixels already printed before rewind cannot be physically erased
from scrollback in a portable way. The CLI must treat rewind as state restoration
and transcript correction, not as guaranteed terminal scrollback redaction. For
private play, this is acceptable only because the image was player-safe for the
recipient at the time it was displayed.

## Failure Behavior

Image delivery failures are non-spoiling and local to the CLI operator.

Contract failures include missing packs, pack/hash mismatches, unsafe schemes,
missing or unreviewed assets, unsafe MIME types, oversized bytes, hash mismatch,
and reveal/observer violations. These are loud runtime errors before delivery is
reported as successful. The CLI-facing text should be generic, such as:

```text
Could not display the revealed image.
```

Transport failures include unsupported terminal, tmux/multiplexer limitations,
missing `wezterm` executable when the implementation depends on it, failed
protocol write, cache write failure, or viewer refusal. These may fall back to a
safe export path only when the export path has already been generated from
reviewed safe bytes and export fallback is enabled.

The CLI must not:

- print raw source paths, private extraction paths, source PDF paths, source
  image filenames, hidden map labels, or protected excerpts;
- print unsafe `delivery_ref` values or raw non-`asset://` refs;
- print a caption, asset id, or logical ref as if an image was displayed;
- ask the router, narrator, or D&D combat manager to improvise a replacement
  image;
- collapse per-POV failures into a public/session-wide reveal.

## Test Strategy

Implementation tests should use synthetic packs and generated test images, not
private module files.

Minimum coverage:

- backend selection identifies supported iTerm2/WezTerm sessions and marks tmux,
  dumb terminals, and unknown terminals unsupported unless a compatible passthru
  path is explicitly implemented;
- text-only caption/ref output is labeled as degraded fallback and does not pass
  image-display success assertions;
- resolver/cache tests reject absolute paths, relative source paths, unapproved
  schemes, source filenames, unreviewed assets, unsafe MIME types, oversize
  files, and hash mismatches;
- cache filenames are generated from hashes or opaque ids and do not include
  source refs, asset titles, room names, handout titles, or hidden labels;
- two player-bound characters in one CLI process receive distinct image payloads
  when visibility or fog/crop state differs;
- `all_observers` and `only` produce different per-POV render sets;
- rewind restores reveal authorization and does not use a stale broader crop or
  stale display marker after the checkpoint rolls back;
- normal CLI output and normal logs omit raw source paths, private extraction
  paths, protected sentinel strings, hidden labels, raw OCR, and unsafe
  delivery refs;
- unsupported-backend failure remains non-spoiling and does not expose the safe
  caption/title unless that caption/title was already authorized for the same
  POV.

No live terminal integration test should be required in offline CI. The terminal
writer should be isolated behind an interface so unit tests can assert the bytes
or command arguments sent to the backend without requiring iTerm2, WezTerm, a
real TTY, or a hosted LLM key.

## Implementation Status

The initial CLI path now has:

- `TurnResponse` and frontend DTO fields for per-POV safe asset reveals;
- a byte resolver for `asset://<pack_id>/<asset_id>` that validates reviewed
  pack rows, hashes, MIME types, size limits, and player-safety gates;
- an Ayoa-managed safe cache writer with generated non-spoiling names;
- an iTerm2 inline image protocol renderer with terminal/session detection and
  no raw source-path arguments;
- per-POV CLI rendering for multiple locally controlled characters;
- output and cache hygiene tests using synthetic sentinel data.

Remaining follow-up work is additive: more terminal backends, richer local
viewer/export integrations, and explicit rewind transcript bookkeeping beyond
the current checkpoint-authorized re-resolution path. Unsupported terminals may
show explicit degraded notices and, only with an explicit CLI flag, the generated
safe cache path.

## External Backend References

- Kitty graphics protocol: https://sw.kovidgoyal.net/kitty/graphics-protocol/
- iTerm2 inline image protocol: https://iterm2.com/documentation-images.html
- WezTerm `imgcat`: https://wezterm.org/imgcat.html
