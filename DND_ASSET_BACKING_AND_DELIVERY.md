# D&D Asset Backing And Private Delivery Semantics

Status: accepted decision for the asset backing and delivery slice.

This document decides how player-safe image assets are backed, resolved to
bytes, delivered to Discord and CLI surfaces, and unwound during rewind. It is
grounded in the current asset catalog, reveal schema, Discord delivery helpers,
CLI display path, and module-import contract.

## Current Surfaces

The existing asset and delivery code establishes these constraints:

* `app/schemas/content_pack.py` defines `ContentImageAsset`, `AssetReveal`,
  and `SafeAssetRevealPayload`. The reveal audience is currently
  `all_observers` or `only`; the presentation is `inline`, `attachment`,
  `reference`, or `map_overlay`.
* `app/engine/content_assets.py` stores asset catalog rows, removes private
  metadata keys, strips absolute paths from text, filters unsafe/unreviewed
  assets, and returns only `SafeAssetRevealPayload` records. Delivery-ref
  validation accepts matching `asset://<pack_id>/<asset_id>` refs and rejects
  unsafe schemes and paths before byte resolution.
* `app/engine/content_asset_bytes.py` resolves reviewed player-safe assets to
  verified bytes with MIME, size, and hash checks.
* Asset coverage is split across `tests/test_content_asset_bytes.py`,
  `tests/test_cli_image_display.py`, `tests/test_bot_internals.py`, and
  `tests/test_play_cli.py`.
* `app/bot/commands.py` currently sends text renders through
  `_post_actor_render`, `_post_to_pov`, and `_send_public_turn_render`.
  Asset delivery uses attachment-aware helpers that avoid the public fallback
  branch for private image delivery.
* `app/bot/session_map.py` records Discord turn messages with
  `channel_id`, `session_id`, `turn_index`, `discord_channel_id`,
  `message_id`, `delivery`, and optional `recipient_user_id`.
  `app/bot/commands.py` rewind cleanup uses those refs to delete or hide
  messages from rewound turns.
* `DND_MODULE_IMPORT.md` treats adventure images as private table assets,
  stores refs and reveal state instead of bytes in checkpoints, and requires
  private/single-POV image reveals never to fall back to public posting.

## Decision

Use `asset://` as the canonical runtime delivery ref. An asset reveal is not a
completed image feature until the runtime can resolve that ref to bytes and
deliver those bytes to the intended human surface.

Approved HTTPS/CDN storage is allowed only as resolver backing, not as a
router-authored public shortcut. The resolver may dereference a configured,
allowlisted HTTPS origin for a reviewed asset, but the checkpoint, router
payload, and player-facing reveal contract should continue to name the asset by
stable `pack_id` and `asset_id`. This keeps provenance, privacy, and rewind
state independent from storage location.

The canonical shape is:

```text
asset://<pack_id>/<asset_id>
```

The resolver must validate that the URI pack id and asset id match a catalog
row and that the row is safe for the requesting audience before reading bytes.
The resolver may map that row to one of these backing stores:

* compiled-pack media addressed by content hash
* an Ayoa-managed private media cache addressed by content hash
* an approved HTTPS/CDN object listed in pack configuration and verified by
  hash before delivery

The resolver must not read raw source paths or private extraction artifacts.

## Rejected Backing Schemes

Reject these as asset delivery refs:

* absolute filesystem paths, including source, cache, repo, and home paths
* relative filesystem paths, because they are ambiguous and can silently become
  repo/workdir dependent
* `file://`, `data:`, `javascript:`, `ftp://`, `s3://`, `gs://`, and other
  non-approved schemes
* raw Discord attachment/CDN URLs as canonical refs
* arbitrary `http://` or `https://` URLs not declared by pack configuration as
  approved media backing
* text-only labels, captions, source refs, or asset ids presented as if they
  were delivered images

`source_ref` remains private provenance. It is not a delivery ref and must not
be sent to players, LLM prompts, or Discord attachments.

## Allowed Delivery Refs

At runtime, a player-safe `SafeAssetRevealPayload.delivery_ref` may be:

* `asset://<pack_id>/<asset_id>`, preferred and canonical
* an approved HTTPS/CDN backing ref only inside the resolver or compiled pack
  configuration, where the host is allowlisted and the downloaded bytes verify
  against the catalog hash before use

Direct HTTPS refs in player-facing payloads should be avoided for private
packs. If an open/public pack later needs stable direct HTTPS delivery, that is
a separate public-pack mode and must still require HTTPS, host allowlisting,
content length limits, MIME validation, and hash verification.

## Asset Byte Resolution

The byte resolver input is the safe payload plus the current pack registry. It
must perform these checks before returning bytes:

1. The delivery ref uses an allowed form.
2. The referenced pack is installed and matches the checkpoint's expected
   `pack_id`, version, and content hash.
3. The catalog row exists and matches the requested `asset_id`.
4. The asset is `reviewed` or `approved`, `safe_for_players=true`, and has a
   safe delivery ref.
5. The resolved bytes match the catalog `sha256`.
6. The MIME type and file extension are safe for the target surface.
7. The byte size is within the Discord or CLI delivery limit.

Missing packs, pack-hash mismatches, missing asset refs, hash mismatches,
unsafe MIME types, and oversized assets are hard delivery errors. They must not
prompt the router or narrator to improvise a replacement image.

## Discord Delivery

Discord delivery uses the same per-POV visibility filtering as text renders,
but image attachment delivery must be its own path rather than blindly reusing
the text fallback cascade.

For each safe reveal payload:

1. Filter by viewer using `all_observers` or `only` semantics.
2. Resolve bytes through the asset resolver.
3. Send a Discord attachment with a safe generated filename, safe caption, and
   optional embed metadata. The filename must not expose source filenames,
   labels, or private paths.
4. Record every successful asset message in `SessionMap.turn_messages` with
   the engine `turn_index`.

Private and single-POV image delivery order is:

```text
private POV thread -> user DM -> private failure report
```

It never continues to:

```text
public channel fallback
```

This differs from `_post_actor_render` text delivery, where public fallback can
be acceptable for the actor's prose. A private image failure must remain
private. If both the thread and DM fail, the runtime should report a
non-spoiling delivery failure to the invoker or session owner when possible and
log the technical reason server-side. It must not post the image, safe caption,
asset id, or spoiler-bearing title to the public channel.

## Public Map Placement

With the current `AssetReveal` schema, public maps should still be delivered
only through filtered POV surfaces.

`audience="all_observers"` means all characters in the current canonical
event's observer set, not all Discord users in the channel and not every
character in the session. A main-channel image post cannot express that
observer boundary. Therefore an `all_observers` map reveal is broad within the
event, but not automatically channel-public.

A future explicit channel reveal state may allow main-channel map posts when
the router/reveal contract says the asset is visible to the shared table
surface, such as a fully player-safe battle map that every current player is
allowed to see. That requires an explicit channel-wide reveal marker, not an
inference from `all_observers` alone. Until that marker exists, Discord asset
delivery stays POV-thread/DM scoped.

## CLI Delivery

CLI uses the same resolver and the same safe payloads as Discord. It may not
expose raw source paths, private extraction paths, source filenames, or hidden
map labels.

CLI image delivery is considered complete only when the CLI resolves bytes and
does one of the following:

* renders the image through a terminal image protocol
* writes a player-safe copy into an Ayoa-managed cache/export path named by
  safe generated filename or content hash and reports that safe path
* opens a player-safe copy through an explicit local viewer integration

Printing a caption, `asset://` ref, or local source path alone is a degraded
non-image notice, not a completed image feature. Safe captions and asset refs
may be useful while the image feature is incomplete, but they must be labeled
as non-image fallback behavior in implementation and tests.

CLI failures should be non-spoiling and local to the CLI operator, for example
"Could not display the revealed handout." Technical details can go to logs.

## Visibility Semantics

`AssetReveal` follows the same visibility boundary as observable facts:

* `all_observers` reveals to each viewer whose character id is in the current
  event observer set. If the observer set is empty, no player receives it.
* `only` reveals to explicit `visible_to_character_ids` or
  `visible_to_user_ids`. Character ids are preferred because they survive
  rebinding; user ids are for delivery cases where the reveal is truly bound to
  a Discord user or private handoff.

Neither mode sends hidden metadata to the narrator, character agents, or the
router as image bytes. If an image changes what an NPC or player can perceive,
the router should canonicalize that perception as text facts and use the asset
payload only for human image delivery.

## Failure Reporting

Failures are split into contract failures and transport failures.

Contract failures include missing packs, pack-hash mismatches, invalid refs,
unsafe schemes, missing assets, unreviewed assets, unsafe MIME types, oversized
bytes, and hash mismatches. These should fail loudly in runtime delivery with a
non-spoiling player-facing error and detailed server logs. They should not be
silently stripped, summarized as if delivery succeeded, or replaced by
router/narrator prose.

Transport failures include Discord thread, DM, attachment upload, or CLI
viewer/write failures after the resolver has produced safe bytes. These are
reported only to the intended private recipient, invoker, or session owner
when possible. A private/single-POV transport failure must never become a
public post.

For multi-recipient `all_observers` delivery, one recipient's transport
failure does not block delivery to other eligible recipients. The failed
recipient receives a private non-spoiling notice when possible, and the logs
identify the asset and delivery target for operator diagnosis.

## Rewind Requirements

Checkpoints must record reveal state, not image bytes. For each revealed asset,
the rewindable state must include:

* `pack_id`
* expected pack version and content hash
* `asset_id`
* reveal event id or turn index
* `audience`
* visible character ids and user ids
* presentation mode
* safe caption/alt-text override if it affects what the player saw
* map fog, crop, overlay, or generated derivative ids and hashes when relevant

Discord must record every successful asset-bearing message through
`SessionMap.record_turn_message` with the same fields used for prose and dice
messages: session channel id, session id, turn index, Discord channel id,
message id, delivery kind, and recipient user id for private deliveries. An
attachment is cleaned up by deleting or hiding the message that carries it, so
asset messages must not bypass this registry.

Failure notices that are posted as Discord messages for a rewound turn should
also be recorded, because they are part of the turn's external transcript. If a
delivery failed before any message was created, there is no Discord message to
clean up, but the checkpoint reveal state still rewinds with the checkpoint.

CLI rewind restores checkpoint reveal state and map fog/crops. Any
Ayoa-managed local cache files are content-addressed delivery artifacts and do
not need per-turn deletion, but they must not be raw source files or private
extraction artifacts.

## Implementation Status And Remaining Coverage

The runtime patch now validates `asset://` delivery refs, resolves reviewed
player-safe bytes with hash and MIME checks, delivers Discord attachments
through no-public-fallback helpers, records successful asset messages for
rewind cleanup, and supports CLI display with source-path redaction.

Tests should continue to cover runtime contracts and forbidden leakage rather
than prompt wording. The core coverage is:

* delivery ref validation rejects absolute paths, relative paths, and unsafe
  schemes
* asset resolver returns bytes only for reviewed player-safe assets with
  verified hashes
* `all_observers` and `only` produce different per-viewer attachment sets
* private image delivery failure does not call `_send_public_turn_render` or
  any public channel send
* successful asset messages are recorded for rewind cleanup
* CLI delivery never prints raw source paths or hidden filenames

Remaining expansion areas include broader public/channel reveal semantics,
additional terminal backends, map overlays, fog/crop derivatives, and
approved-CDN resolver configuration beyond local pack media.
