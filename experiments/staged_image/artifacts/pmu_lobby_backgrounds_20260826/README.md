# PMU lobby background review pack

This isolated experiment screens the 22 files whose basenames begin `PMU` in:

`C:\Users\danim\Pictures\Ayoa\ManhwaPanelDatasetReview_20260813\raw_strips`

It contains seven approved visual-novel background plates across two architectural tiers. The five reviewed 1F plates are promoted into the One-Star story as authored location guides; the ornate pair remains registered but unbound for later 2F/3F progression. Runtime LLMs receive only opaque handles and authored selection hints, never image data, paths, hashes, crop metadata, or image-derived analysis.

## Beginner-tier 1F candidates

These two files were generated in the `ayoa-nptq` extension. They reuse the frozen PMU references, method, rendering language, daylight, framing, and VN staging contract from the first run. Architectural tier is the only intended experimental variable.

| Choice | File | Size | SHA256 | Manual review |
| --- | --- | --- | --- | --- |
| 1F A | `backgrounds/1f/lobby_1f_open_air_courtyard_v1.png` | 1664x936 | `04ac3c7d7926fb783e40b979254ac77d4956f450153b37f13bb61bd10d8f228f` | Pass: plain low-rise plaster-and-timber courtyard, single modest arch, practical paving and railings, character-free and text-free, with broad calm staging space. No banner, crest, tower, spire, tracery, gold, ceremony, damage, or grime. |
| 1F B | `backgrounds/1f/lobby_1f_covered_pavilion_v1.png` | 1664x936 | `d1e014fa5dc5349334c2a4e178a6f9ed74db234a101f07f7adff5a8562256a3c` | Pass: plain one-story timber shelter with square posts, simple roof and stone floor, character-free and text-free, with broad calm staging space. No banner, crest, tower, spire, arch array, lantern array, gold, ceremony, damage, or grime. |

`review_overview_1f.jpg` shows the two new 1F plates side by side. `review_overview_tiers.jpg` places them above the byte-preserved higher-tier plates for a direct tier check. The pre-call conflict audit is in `prompt_conflict_audit_1f.md`.

## Beginner-tier 1F scene variety

The scene extension adds two purpose-specific plates without replacing the
general courtyard or pavilion.

| Scene | File | Size | SHA256 | Manual review |
| --- | --- | --- | --- | --- |
| Crack of Space and Time | `backgrounds/1f/lobby_1f_crack_of_space_and_time_v1.png` | 1664x936 | `82b568c7b8676c4122e24b5f853bc005738285c31fa5e15b14e129cca4b640ce` | Pass: matches the supplied source's segmented white/cyan seam, dark ribbed aperture, and enclosing arch; surrounding courtyard remains plain 1F construction; no people, text, UI, logo, or watermark; broad foreground remains usable. |
| Synthesis Chamber | `backgrounds/1f/lobby_1f_synthesis_chamber_v1.png` | 1664x936 | `668e6fc4c1e7ac9b79601e6024aaf97f3565901ba9cbacc0aeac9e50d1d4afc1` | Pass: clean sparse plaster/timber/stone room with one low cold-blue floor recess and barrier; no people, text, UI, console, loose props, restraints, gore, logo, or watermark; broad foreground remains usable. |

`review_overview_1f_scenes.jpg` shows all four active 1F choices. Exact
scene prompts and their pre-call conflict audit are in `prompts/1f/` and
`prompt_conflict_audit_1f_scenes.md`.

## Approved 1F Promotion Chamber

| Scene | File | Size | SHA256 | Manual review |
| --- | --- | --- | --- | --- |
| Promotion Chamber | `backgrounds/1f/lobby_1f_promotion_chamber_candidate_v1.png` | 1664x936 | `c73621400a8d9c960a38391816c3fe16f57d5e04fb00e4bc2968d7fdeb07512a` | Approved exact pixels: sparse maintained plaster, timber, and gray-stone facility; one shallow rear platform under restrained pearly light; no people, text, symbols, UI, loose props, laboratory barriers, logo, or watermark; broad foreground remains usable. |

The exact prompt is `prompts/1f/promotion_chamber_candidate_v1.txt`, and
the pre-call contradiction review is
`prompt_conflict_audit_1f_promotion_chamber.md`. Human approval promoted these
exact bytes into both One-Star seeds as `osa_loc_1f_promotion_v1`, fixed to
`niflheim_promotion_chamber`.

## Preserved higher-tier 2F/3F candidates

The original A/B files are intentionally unchanged. Their more monumental gate, tracery, towers, banners, elaborate roof framing, and lanterns make them better fits for later 2F/3F progression than for the beginner lobby.

| Choice | File | Size | SHA256 | Manual review |
| --- | --- | --- | --- | --- |
| 2F/3F A | `backgrounds/lobby_open_air_courtyard_v1.png` | 1664x936 | `5a454563d6ba57aee4cccdb69395862b9997e86098c85eb7a05abcd6bee57ac0` | Preserved byte-identically: bright outdoor lobby, character-free, text-free, wide calm sprite-staging area. Abstract banner emblems and tiny side fixtures are non-text and non-dominant. |
| 2F/3F B | `backgrounds/lobby_covered_arrival_pavilion_v1.png` | 1664x936 | `d430ccc7cbe902f0dc8144737e851b941a84ec5b01bf13506b08b3d9d70830b2` | Preserved byte-identically: bright open-sided concourse, character-free, text-free, strong central staging floor. Abstract banner emblems and hanging lamps are non-text and non-dominant. |

`review_overview.jpg` remains the original higher-tier A/B overview. The Windows review copy is at:

`C:\Users\danim\Pictures\Ayoa\PMU Lobby Background Review 20260826`

## Generation method

- Mode: built-in `image_gen`, reference-guided generation. The built-in service does not expose a model identifier.
- Calls: two in the original run, two in the architectural-tier extension, three in the scene extension, and one for the approved Promotion Chamber. The interrupted scene call still completed both outputs; its first Crack was rejected after the canonical source arrived, and one independent corrected Crack call replaced it.
- Raw tool output: 1672x941 PNG in `generation_raw/` for the preserved 2F/3F pair and `generation_raw/1f/` for the 1F pair.
- Final normalization: deterministic center crop `[4, 2, 1668, 938)` to 1664x936, an exact 16:9 ratio. There was no resampling, repainting, or other post-processing.
- Prompts: exact call text is preserved in `prompts/` and `prompts/1f/`.
- References: accepted decoded-pixel crops in `crops/accepted/`; each input was manually inspected before use.

The open-air candidate in each tier used references 01, 02, and 03. The covered candidate in each tier used references 01, 03, and 04. The Synthesis Chamber used the two approved 1F plates plus reference 03. The corrected Crack used the user-supplied canonical Crack image plus the two approved 1F plates. Reference images supplied canonical object identity, style, palette, architecture, and spatial cues only; every prompt treats them as references rather than edit targets. The positive-instruction audits and frozen prompt hashes are recorded in `prompt_conflict_audit_1f.md` and `prompt_conflict_audit_1f_scenes.md`.

## Accepted reference crops

Crop rectangles use Pillow's half-open pixel convention: `[left, top, right, bottom)`.

| Reference | Source and source SHA256 | Rectangle | Crop size and SHA256 | Why retained |
| --- | --- | --- | --- | --- |
| 01 `ref01_open_gate_plaza.png` | `PMU070_p002_d51049bebf.webp`, `d2144c90ab2c892450721ba53913b4379bd35c45247681b3a3f1d73b175aa10d` | `[0, 2354, 800, 2807)` | 800x453, `79fcdcd114382e95457eabe332012cecc9d2e3858ab3c7a71e9abc0b2ab2dd8f` | Wide plaza, gate scale, eye-level staging depth. A tiny distant group is present but does not dominate; prompts explicitly excluded people. |
| 02 `ref02_aerial_lobby_courtyard.png` | `PMU070_p006_1416c20d25.webp`, `3f8da01d4549188df33d7c0cedb8e520030c23a500db9c200d928fcff801c494` | `[0, 9932, 800, 10822)` | 800x890, `387dac115c5eab246c3fee4c26da1a3a82894f14bc46c243a0ef134f7f2499f8` | Lobby campus organization, greenery, small-building rhythm. Tiny formation figures at far left are non-dominant and were excluded by the prompt. |
| 03 `ref03_open_training_courtyard_architecture.png` | `PMU150_p007_e12a70ed75.webp`, `165b3ec2e6bf60e4dde5a53cab7a77c395b473f9491919c2e6c99780979a7eec` | `[0, 6550, 800, 7244)` | 800x694, `d6eeb072119f425627ca74204fd5bbda90ef1ff93e2fe3238cf82234185cb0bf` | Airy sky, pale perimeter architecture, simple courtyard structures. |
| 04 `ref04_bright_pavilion_exterior.png` | `PMU150_p009_bba679eca4.webp`, `b5826dc2e96081ca084eaf5d0d49cf83d59a3058d64d867d67e79378d4023c18` | `[114, 6755, 800, 7254)` | 686x499, `1740a76ea83491aea302c518af7b51d614f42dc2e8210266b6f7622d3f5d56fa` | Bright masonry, repeated arched windows, warm/cool pavilion material cue. |

The PNG crops are lossless with respect to the decoded RGB pixels of the untouched WebP sources. The source files themselves were never modified.

## Canonical Crack source reference

The user-supplied 765x1024 PNG is preserved byte-for-byte at
`source_references/crack_of_space_and_time_source.png`, SHA256
`3b85b43ed9e007c1f93bd2c441702ffac1ac89e55886beb135b165d7cdb8ba43`.
It is an authoring-time visual reference only. Its recognizable traits were
manually authored into the corrected generation prompt; no runtime model reads
the source image.

## Screening and rejections

The two `*_all_strips_overview` contact sheets preserve the full 22-file screening surface. The detailed `*_y_index` sheets add 1200-pixel y-coordinate windows for the environment-bearing strips so every retained rectangle can be independently checked against its source.

Notable rejected directions:

- `PMU070_p003_1a720001b0.webp`: monumental door sequence was dark and funerary rather than a welcoming lobby.
- `PMU070_p004_3c603807f8.webp`: blue arrival chamber contained speech bubbles and reproduced the clammy corridor mood the experiment was meant to avoid.
- `PMU070_p007_748c4d075a.webp`: empty workshop was clean but dark and dominated by barrels, crates, and benches.
- `PMU070_p008_930070ca2c.webp` and `PMU150_p008_26500f66f5.webp`: useful spatial context was substantially occupied by characters and dialogue.
- `PMU150_p010_48d70309ed.webp` and `PMU150_p011_ba8a01f6f4.webp`: drawing-room lobby scenes were bright but character-, dialogue-, and furniture-dominated.

The broad screening crops remain under `crops/screening/` as rejection and refinement evidence; only files under `crops/accepted/` were supplied to image generation.

## Production promotion

The One-Star checkpoint registers all seven exact files under its story-owned
visual-reference root. The active 1F bindings offer the general courtyard,
covered pavilion, and Crack approach for `niflheim_lobby`, the Synthesis
Chamber only for `niflheim_synthesis_chamber`, and the Crack plate alone for
its dedicated facility label. The approved Promotion Chamber is fixed to
`niflheim_promotion_chamber`. The 2F/3F pair is deliberately absent from the
active location map until lobby progression supplies an authored higher-tier
binding.

For ordinary nonfixed visual-novel locations, the director receives only the
current visible facts and opaque handles plus selection hints. Synthesis and
Promotion Chamber sections instead resolve their fixed reviewed bytes directly,
without a runtime model choosing or generating a substitute. Directly embodied
visible characters may establish the depicted location for a remote
non-depictable viewer, while indirect reports remain anchored to the viewer's
own location. No production prompt, `.env`, or runtime model receives image
bytes or source provenance.
