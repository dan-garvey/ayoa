# 1F prompt conflict audit

Status: **PASS before generation calls**

Audited prompt hashes:

- `basic_open_air_courtyard.txt`: `c77ae40747b1a584850dc3629ad641811abc4bbf97f92c6bb20281e76c39b5ae`
- `basic_covered_pavilion.txt`: `3536423ce89030a219656e3dda81323958b5270db677e3b9550f79d92279d952`

Scope: positive instructions in `prompts/1f/basic_open_air_courtyard.txt` and `prompts/1f/basic_covered_pavilion.txt`, checked against the user-specified 1F tier and avoid contract. The existing references and 2F/3F outputs were also inspected for visual cues that must not transfer.

## Frozen invariants

- Same four PMU crops and their hashes from the original manifest.
- Same built-in `image_gen` reference-guided generation method, one call per final asset.
- Same polished webtoon rendering language, bright late-morning daylight, eye-level 16:9 framing, and broad central/lower VN sprite-staging space.
- Single intended experimental variable: architectural tier.
- Existing A/B files remain byte-identical and are reclassified only in documentation as higher-tier 2F/3F candidates.

## Positive-instruction scan

| Prompt | Potential conflict found in positive wording or references | Resolution before call |
| --- | --- | --- |
| Open-air 1F | `guild lobby` can invite heraldry, and references 01/02 include a monumental gate or tall wall. | Positive scene is explicitly small, low-rise, ordinary, and undecorated. `Reference exclusions` rejects the gate, wall scale, tracery, and people; `Avoid` rejects heraldry, banners, gold, ceremony, palace scale, and monuments. |
| Open-air 1F | `simple round arch` could drift upward in scale. | It is limited to one modest human-scale entry, paired with an alternative plain wood frame, and the composition forbids a focal monument. |
| Covered 1F | `pavilion` can imply ceremonial symmetry or a high vaulted hall; reference 04 contains ornamental arched windows. | Positive scene is a one-story utilitarian shelter with a simple pitched roof, uncarved square posts, rectangular openings, and no decorative program. `Reference exclusions` rejects ornamental windows and vaulted scale. |
| Both | `polished` could be misread as luxurious architecture. | Each prompt states that polish applies to rendering quality while architecture remains inexpensive and restrained. |
| Both | Bright, welcoming design could invite decorative fixtures or flags. | Positive scene limits detail to ordinary materials, sparse perimeter greenery, and functional posts/railings. Avoid list explicitly removes lantern arrays, flags, banners, crests, gold trim, ornate carving, and ceremonial symmetry. |

## Lexical and semantic result

- No positive instruction requests a monumental gate, cathedral tracery, tower, spire, grand wall, elaborate lantern array, banner, crest, gold trim, ceremonial symmetry, palace scale, luxury, ruin, grime, damage, or dilapidation.
- Any occurrence of those concepts in a positive-labeled line is directly negated (`no dominant focal monument`, `no grandeur`, `neither ... luxurious`) rather than requested; the remaining occurrences are in `Reference exclusions`, `Constraints`, or `Avoid` clauses.
- `polished`, `original`, and `finished` describe asset/render completeness, not architectural cost or status.
- A boundary-aware lexical scan of positive-labeled lines passed for both frozen prompt files. The scan excludes explicit `Reference exclusions`, `Constraints`, and `Avoid` clauses, then permits only directly negated occurrences such as `no dominant focal monument`.
- Result: both prompts are internally consistent with a clean, maintained, modest 1F tier and are approved for their single generation call.

## Post-generation conformance

- 1F A passed: low-rise plaster and ordinary timber, one modest human-scale arch, practical paving and railings, sparse greenery, and no people, text, banner, crest, tower, spire, tracery, gold, ceremonial feature, ruin, grime, or damage. A small diamond-shaped glazed opening in the left door reads as ordinary joinery rather than heraldry.
- 1F B passed: one-story plain timber shelter, uncarved square posts, simple pitched roof, ordinary stone floor, rectangular openings, and no people, text, banner, crest, tower, spire, grand arch, lantern array, gold, ceremonial feature, ruin, grime, or damage.
- Both preserve the bright daylight, eye-level 16:9 framing, and broad central/lower VN staging space of the higher-tier pair while materially reducing architectural status.
