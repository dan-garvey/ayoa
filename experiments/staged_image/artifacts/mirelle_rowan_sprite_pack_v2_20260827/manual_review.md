# Manual review: Mirelle Voss and Rowan Kest v2

These are experimental review dispositions, not production approval or locking.
Every selected sprite was inspected at original resolution and again in its
eight-up sheet on alternating dark and light panels.

## Rubric

- **Identity** — same recognizable person as the four locked references.
- **Face and hair** — canonical facial geometry, eyes, hair shape, color, and
  label-specific expression.
- **Outfit** — canonical construction, palette, coverage, and continuity.
- **Anatomy and hands** — coherent adult proportions, limb count, joins,
  fingers, grip, and pose.
- **Weapon** — exact ownership/count and recognizable canonical construction.
- **Transparency** — real alpha; no hot-magenta rim, pale cutout halo, eaten
  fine detail, or opaque background.
- **Pose and framing** — readable body language, full silhouette, common 1100 x
  1500 canvas, consistent perceived scale/baseline, and dialogue-safe crop.
- **Semantic distinction** — the label is unambiguous and not a duplicate of
  another slot in the same sweep.

## Selected variants

| Character | Label | Identity | Face/hair | Outfit | Anatomy/hands | Weapon | Transparency | Pose/framing | Semantic distinction | Experimental disposition and notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Mirelle Voss | neutral | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Upright composed stance; canonical double-ended spear; deterministic metal repair selected. |
| Mirelle Voss | happy | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Open smile and welcoming reach. Correct upper primary end repaired in v2; wrong-end v1 preserved. |
| Mirelle Voss | concerned | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted fresh v3 candidate. Strong worried reach; complete unobstructed anatomy/garments; one coherent off-body double-ended spear. Supersedes destructive v2 repair attempts. |
| Mirelle Voss | tense | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Anxious threat anticipation is distinct from angry/concerned; tense raw v2 and rigid-streamer graft v1 preserved as rejections. |
| Mirelle Voss | skeptical | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Side-eye and guarded appraisal remain readable; canonical primary metal only. |
| Mirelle Voss | angry | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Aggressive diagonal pose; one continuous double-ended spear and exact two-hand relationship. |
| Mirelle Voss | sad | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Bowed face/body, clean upper primary repair, original gravity-aware cloth. |
| Mirelle Voss | surprised | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Startled recoil/open hand; upper-left primary end repaired and lower cap occlusion preserved. |
| Rowan Kest | neutral | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Relaxed watchfulness; exactly two mismatched knives. |
| Rowan Kest | happy | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Open smile and welcoming posture; canonical scarf/map case and paired knives. |
| Rowan Kest | concerned | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Worried reach and guarded torso; prop ownership/count stable. |
| Rowan Kest | tense | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Vigilant readiness remains distinct from anger. |
| Rowan Kest | skeptical | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Raised brow/palm-up doubt; two sheathed knife hilts remain distinct. |
| Rowan Kest | angry | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Forward confrontation; no duplicate/detached knife. |
| Rowan Kest | sad | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted v2 replacement. Lowered head, slump, and self-comforting hand; exactly two visibly distinct sheathed knives. |
| Rowan Kest | surprised | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Accepted candidate. Clear recoil and widened expression; full silhouette and prop count intact. |

## Cross-sweep findings

- Mirelle remains recognizable across all eight bases; her face, braid, cream
  coat, burgundy layers, silver armor, and spear ownership are stable.
- Concerned v3 is the only fresh fallback selected after the v2 prop-repair
  lane exposed source-hidden body pixels. It requires no deterministic metal
  graft and is normalized directly from its hash-pinned chroma source.
- The rigid metal/pose-specific cloth split avoids rotating streamers into hair
  and faces. Endpoint selection is based on weapon design and footprint rather
  than assuming that the top of the screen is always the primary end.
- Rowan's seven inherited sprites are byte-derived from the original pack's
  accepted sources; only `sad` was regenerated. The replacement fixes both the
  semantic read and the paired-knife continuity concern.
- The opaque-hybrid matte restores exact opaque subject color while retaining
  the clean physical-matte alpha and boundary RGB. Fine hair, fingers, weapon
  edges, burgundy clothing, and red tassels survive both dark and light panels.
