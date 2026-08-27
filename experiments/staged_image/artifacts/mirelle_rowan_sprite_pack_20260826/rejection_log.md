# Rejected pilot iterations

| File | Intended change | Rejection reason | Preserved value |
| --- | --- | --- | --- |
| `rejected/mirelle_voss/angry_pilot_v1.png` | Generate Mirelle `angry` with genuine transparency. | PNG is RGB; the checkerboard is painted into the pixels, so there is no alpha channel. | Strong identity, anatomy, one-spear ownership, and angry pose established the visual pilot. |
| `rejected/rowan_kest/skeptical_pilot_v1.png` | Generate Rowan `skeptical` with genuine transparency. | PNG is RGB; the checkerboard is painted into the pixels, so there is no alpha channel. | Strong identity, paired sheathed knives, and skeptical pose established the visual pilot. |
| `rejected/mirelle_voss/angry_pilot_alpha_edit_v2.png` | Remove only the checkerboard and emit alpha. | Targeted edit still returned RGB with a painted checkerboard. | Proved the failure was not repaired by repeating “genuine alpha” more explicitly. |
| `rejected/rowan_kest/skeptical_pilot_alpha_edit_v2.png` | Remove only the checkerboard and emit alpha. | Targeted edit still returned RGB with a painted checkerboard. | Proved the failure was not character- or prompt-specific. |

The next single protocol change replaced only the background request with a
uniform magenta chroma field. `generation_raw/mirelle_voss/angry_chroma_v3.png`
and `generation_raw/rowan_kest/skeptical_chroma_v3.png` succeeded as clean
intermediates; deterministic chroma removal produced the RGBA pilot candidates.
