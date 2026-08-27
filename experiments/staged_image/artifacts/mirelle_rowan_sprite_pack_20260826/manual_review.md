# Manual sprite review

Ratings: **P** = pass, **W** = warning/review choice, **F** = fail and regenerate.
The review is manual; no runtime or authoring-time vision model supplied these
judgments.

| Character | Variant | Identity | Face | Hair | Outfit | Anatomy / hands | Weapon | Alpha / edges | Pose legibility | Framing / baseline | Semantic distinction | Disposition | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mirelle | neutral | P | P | P | P | P | P | P | P | P | P | candidate | Relaxed spear-planted captain silhouette; useful baseline. |
| Mirelle | happy | P | P | P | P | P | P | P | P | P | P | candidate | Open palm, forward energy, and broad smile are clear without caricature. |
| Mirelle | concerned | P | P | P | W | P | P | P | P | P | P | review | Feather clasp and coat construction follow the facial reference more strongly than the neutral sprite. |
| Mirelle | tense | P | W | P | W | P | P | P | P | P | W | review | Guard is excellent; face can read determined, and the short upper coat, clasp, chest tassel, and stitching drift from neutral. |
| Mirelle | skeptical | P | P | P | P | P | P | P | P | P | P | candidate | Raised brow, hip-set weight, crossed forearm, and planted spear read clearly. |
| Mirelle | angry | P | P | P | P | P | P | P | P | P | P | candidate | Strong forward brace and two-handed single-spear guard; no weapon duplication. |
| Mirelle | sad | P | P | P | P | P | P | P | P | P | P | candidate | Bowed head, collapsed shoulders, narrow stance, and hands stacked on spear are coherent. |
| Mirelle | surprised | P | P | P | P | P | P | P | P | P | P | candidate | Recoil, open hand, widened face, and controlled tilted spear are distinct. |
| Rowan | neutral | P | P | P | P | P | P | P | P | P | P | candidate | Clean baseline with two distinct sheathed knife hilts. |
| Rowan | happy | P | P | P | P | P | P | P | P | P | P | candidate | Open joking gesture and broad smile remain recognizably Rowan. |
| Rowan | concerned | P | P | P | P | P | P | P | P | P | P | candidate | Smile disappears; reaching posture and paired sheaths remain clear. |
| Rowan | tense | P | P | P | P | P | P | P | P | P | P | candidate | Exactly one broad knife drawn and one narrow knife visibly sheathed; compact ready stance. |
| Rowan | skeptical | P | P | P | P | P | P | P | P | P | P | candidate | Head tilt, raised brow, palm-up question, and asymmetric weight work together. |
| Rowan | angry | P | P | P | P | P | P | P | P | P | P | candidate | Exactly two visibly mismatched knives drawn; no duplicate sheathed hilts. |
| Rowan | sad | P | P | P | P | P | F | P | P | P | P | regenerate | Exactly two weapons remain, but their exposed pointed pommels do not read as canonical sheathed knife hilts. |
| Rowan | surprised | P | P | P | P | P | P | P | P | P | P | candidate | Recoil and two open hands are clear; exactly two knife hilts remain sheathed. |

## Structural gate

All sixteen normalized files are 1100 x 1500 RGBA images with both fully
transparent and fully opaque pixels. Every file has one significant connected
foreground component at alpha 64 or greater. Manual inspection found exactly
one character, no environment, no text/logo/watermark, no duplicate person, and
no extra weapon in every candidate.

## Pack-level judgment

This is a successful proof of pose-plus-expression generation, not an approved
production pack. Thirteen variants are clean candidates, two Mirelle variants
need a costume-authority choice, Mirelle `tense` has an additional semantic
warning, and Rowan `sad` should be regenerated. Human review should select the
preferred costume continuity before any second-pass correction.
