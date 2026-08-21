# One-Star initial input contract audit

Status: complete, offline, and based on constructed production inputs. No model was called and no live harness was run.

## Scope and artifacts

This audit follows every first-contact surface from the authored seed into the text actually shown to a player or sent to a model. It covers:

- the public story primer;
- seat guidance before a claim;
- the private dossier immediately after a claim;
- the first character-agent input for every seeded record except the intentionally blank Newcomer;
- the generated first-wave one-stars and the co-spawned Floor 2 goblins;
- the character-generation input for a tier-one Hero and a tier-zero goblin;
- three `(begin)` router constructions: Master only, Newcomer only, and all playable seats;
- the Master opening narrator input both before and after same-beat spawn records exist.

The raw corpus is content-addressed so identical cached prefixes are stored once:

- `raw_rendered_inputs.json`: 34 named surfaces backed by 39 exact message blobs.
- `surface_index.json`: audience, provenance, message roles, sizes, hashes, and the complete 15-record seed roster.

The representative claimed Newcomer name and appearance in the corpus are audit fixtures, not seed changes. Dormant character-agent renders are dry renders for contract review; runtime correctly waits until activation before dispatching them.

## Verdict

The initial-input contract is not yet coherent. The router ownership boundary and blank-Newcomer lifecycle are sound, but four different kinds of content are mixed elsewhere:

1. human play instructions;
2. facts a fictional character consciously knows;
3. model-only portrayal direction;
4. author-only or omniscient truth.

The largest immediate defect is the post-claim dossier. Its own code says `personality` is model-only and excluded, but the implementation renders it and omits `player_guidance`, the field meant for the human player. The One-Star seed then compounds this by storing ignorance, hidden implementation truth, and model imperatives in fields labeled as character knowledge or secrets.

Character generation has the inverse problem: a narrow tier-one knowledge budget is sent after a system message containing the full world lore. The generated Davan record demonstrates that the broader context wins in practice.

## Actual first-contact flow

| Moment | Consumer | Input it receives | What waits for it |
|---|---|---|---|
| Browse story | Prospective player | Public primer | Seat selection |
| Browse a seat | Prospective player | That record's `player_guidance` | Claim and any required identity authoring |
| Claim completes | Claiming player | `EngineBridge.build_character_dossier` | Begin or later player action |
| `(begin)` | Event router | Stable omniscient system prefix plus active roster, semantic opening participants, opening authority, and actor submission in the user tail | Canonical event and any spawn requests |
| Spawn requested | Character-generation model | Generic generation system prompt plus full common lore; request, tier budget, and existing identities in the user tail | Materialized `CharacterRecord` values |
| Opening render | Narrator for one viewpoint | Stable narrator rules plus visible canonical facts and viewpoint material | Accepted prose or narrator continuation |
| Routed character turn | Character agent | Shared cached agent system prefix plus identity, private state, known context, and current observations in the user tail | Public action and trailing private intent |

The opening ordering is currently wrong for generated identities: narration can occur before the orchestrator has made the generated records available. The raw corpus therefore includes both the actual pre-spawn narrator input and the intended post-spawn comparison.

## Audience and knowledge contract

| Surface | Intended audience | Allowed content | Content that must stay out |
|---|---|---|---|
| Public primer | Anyone considering the story | Premise, tone, high-level seat differences, consent-relevant themes | Plot solutions, hidden identities, character-private truth |
| Seat guidance | Anyone considering that seat | Concise affordances, limitations, and the experience promised by the seat | Agent portrayal instructions, secrets unlocked only after selection |
| Player dossier | The person who claimed the character | Human control guidance; consciously remembered backstory and world knowledge; consciously held goals, objectives, and secrets | AI portrayal direction, router-only truth, facts defined by what the character does not know |
| Character agent | One fictional character model | Stable identity, portrayal direction, consciously known/private state, witnessed canon, current local input | Omniscient lore the character lacks, controller or interface metadata |
| Character generation | Character-authoring model | Spawn request, public setting constraints, exact requested knowledge budget, existing public identities needed to avoid duplication | Lore or later-tier concepts outside the requested budget |
| Event router | Omniscient adjudicator | Stable world truth, hidden authority, active fictional roster, semantic participants, canonical history, current actor submission | Human/agent ownership, bindings, downstream UI mechanics |
| Narrator | One delivery viewpoint | Visible canonical facts, established public identities, viewpoint and rendering rules | Hidden causes, unobserved facts, unresolved generated placeholders |

The desired field meanings require no new compatibility schema:

- `player_guidance`: human-facing control contract;
- `backstory` and `known_context`: facts consciously available to the character;
- `private_state.secrets`: consciously known private truths;
- `personality`: character-agent portrayal direction only;
- router hidden lore/facts: author-only truth and adjudication authority.

## Coverage by cast category

| Category | Records reviewed | Initial surfaces | Result |
|---|---|---|---|
| Blank authored Newcomer | `one_star_newcomer` | Primer, guidance, representative post-claim dossier, Newcomer/all-seat begin | Lifecycle and model exclusion pass; dossier omits the play guidance the claimant needs |
| Master | `the_master` | Primer, guidance, dossier, agent, Master/all-seat begin, opening narrator | Guidance is a good concise control contract; dossier exposes portrayal and author-only framing instead |
| Halcyon | `halcyon_of_the_gilded_march` | Primer, guidance, dossier, agent, all-seat begin | Deep tier-six knowledge is plausible; dossier still exposes agent-only portrayal imperatives |
| System guide | `iselle_the_guide` | Agent, active router roster, opening narrator | Agent context is rich, but `known_context` and `secrets` mix knowledge, behavioral rules, and implementation truth; separate observation/timing defect remains |
| Birth one-star reserve | `renna_holt` | Dormant agent dry render; generation-avoidance roster | Knowledge tier is narrow, but hidden potential is presented as a secret she keeps despite having no visible or necessarily conscious access to it |
| Birth two-star reserves | `rowan_kest`, `liora_fen` | Dormant agent dry renders; generation-avoidance roster | Tier shape is present; authored identity is unavailable to the router at later activation |
| Birth three-star reserves | `wren_thelantern`, `mirelle_voss` | Dormant agent dry renders; generation-avoidance roster | Tier shape is present; same activation identity gap |
| Birth four-star reserves | `castor_valebrand`, `seris_nightglass` | Dormant agent dry renders; generation-avoidance roster | Tier shape is present; same activation identity gap |
| Birth five-star reserves | `soren_ironvow`, `aveline_morcant` | Dormant agent dry renders; generation-avoidance roster | Tier shape is present; same activation identity gap |
| Off-book informed Hero | `veil_the_unnumbered` | Dormant agent dry render; generation-avoidance roster | Character knowledge is plausible, but `known_context` contains model direction such as whom she must not tell |
| Persistent non-social boss | `warden_of_the_eighth` | Dormant agent dry render; generation-avoidance roster | Internally contradictory: the seed forbids intentions and human interiority while the universal agent contract requires a trailing private intent |
| Generated first wave | Davan, Petra, Tam | Tier-one generation, materialized agent inputs, opening narrator comparison | Public differentiation succeeds; knowledge boundary and first-narration identity fail |
| Co-spawned opposition | Grik, Skav, Vorg | Tier-zero generation and materialized agent inputs | Full lore is sent with no knowledge grant; making every goblin a persistent agent is separately tracked as unnecessary call and prose load |

## Findings and remediation

### 1. Player dossier projection reverses its declared audience contract

`EngineBridge.build_character_dossier` says that `personality` is authorial direction for an AI agent and must be excluded. The implementation nevertheless adds it under “How You Think & Feel.” It never adds `player_guidance`.

Observed results:

- Newcomer dossier: 497 characters / 79 words. It has the audit-authored identity, role, faction, and appearance, but not the Newcomer's System-sight and control contract.
- Master dossier: 5,292 characters / 873 words. It contains the 1,239-character agent personality and malformed “secrets” such as being a genuine novice, needing warm bodies, not knowing Heroes are stolen people, and misunderstanding Promotion.
- Halcyon dossier: 9,144 characters / 1,445 words. It exposes model direction including “Do not play him as a secret sneering villain.”

Remediation: `ayoa-31wa`, broadened by this audit to align the shared projection and the underlying field semantics across every seat.

### 2. One-Star character fields mix knowledge with authoring directives

Examples from exact agent inputs:

- Iselle's `secrets` say she is “a scripted interface persona, not a person.” That is author-only implementation truth, not necessarily something the persona consciously conceals.
- Iselle's `known_context` repeatedly says what she must never claim or invent. Those are portrayal/adjudication rules, not world knowledge.
- Renna's “secret” dormant growth potential may be hidden truth without being consciously known to Renna.
- Veil's `known_context` includes a disclosure imperative rather than only the information she knows.
- The Warden's `known_context` says it must never receive intentions, while `agent.txt` requires every agent turn to end in a private intention.

The ordinary social-character cleanup belongs in `ayoa-31wa`. The Warden exposes a separate rules-neutral ownership gap tracked by `ayoa-xruo`: persistent non-social hazards need continuity without fabricated character interiority. That issue is related to `ayoa-j0w9` and `ayoa-xvpg`.

### 3. Tiered character generation receives forbidden higher-tier context

The first-wave tier-one generation input is 28,986 characters / 4,417 words. Its 16,864-character system message includes all 9,813 characters of One-Star common lore, including synthesis, promotion, the full star-memory ladder, Moebius, and the Fade. Only afterward does the user tail say that the tier-one budget is exact and permits merely the sanctioned Master/Tower/gate framing.

The accepted Davan record proves this is not a theoretical leak. His `known_context` says the synthesis chamber is something that happens to real people, and a secret says he recognized it on arrival although no one told him.

Tier-zero co-spawn generation is worse structurally: it receives the same full lore with no knowledge grant. Its three goblins happened to remain local in this playtest, but the input contract does not enforce that outcome.

Remediation: `ayoa-28hf`. Low-cognition D&D projection remains separately scoped to `ayoa-se0`.

### 4. Dormant authored identities disappear at the router activation boundary

The turn-one roster correctly includes active records only. Later, however, a router asked to activate an authored reserve lacks the exact public record unless it is projected at that transition. The latest playtest consequently changed Rowan from mismatched knives to bow-and-blade and Renna from an ash shortbow to a rusted sword.

Remediation: `ayoa-n8jq`. This must preserve the blank Newcomer exception; an unclaimed authored slot is not an activation candidate.

### 5. The first narrator render can precede same-beat spawn identity

The actual pre-spawn narrator user message is 1,485 characters and contains `niflheim_first_summon_01`, `02`, and `03`, all flattened into rusted-blade silhouettes. The post-spawn comparison is 2,645 characters and contains Davan, Petra, Tam, and each generated loadout. The shared narrator system message is identical.

Remediation: `ayoa-sdwi`, which must await one shared generation result without prematurely committing it on narrator continuation or failure.

### 6. Iselle has authored expertise but not the observation channel needed to use it on time

Iselle's initial agent input is 21,738 characters / 3,544 words and includes broad lobby-management tutorial goals. That prompt does not solve the playtest failure: the build and first use of the Synthesis Chamber did not make Iselle an observer, so her overview arrived only later and the router authored her speech without an Iselle intention.

Remediation: `ayoa-k3xp` for a fictional System observation capability and timely exactly-once tutorial dispatch; `ayoa-xvpg` for the generic persistent-character ownership guard.

### 7. Master interface choices are described, but consequential transitions need embodied response stages

The Master guidance correctly enumerates lobby and pre-deployment choices. The seed and runtime still need to distinguish selecting a Hero from physically moving that Hero into a chamber or gate. Otherwise a UI request can become irreversible movement or culling before the selected persistent character has an intention.

Remediation: `ayoa-6r3g` for physical, interruptible gate and chamber transitions; `ayoa-wgdc` for one-source-or-more synthesis and birth-star-safe premium summon authority.

### 8. The One-Star router prefix repeats story authority at excessive scale

All three begin variants share the same 107,697-character system message and differ only in semantic user-tail participants. That is good cache and ownership behavior, but the prefix itself contains overlapping authority:

- common lore: 9,813 source characters;
- public facts: 35 entries totaling 12,698 source characters;
- hidden lore: 20,435 characters;
- hidden facts: 41 entries totaling 13,778 source characters;
- additional narrative and character-local restatements of the same mechanics.

The tower-exit law alone appears four times in the rendered router system message. Similar overlap exists for summon pools, synthesis, promotion, lethality, System visibility, and Master mission control. Cache reads reduce repeated billing; they do not remove model attention cost, cold-call latency, ambiguity, or maintenance drift.

Remediation: `ayoa-shtx`, using one authoritative source per semantic rule and a smaller deterministic router projection rather than another summary layer.

### 9. Existing tests preserve prose duplication instead of audience behavior

`tests/test_one_star_ascension_checkpoint.py` contains many positive checks for required phrases, word counts, and repeated copies. One test constructs a supposed player contract by concatenating `player_guidance`, `personality`, and `known_context`, directly encoding the audience mistake found here.

Remediation: `ayoa-bgzw`, replacing wording freezes with rendered placement, forbidden leakage, schema, and observable behavior tests while preserving structural seed invariants.

### 10. Co-spawned goblins demonstrate the persistent-minor-opposition cost

The goblin generation and three individual character-agent inputs are internally well-formed, but they consume generation, identity context, and sequential agent turns for disposable scene opposition. This is not a first-input knowledge contradiction by itself; it is evidence for the already-open rules-neutral ownership/performance issue.

Remediation: `ayoa-j0w9`. The new `ayoa-xruo` keeps the separate case of named persistent hazards from being forced back into a social character agent.

## Confirmed non-findings

### Blank Newcomer lifecycle and newcomer-first opening branch

The unclaimed `one_star_newcomer` remains dormant at `not_yet_fictional`, has no fallback identity or private fields, is absent from the active router roster and generation-avoidance roster, and is rejected by character-agent and perception dispatch. Once claimed, the chosen name and appearance enter only through the semantic opening-participant block. If the Newcomer is among multiple claimed seats, the existing Newcomer is activated and generated substitutes are suppressed. This is the current behavior to preserve.

### Router controller agnosticism

The Master-only, Newcomer-only, and all-seat begin system messages are byte-identical with SHA-256 `cfbe9211d153aa98358336c7522b6eef86aab02cd44cc7dd704eea8f9468f985`. Their user tails differ by fictional actor and exact semantic opening participants, not by human, agent, binding, or controller metadata. Controller-like words found in the system prefix are fictional uses such as human species descriptions, “new player” for the in-world Master premise, or unrelated words such as “star-bound.”

### Agent cache placement

Every character-agent render shares the same 13,828-character system message. Character identity, private state, current location, and observations live in the user tail. The main agent-input defect is semantic field content, not volatile data contaminating the cached prefix.

### Public primer

The 263-word primer gives the three perspectives, their differing leverage, and the story's synthesis horror without revealing Moebius, the Fade, or plot solutions. It is doing public-pitch work rather than attempting to be a player dossier.

## Recommended implementation order

1. `ayoa-31wa`: correct the generic dossier projection and One-Star field audiences first, because every other review depends on knowing which field is authoritative.
2. `ayoa-28hf`: stop sending knowledge a generated character is forbidden to know.
3. `ayoa-n8jq` and `ayoa-sdwi`: preserve exact identity through activation, generation, and first narration.
4. `ayoa-k3xp` and `ayoa-xvpg`: give Iselle the right fictional observation channel and enforce persistent-character action ownership.
5. `ayoa-6r3g` and `ayoa-wgdc`: make consequential Master choices embodied and align state authority for summons and synthesis.
6. `ayoa-j0w9` and `ayoa-xruo`: distinguish disposable scene forces, persistent hazards, and social characters.
7. `ayoa-shtx` and `ayoa-bgzw`: collapse duplicated authority and replace prose freezes after the semantic owners are settled.

## Validation boundaries

This audit used production builders and exact stored checkpoints but deliberately performed zero model calls. It does not claim that a dry-rendered agent had an empty inbox in actual play, nor that every dormant reserve is immediately dispatchable. It establishes what each model would receive at its first eligible call and what each player sees at first contact. Live behavior remains the responsibility of each remediation issue; no live harness was required or run here.
