# One-Star visual-novel Luna playtest

Date: 2026-08-26 (America/Chicago)

Session: `one-star-vn-luna-playtest-20260826`

Repository revision: `f483281`

## Outcome

The ADV card renderer itself is usable: text is legible, dialogue receives a clear
speaker plate, narration does not, cards advance reliably, and the accessible
transcript matches the rendered pages. The production visual experience is not
ready, however. It produced one dark foggy lobby plate, reused it for one passage,
then cleared the stage for every remaining passage. Thirteen of nineteen cards
therefore showed only the neutral navy background.

The narrator needs targeted tuning, but it was not the main source of failure.
It faithfully preserved the canonical events across all seven passages. Its clear
defects were an unfinished sentence split across two cards, robotic expansion of
direct address into repeated full names, and overly frequent handoffs to a Master
who had no meaningful response available. Iselle's repeated `Acknowledged!`
register came from the character agent, not the narrator.

## Configuration and coverage

The Master was the only claimed seat, leaving every embodied character agent-owned.
The run used the real CLI/EngineBridge, durable image queue, remote FLUX.2-dev
gateway, production VN stage director, and production card compositor.

Configured model roles:

| Role | Model |
| --- | --- |
| `agent` | `openai:gpt-5.6-luna` |
| `agent_standard` | `openai:gpt-5.6-luna` |
| `agent_convenience` | `openai:gpt-5.6-luna` |
| `character_manager` | `openai:gpt-5.6-luna` |
| `event_router` | `openai:gpt-5.6-terra` |
| `narrator` | `openai:gpt-5.6-terra` |
| `image_director` | `openai:gpt-5-mini` |

All three character-generation calls, all Iselle calls, and all Mara/Tovin calls
logged `gpt-5.6-luna`. The default `agent` tier was configured as Luna but was not
selected by this cast.

Player inputs were:

1. `/join 2`
2. `/settings presentation_mode visual_novel`
3. `/begin --confirm`
4. Read-only `/status`, `/master status`, and `/history 3`
5. Six `/defer` commands, followed by `/quit`

This produced seven accepted VN decks containing nineteen cards and seven durable
director decisions: one `replace`, one `reuse`, and five `clear` transitions. No
runtime error or image-job failure occurred.

## Deck ledger

| Update / event | Director | Cards | Deck id | Visual result |
| --- | --- | ---: | --- | --- |
| 2 / `event_opening_0001` | replace | 2 | `29030801c72e63e143849430a4b0adc0950c376b7f36a77ccc71f6550e3b055f` | generated lobby |
| 3 / `event_0002` | reuse | 4 | `3960db97c365dcb9f23167afb689890d2bc9d331e3d640ebacbc689afb1aa9bf` | same lobby |
| 4 / `event_0003` | clear | 2 | `28c31fabf23fd012ea728cffd2304b9b9eb5c97e8d6a0104c4f9849fce5792a9` | neutral stage |
| 5 / `event_0004` | clear | 2 | `c0240d8b7f69de7ec37eecd6852e4251da9ec80aae2a4c3c3907211c15c77840` | neutral stage |
| 6 / `event_0005` | clear | 3 | `67d594c282c80f35a467e69c44d056d3ed5fe8b80c403bb4d78a23a0999b5d88` | neutral stage |
| 7 / `event_0006` | clear | 4 | `dbae6a0bdf489d26d332660531635088a30f4c707f9510d38fd8a7328bd3cd60` | neutral stage |
| 8 / `event_0007` | clear | 2 | `4cd4828460b9d207cb46c3264fc4ffeb7492bf9694f8768e415821a1f6cd6325` | neutral stage |

The first two manifests use stage hash
`b2a662f706f5188bb0b567dc6b692feb2096ad1d9994c0af2e6f9b265b2d4aad`.
The other five use the compositor's neutral-stage hash
`0b2ecf51d18e9fad0d262b8eafa899e85df59e83d6e3d82f1e94c18e60558238`.

## Feedback by source layer

### P1: stage and identity lifecycle

The opening director request explicitly asked for a cold stone summoning lobby with
`tall, shadowed pillars` and `low fog`. The resulting image is technically coherent,
but it reproduces the rejected clammy-hallway direction instead of the reviewed
outdoor beginner lobby. It also depicts Iselle far too large relative to the three
human silhouettes.

The stage was reused for Iselle's introduction, which proves the stage hash and
card composition path work. That reuse also proves the current implementation is
only a frozen full plate: Iselle keeps the same smile and pose while narration says
she bows, points at the gate, and changes expression. The referenced sealed gate is
not visible on the plate.

When Mara first speaks, the director sees an unanchored generated character and
chooses `clear`. The VN path did not create an individual identity anchor for any of
the three generated recurring Heroes. Later director projections expose no
selectable reference options, even for Iselle, and five consecutive events remain
clear. This is why the conversation disappears rather than a diffusion failure.

The reviewed staged-background/sprite work from `ayoa-cnxp` and `ayoa-ydgt` has not
been promoted into the production lifecycle. Follow-up: `ayoa-yvja`.

### P2: narrator page contract

Deck `67d594c...` contains a narrator-authored boundary failure:

> Page 1: `...her knife stays tight in her`
>
> Page 2: `other hand.`

The raw narrator conversation already contains two pages, so the compositor did not
cause the split. The preferred 360-character target appears to have won over the
semantic requirement that one card contain a readable beat.

In `event_0004`, Iselle's agent says `I can't offer you another way out ... but you
may inspect`. Router canonicalization safely binds both references to Mara. The
narrator then renders `Mara Pell` twice in one spoken line. It leaks no id and keeps
the target correct, but it sounds unlike direct speech.

Follow-up: `ayoa-fhm6`.

### P2: spectator handoff pacing

After the opening, the Master had to enter `/defer` six times. Every accepted event
already had an owed agent responder and the Master could neither answer the
embodied dialogue nor control anyone's body. The narrator nevertheless rendered
after each single event, so the player alternated between a command and one short
two-to-four-card update.

Frequent interruption is appropriate for an embodied player who can answer. A
spectator-only feed needs a bounded autonomous continuation rule that stops at a
real management choice or meaningful interruption point, not after every routine
answer. This belongs in the same narrator/handoff follow-up, with the existing
runaway-NPC-dialogue guard preserved.

### P3: Iselle voice is upstream of narration

`Acknowledged!`, `Inspection acknowledged! No following.`, and the account-style
roster report are present verbatim in Iselle's Luna agent outputs. The router and
narrator preserve them. If that voice is undesirable, tune Iselle's story-specific
character context or agent behavior; weakening narrator faithfulness would hide the
source and risk changing other characters' dialogue.

### Performance observation

The cold `/begin` took about 5 minutes 39 seconds from command to first deck. The
single image job occupied about 4 minutes 22 seconds of that interval. This run
started the remote service and loaded FLUX immediately beforehand, so it is a cold
start measurement, not representative steady-state latency. Subsequent no-generation
turns took roughly 13-21 seconds before their decks appeared. The blocking CLI
experience is already tracked by `ayoa-ke4k`.

## What worked

- All observed character and character-generation calls used Luna.
- Router facts remained the authoritative source; narrator pages did not invent
  hidden thought, unseen action, or Master dialogue.
- Owed responders were preserved. Tovin eventually spoke after Mara and Iselle;
  the apparent delay was not responder starvation.
- Speaker nameplates, narration cards, line wrapping, page indices, transcript
  projection, and keyboard navigation worked.
- The opening stage was reused byte-for-byte across a second deck.
- The image job completed with the pinned FLUX.2-dev model and no fallback.
- No source identifiers appeared on cards.

## Preserved artifacts

Raw local evidence (intentionally ignored by Git):

- `app/storage/playtest_reports/one-star-vn-luna-playtest-20260826/transcript.tty`
- `app/storage/playtest_reports/one-star-vn-luna-playtest-20260826/transcript.clean.txt`
- `app/storage/sessions/one-star-vn-luna-playtest-20260826/ckpt_0000.json` through
  `ckpt_0008.json`
- `app/storage/runtime/visual_novel_presentation/decks/<deck-id>/`
- `app/storage/runtime/image_generation/artifacts/b2/b2a662f706f5188bb0b567dc6b692feb2096ad1d9994c0af2e6f9b265b2d4aad.webp`
- `app/storage/runtime/image_generation/jobs.sqlite` (director/job provenance)
- `logs/play_cli.log`

The seven deck folders, clean transcript, and this report are also copied into
`C:\Users\danim\Pictures\Ayoa\Visual Novel Dialogue Proofs\one-star-vn-luna-playtest-20260826`
for review in Windows.
