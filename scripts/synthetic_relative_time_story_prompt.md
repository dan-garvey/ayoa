# Synthetic Story Prompt: The Lantern Clockhouse

Use this prompt to import or audit a small rules-neutral story that stresses
relative time, private ongoing commitments, multi-POV rendering, and off-stage
ticks.

## Premise

The Lantern Clockhouse is a coastal observatory built around five separate
work areas: the north gallery, the archive, the glasshouse, the gatehouse, and
the stair landing between them. A red harbor lantern is expected to move after
roughly thirty minutes. Several characters care about that timing, but they are
not all in the same scene.

The story is not D&D and has no mechanical rules adapter. It should use only
ordinary narrative routing: visible facts, observer lists, relative clocks,
private open commitments, and narrator POV renders.

## Starting Facts

- The session begins at relative time `0s`, just after dusk.
- The north gallery overlooks the harbor lantern.
- The stair landing is adjacent to the north gallery; speech or shutter noise
  from one can be perceived from the other when the router includes observers.
- The archive, glasshouse, and gatehouse are separate locations. Actions there
  can happen in parallel with the north gallery rather than being forced to the
  session-leading time.
- A brass key and a sealed tide ledger are visible in the north gallery.
- The clockhouse bell can be heard from every location.
- Open commitments are private routing state. They must not appear in narrator
  prose unless a later visible event reveals a surface consequence.

## Playable Characters

- Mira Vale (`mira`): a careful watch officer in the north gallery. She is
  willing to keep long silent watch for the harbor lantern.
- Theo Rusk (`theo`): a courier at the stair landing. He moves quickly and
  tends to interrupt with new information.
- Jun Park (`jun`): an archivist in the archive, tracking the tide ledger.
- Rhea Sol (`rhea`): a glasshouse technician watching instruments.
- Cal Ives (`cal`): a gatehouse keeper listening for arrivals.

## Off-Stage NPCs

- Keeper Solan (`keeper_solan`): an active NPC in the map room. He wants to
  move a silver case before the bell marks the next interval.
- Scribe Nell (`scribe_nell`): an active NPC in the lower archive. She wants to
  copy one line from the tide ledger and then hide the copy.

## Scenarios To Exercise

- A single player starts a long watch or search. The router should open a
  private commitment; the narrator should only render visible setup.
- Another player changes the same scene before that commitment resolves. The
  committed player should receive a chance to revise or continue.
- A `(continue)` response should consume the pending revision prompt only when
  it is actually routed as that committed player.
- A contested physical/social action over the brass key should open Cat II,
  then resolve at the same fictional start time as the opening event.
- While a Cat II action is waiting on a required responder, unrelated player
  actions should be rejected without mutating clocks, events, commitments, or
  the open Cat II state.
- A remote player should be able to act at their local clock even if another
  player advanced far ahead by watching or waiting.
- Multiple remote players can hold separate private commitments at the same
  time. A building-wide audible event should interrupt every committed player
  who perceives it, without letting an unrelated player clear their revision
  prompts.
- A later turn should not silently insert an event into an already-advanced
  player observer's past. If a player already acted beyond that moment, the
  run should either avoid making them an observer or surface a revision path.
- When a committed player explicitly works through the commitment's expected
  or maximum duration, the commitment should resolve rather than staying open.
- Private codes written in one remote scene should stay out of other players'
  narrator renders unless the code is deliberately communicated in-fiction.
- A deliberately scoped private-code reveal should render only to the intended
  local audience and should not leak to advanced remote players.
- A player cannot backdate a new action before that character's already
  established local clock after they have moved on.
- Explicit waits such as "wait five full minutes, then..." should be encoded
  as completed duration, not as a short unresolved commitment.
- Off-stage NPC ticks should create canonical events with only off-stage
  observers, not player-facing narrator renders.
