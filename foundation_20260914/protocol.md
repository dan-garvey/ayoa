# Foundation implementation smoke playtests

This is the approved implementation check: an opening plus three player turns
for Covenant and Breakwater, each with a fresh Terra/max author and a fresh
Terra/max editor. The runner freezes source snapshots and uses one message builder
for API and proxy requests. Author -> editor is sequential within a turn; the
two stories are independent and can proceed alongside one another.

Root chooses each next player submission from the published scene. Across the
two runs, submissions exercise initiative, observation, refusal, quiet social
interaction and explicitly delegated activity. Choices do not exploit hidden
story facts. There are no model-controlled players or extra narrative reviewers.

The model receives the complete generated role-delimited request, including rich
canon, story direction, reusable instructions and published history. The editor's
only additional writing instruction is the selected 40-word request, following
the current draft. Agents are instructed to read the entire request, write one
response artifact and return that prose. No other story or review files may be
read. Root supplies no corrective feedback or replacement outputs. All first
outputs and any technical failures remain evidence.

Root reviews complete draft/final pairs for engaging prose, distinguishable NPCs,
independent interests, plausible knowledge, player authorship, contradictions,
consequences and flexible initiative. Consistent new history, wit, affection and
quiet are permitted. These eight short published passages validate the foundation
workflow and expose local literary problems; they cannot qualify the broader
long-session research goal or establish the augmented editor's superiority.

Exact requests, agent-written outputs, public final captures, source migration
records and completed session snapshots belong on the evidence branch. Keep
private reasoning outside the audit. Record model/effort and public tool metadata
where available; do not claim direct API context, token cost or latency equivalence.
