import re
from pathlib import Path


PROMPTS_DIR = Path("app/prompts")

# Negative prompt contract only: do not freeze approved prompt wording here.
# This list catches implementation details the model has no need to know.
FORBIDDEN_PROMPT_PATTERNS = [
    (re.compile(r"<!--|-->"), "prompt-source comment block"),
    (re.compile(r"\bAnthropic\b", re.IGNORECASE), "provider name"),
    (re.compile(r"\bClaude\b", re.IGNORECASE), "provider/model name"),
    (re.compile(r"\bOpenAI\b", re.IGNORECASE), "provider name"),
    (re.compile(r"\bChatGPT\b", re.IGNORECASE), "provider/model name"),
    (re.compile(r"\bMessages API\b", re.IGNORECASE), "transport API"),
    (re.compile(r"\bAPI key\b", re.IGNORECASE), "secret/config detail"),
    (re.compile(r"\bSDK\b", re.IGNORECASE), "client implementation detail"),
    (re.compile(r"\bAPI clients?\b", re.IGNORECASE), "client implementation detail"),
    (re.compile(r"\bengine\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\borchestrator\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bdispatcher\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bturn loop\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bpipeline\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bcontext builder\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bdownstream\b", re.IGNORECASE), "pipeline implementation detail"),
    (re.compile(r"\bconversation history\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\brolling history\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\bcontext window\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\bprior turns\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\bprior messages in this conversation\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\bthis conversation is your memory\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\binherit (?:them|it) from prior messages\b", re.IGNORECASE), "context-window coaching"),
    (re.compile(r"\bPydantic\b", re.IGNORECASE), "Python validation detail"),
    (re.compile(r"\bpytest\b", re.IGNORECASE), "test harness detail"),
    (re.compile(r"\bunit tests?\b", re.IGNORECASE), "test harness detail"),
    (
        re.compile(
            r"\b(PromptManager|LLMClient|EngineBridge|CheckpointManager|Orchestrator)\b"
        ),
        "internal Python class name",
    ),
    (re.compile(r"\bapp/[A-Za-z0-9_./-]+"), "repo-internal file path"),
    (re.compile(r"\btests/[A-Za-z0-9_./-]+"), "repo-internal file path"),
    (re.compile(r"\.py\b"), "Python filename"),
]


def test_prompt_files_do_not_leak_implementation_details():
    failures = []

    for path in sorted(PROMPTS_DIR.rglob("*.txt")):
        text = path.read_text()
        for pattern, reason in FORBIDDEN_PROMPT_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path}:{line_no}: {match.group(0)!r} ({reason})")

    assert not failures, "Forbidden prompt internals found:\n" + "\n".join(failures)


def test_narrator_prompt_does_not_leak_routing_structures():
    text = (PROMPTS_DIR / "narrator_phase2.txt").read_text()
    forbidden = [
        "canonical_event",
        "canonical event",
        "observable_facts",
        "observable facts",
        "event_id",
        "Render Mode",
        "PARTIAL MODE",
        "Cat II",
        "v11",
        "router",
        "schema",
        "acting_character_name",
    ]

    leaks = [term for term in forbidden if term.lower() in text.lower()]
    assert not leaks, "Narrator prompt leaks routing internals: " + ", ".join(leaks)


def test_visual_novel_narrator_has_no_mechanical_page_budget_or_literal_anchor_rule():
    text = (PROMPTS_DIR / "narrator_visual_novel.txt").read_text()
    forbidden = [
        re.compile(
            r"\b(?:under|maximum|max|no more than|fewer than)\s+"
            r"\d+\s+characters?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:only|lightly)\s+(?:adjust|change)\s+punctuation\b",
            re.IGNORECASE,
        ),
    ]

    leaks = [pattern.pattern for pattern in forbidden if pattern.search(text)]
    assert not leaks, "Conflicting visual-novel narrator limits: " + ", ".join(
        leaks
    )


def test_dnd_cat_ii_prompt_does_not_receive_runtime_control_policy():
    text = (PROMPTS_DIR / "dnd_cat_ii_router.txt").read_text().lower()
    forbidden = [
        "human",
        "player-controlled",
        "player controlled",
        "player_controlled",
        "player_roll_mode",
        "agent-controlled",
        "agent controlled",
    ]

    leaks = [term for term in forbidden if term in text]
    assert not leaks, "D&D Cat II prompt leaks runtime control policy: " + ", ".join(
        leaks
    )
