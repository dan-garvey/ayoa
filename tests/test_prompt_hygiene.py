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
