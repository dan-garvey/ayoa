"""Validate frozen inputs and export every first result; no model calls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--source-session", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree))
    from narrative import core

    manifest = core.read_json(ROOT / "manifest.json")
    for name, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name in ("manifest.json", "state.json", "transcript.md"):
        assert (ROOT / "originals" / name).read_bytes() == (
            args.source_session / name
        ).read_bytes(), name
    for path in (ROOT / "originals" / "snapshot").iterdir():
        assert path.read_bytes() == (args.source_session / "snapshot" / path.name).read_bytes(), (
            path.name
        )
    rows = []
    transcript = [
        "# Every frozen-prefix replay\n",
        "These are independent continuations, not a new sequential story.\n",
    ]
    for cell in manifest["cells"]:
        path = ROOT / "cells" / cell["id"]
        assert (
            hashlib.sha256((path / "request.json").read_bytes()).hexdigest()
            == cell["request_sha256"]
        )
        source = core.read_json(ROOT / "originals" / cell["case"] / "request.json")
        request = core.read_json(path / "request.json")
        if cell["condition"] == "api_flat":
            expected = {k: v for k, v in source.items() if k not in ("instructions", "input")}
            expected["input"] = [{"role": "user", "content": core.render_request(source)}]
            assert request == expected
        else:
            assert request == source
            assert (path / "request.json").read_bytes() == (
                ROOT / "originals" / cell["case"] / "request.json"
            ).read_bytes()
        assert (path / "request.txt").read_text() == core.render_request(source)
        result = core.read_json(path / "result.json")
        response = (
            core.read_json(path / "response.json") if (path / "response.json").exists() else None
        )
        if result["status"] == "completed":
            output = core.response_text(response)
            assert output == (path / "output.md").read_text()
            assert core.response_summary(response) == (path / "summary.txt").read_text()
            if cell["condition"] != "codex_proxy":
                assert response["raw"]["model"] == "gpt-5.6-terra"
                assert response["raw"]["reasoning"]["effort"] == "max"
            transcript.extend(["\n## " + cell["id"] + "\n", output + "\n"])
        else:
            transcript.extend(
                ["\n## " + cell["id"] + "\n", "Failure retained: " + json.dumps(result) + "\n"]
            )
        rows.append({"cell": cell["id"], **result})
    (ROOT / "TRANSCRIPTS.md").write_text("\n".join(transcript))
    core.write_json(
        ROOT / "validation.json",
        {
            "validated_at": core.now(),
            "cells": rows,
            "frozen_sources_match": True,
            "original_session_unchanged": True,
            "all_requests_match_conditions": True,
            "all_exports_match_raw_responses": True,
        },
    )
    print(
        json.dumps(
            {
                "cells": len(rows),
                "completed": sum(r["status"] == "completed" for r in rows),
                "original_session_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
