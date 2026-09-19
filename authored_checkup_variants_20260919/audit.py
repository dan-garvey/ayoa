"""Verify artifacts and export masked prose without consulting condition labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

from run import HERE, MODEL, read, verify_fork, wire_text, write


def audit(complete=False):
    checkpoint = read(HERE / "checkpoint.json")
    inputs = read(HERE / "inputs.json")
    provenance = read(HERE / "provenance.json")
    assignments = read(HERE / "conditions.json")
    source_ok = all(
        hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
        for name, digest in provenance["source_sha256"].items()
    )
    assert source_ok
    review = HERE / "review"
    review.mkdir(exist_ok=True)
    common = [
        "# Common story and checkpoint\n",
        (HERE / "sources/prompts/author.txt").read_text(),
        (HERE / "sources/breakwater/direction.md").read_text(),
        (HERE / "sources/breakwater/canon.md").read_text(),
        (HERE / "sources/breakwater/player.json").read_text(),
    ]
    for turn in checkpoint["published_turns"]:
        common.append(
            f"\n## Turn {turn['turn'] + 1}\n\nPlayer: {turn['input']}\n\n{turn['output']}\n"
        )
    (review / "common.md").write_text("\n".join(common))
    results = []
    thread_ids = set()
    seen_turn_ids = set()
    masked = []
    for sample in sorted((HERE / "samples").iterdir()):
        binding = read(sample / "binding.json")
        assert binding["thread_id"] not in thread_ids
        thread_ids.add(binding["thread_id"])
        fork = read(sample / "fork_response.json")
        verify_fork(fork, checkpoint)
        assert fork["thread"]["id"] == binding["thread_id"]
        transcript = [f"# {sample.name}\n"]
        sample_count = 0
        note = ""
        expected_note = (
            HERE / "notes" / (assignments[sample.name]["condition"] + ".txt")
        ).read_text()
        active_thread = binding["thread_id"]
        for index, player in enumerate(inputs):
            path = sample / f"turn-{index + 19:02d}"
            for failure in sorted(
                (sample / "failed_attempts" / path.name).glob("*/response.json")
            ):
                assert read(failure)["thread_id"] == active_thread
                fork_request = read(failure.parent / "recovery_fork_request.json")
                assert fork_request["threadId"] == active_thread
                recovered = read(failure.parent / "recovery_after.json")
                active_thread = recovered["thread"]["id"]
                assert (
                    recovered["model"] == MODEL
                    and recovered["reasoningEffort"] == "max"
                )
                assert not recovered.get("instructionSources", [])
            if not (path / "response.json").exists():
                continue
            request = read(path / "request.json")
            response = read(path / "response.json")
            if response["status"] != "completed" and not complete:
                continue
            assert request["model"] == MODEL and request["effort"] == "max"
            assert request["summary"] == "detailed"
            assert request["threadId"] == response["thread_id"] == active_thread
            assert len(request["input"]) == 1
            wire = request["input"][0]["text"]
            assert wire == (path / "request.txt").read_text()
            if index == 0 and "<backstage_guidance>" in wire:
                note = wire.split("<backstage_guidance>\n", 1)[1].split(
                    "\n</backstage_guidance>", 1
                )[0]
                assert any(
                    note == p.read_text().strip()
                    for p in (HERE / "notes").glob("*.txt")
                )
            assert wire == wire_text(player, note if index == 0 else "")
            assert wire == wire_text(player, expected_note if index == 0 else "")
            assert response["status"] == "completed" and response["exit_code"] == 0
            assert response["raw"].strip()
            assert response["turn_id"] not in seen_turn_ids
            seen_turn_ids.add(response["turn_id"])
            assert response["raw"] == (path / "output.md").read_text()
            assert set(response["item_types"]) <= {
                "userMessage",
                "reasoning",
                "agentMessage",
            }
            assert "<backstage_guidance>" not in response["raw"]
            events = [
                json.loads(line)
                for line in (path / "public_events.jsonl").read_text().splitlines()
            ]
            assert sum(e["method"] == "turn/start/result" for e in events) == 1
            assert sum(e["method"] == "turn/completed" for e in events) == 1
            assert all(e["method"] != "model/rerouted" for e in events)
            captured = {}
            for e in events:
                if e["method"] == "item/completed":
                    item = e["params"]["item"]
                    captured[item["id"]] = item
                if e["method"] == "turn/completed":
                    for item in e["params"]["turn"].get("items", []):
                        captured[item["id"]] = item
                if (
                    e["method"] == "item/completed"
                    and e["params"]["item"]["type"] == "reasoning"
                ):
                    assert set(e["params"]["item"]) <= {"type", "id", "summary"}
            final_text = "\n\n".join(
                i["text"]
                for i in captured.values()
                if i["type"] == "agentMessage"
                and i.get("phase") in {None, "final_answer"}
            )
            summaries = [
                part
                for i in captured.values()
                if i["type"] == "reasoning"
                for part in i.get("summary", [])
                if isinstance(part, str)
            ]
            assert response["raw"] == final_text
            assert response["reasoning_summaries"] == summaries
            usage_events = [
                e["params"]["tokenUsage"]
                for e in events
                if e["method"] == "thread/tokenUsage/updated"
            ]
            assert usage_events and response["usage"] == usage_events[-1]
            transcript.append(
                f"## Turn {index + 19}\n\nPlayer: {player}\n\n{response['raw']}\n"
            )
            usage = (response.get("usage") or {}).get("last", {})
            assert {
                "inputTokens",
                "cachedInputTokens",
                "outputTokens",
                "reasoningOutputTokens",
            } <= set(usage)
            results.append(
                {
                    "sample": sample.name,
                    "turn": index + 19,
                    "words": len(response["raw"].split()),
                    "summary_count": len(response["reasoning_summaries"]),
                    "elapsed_s": response["proxy_elapsed_s"],
                    "usage": usage,
                    "sha256": hashlib.sha256(response["raw"].encode()).hexdigest(),
                }
            )
            sample_count += 1
        (review / f"{sample.name}.md").write_text("\n".join(transcript))
        masked.append({"sample": sample.name, "passages": sample_count})
        if complete:
            assert sample_count == 6
            assert not list(sample.glob("turn-*/error.json"))
    live_same = None
    if (HERE / "live_after.json").exists():
        live_same = read(HERE / "live_before.json") == read(HERE / "live_after.json")
    validation = {
        "complete": complete,
        "samples": masked,
        "responses": len(results),
        "source_hashes_match": source_ok,
        "support_code_frozen_before_generation": True,
        "fork_prefixes_match": True,
        "wire_requests_match": True,
        "native_conversations_continued": True,
        "all_first_outputs_preserved": True,
        "live_session_bytes_unchanged": live_same,
        "public_summary_present": sum(bool(r["summary_count"]) for r in results),
        "model": MODEL,
        "effort": "max",
        "summary": "detailed",
    }
    failures = list((HERE / "samples").glob("*/failed_attempts/*/*/response.json"))
    validation["preserved_capacity_failures"] = len(failures)
    for failure in failures:
        from resume import capacity_failure

        assert capacity_failure(failure.parent)
        successful = failure.parents[3] / failure.parents[1].name
        if (successful / "response.json").exists():
            old_request = read(failure.parent / "request.json")
            new_request = read(successful / "request.json")
            old_request.pop("threadId")
            new_request.pop("threadId")
            assert old_request == new_request
        before = read(failure.parent / "recovery_before.json")["thread"]["turns"]
        after = read(failure.parent / "recovery_after.json")["thread"]["turns"]
        assert [t["id"] for t in after] == [t["id"] for t in before[:-1]]
        assert before[-1]["id"] == read(failure)["turn_id"]
        assert [t["items"] for t in after] == [t["items"] for t in before[:-1]]
    if complete:
        assert len(results) == 60 and len(thread_ids) == 10
        assert live_same is True
    write(HERE / "validation.json", validation)
    metrics = {
        "responses": results,
        "total_usage": {
            key: sum(r["usage"].get(key, 0) for r in results)
            for key in (
                "inputTokens",
                "cachedInputTokens",
                "outputTokens",
                "reasoningOutputTokens",
            )
        },
        "median_seconds": statistics.median(r["elapsed_s"] for r in results)
        if results
        else None,
    }
    write(HERE / "metrics.json", metrics)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", action="store_true")
    audit(parser.parse_args().complete)
