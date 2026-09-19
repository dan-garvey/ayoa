import json

from resume import capacity_failure


def test_only_confirmed_capacity_failures_without_model_output_are_retryable(tmp_path):
    response = {"status": "failed", "raw": "", "item_types": ["userMessage"]}
    event = {
        "method": "error",
        "params": {"error": {"codexErrorInfo": "serverOverloaded"}},
    }
    (tmp_path / "public_events.jsonl").write_text(json.dumps(event) + "\n")
    path = tmp_path / "response.json"
    path.write_text(json.dumps(response))
    assert capacity_failure(tmp_path)
    response["raw"] = "A partial passage."
    path.write_text(json.dumps(response))
    assert not capacity_failure(tmp_path)
    response["raw"] = ""
    response["item_types"].append("reasoning")
    path.write_text(json.dumps(response))
    assert not capacity_failure(tmp_path)
    response["item_types"] = ["userMessage"]
    path.write_text(json.dumps(response))
    event["params"]["error"]["codexErrorInfo"] = "unknown"
    (tmp_path / "public_events.jsonl").write_text(json.dumps(event) + "\n")
    assert not capacity_failure(tmp_path)
