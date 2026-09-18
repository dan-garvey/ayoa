"""Read persisted conversations after generation; never start a model turn."""

import tempfile

from rpc import AppServer
from run import HERE, read, write


def verify():
    checkpoint = read(HERE / "checkpoint.json")
    verified = []
    with tempfile.TemporaryDirectory(prefix="ayoa-native-audit-") as workspace:
        server = AppServer(workspace)
        try:
            for sample in sorted((HERE / "samples").iterdir()):
                path = sample / "active_binding.json"
                binding = read(path if path.exists() else sample / "binding.json")
                result = server.rpc(
                    "thread/read",
                    {
                        "threadId": binding["thread_id"],
                        "includeTurns": True,
                    },
                )
                write(sample / "native_final.json", result)
                expected = list(checkpoint["native_turns"])
                for turn in range(19, 25):
                    response = read(sample / f"turn-{turn:02d}/response.json")
                    request = read(sample / f"turn-{turn:02d}/request.json")
                    assert response["status"] == "completed"
                    expected.append(
                        {
                            "turn_id": response["turn_id"],
                            "output": response["raw"],
                            "wire_text": request["input"][0]["text"],
                        }
                    )
                actual = result["thread"]["turns"]
                assert len(actual) == len(expected) == 27
                for native, saved in zip(actual, expected):
                    assert native["id"] == saved["turn_id"]
                    user = [
                        part["text"]
                        for i in native["items"]
                        if i["type"] == "userMessage"
                        for part in i["content"]
                        if part["type"] == "text"
                    ]
                    final = "\n\n".join(
                        i["text"]
                        for i in native["items"]
                        if i["type"] == "agentMessage"
                        and i.get("phase") in {None, "final_answer"}
                    )
                    assert user == [saved["wire_text"]]
                    assert final.strip() == saved["output"].strip()
                verified.append(
                    {
                        "sample": sample.name,
                        "thread_id": binding["thread_id"],
                        "native_turns": len(actual),
                    }
                )
        finally:
            server.close()
    write(
        HERE / "native_validation.json",
        {
            "all_prefixes_inputs_outputs_match": True,
            "samples": verified,
            "generations_during_verification": 0,
        },
    )
    print(
        "Verified all ten full native conversations, including resumed prefixes and six new turns each."
    )


if __name__ == "__main__":
    verify()
