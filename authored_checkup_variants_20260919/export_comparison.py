"""Export all original and new prose for a reviewer without condition or batch labels."""

import hashlib
import shutil

from run import HERE, ORIGINAL, read, write


def export():
    assert read(HERE / "validation.json")["complete"] is True
    assert read(HERE / "checkpoint.json") == read(ORIGINAL / "checkpoint.json")
    inputs = read(HERE / "inputs.json")
    assert inputs == read(ORIGINAL / "inputs.json")
    destination = HERE / "comparison"
    destination.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "review/common.md", destination / "common.md")
    sources = {}
    for label, source in read(HERE / "comparison_mapping.json").items():
        path = HERE.parent / source["experiment"] / "samples" / source["sample"]
        transcript = [f"# {label}\n"]
        for number, player in enumerate(inputs, 19):
            response = read(path / f"turn-{number:02d}/response.json")
            assert response["status"] == "completed"
            transcript.append(
                f"## Turn {number}\n\nPlayer: {player}\n\n{response['raw']}\n"
            )
        text = "\n".join(transcript)
        (destination / f"{label}.md").write_text(text)
        sources[label] = hashlib.sha256(text.encode()).hexdigest()
    write(HERE / "comparison_hashes.json", sources)
    print("Exported twenty masked six-turn continuations; no labels disclosed.")


if __name__ == "__main__":
    export()
