"""One-time, source-audited extraction. Run from the active foundation checkout."""

import hashlib
import json
import re
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
SOURCE = ARCHIVE / "experiments/covenant_single_llm/proxy_trials"
TARGET = Path.cwd()
changes = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


original = (SOURCE / "candidate_41/covenant.txt").read_text()
canon = original
direction = []


def split(old, replacement, instruction, reason):
    global canon
    assert canon.count(old) == 1, old
    canon = canon.replace(old, replacement)
    if instruction:
        direction.append(instruction)
    changes.append({"source": old, "canon": replacement, "direction": instruction, "reason": reason})


campaign = re.search(r"<campaign>.*?</campaign>", canon, re.S).group()
split(campaign + "\n\n", "", campaign, "Genre and campaign direction, preserved verbatim")
tone = re.search(r"<tone_and_opening>.*?</tone_and_opening>", canon, re.S).group()
tone_body = tone.removeprefix("<tone_and_opening>\n").removesuffix("\n</tone_and_opening>")
tone_prose, opening = tone_body.split("\n\nBegin in Rowan's room", 1)
facts, instructions = opening.split("Establish\nan immediate detail", 1)
starting = "<opening_state>\nRowan starts in his room" + facts + "</opening_state>"
split(tone, starting, "<tone>\n" + tone_prose + "\n</tone>\n\n<opening>\nEstablish\nan immediate detail" + instructions + "\n</opening>", "Starting facts stay in canon; portrayal and scene-entry direction move together")
split("Let that shape choices without turning each conversation into\na lecture on the law. ", "", "<portrayal subject=\"Article Nineteen\">\nLet the law shape choices without turning each conversation into a lecture on it.\n</portrayal>", "The law and its actual limits remain unchanged in canon")
split("Do not grant him a hidden magical awakening to solve this.", "", "<portrayal subject=\"Rowan\">\nDo not grant Rowan a hidden magical awakening to solve his magical inertness.\n</portrayal>", "Magical inertness, survival, and ignorance remain explicit canon")
split("She can laugh at him, take his side, or lose\npatience without first making a speech about standards.", "She can laugh at him, take his side, or lose\npatience.", "<portrayal character=\"Ashara\">\nShe can laugh at Rashid, take his side, or lose patience without first making a speech about standards.\n</portrayal>", "Preserve emotional range and the actual two-year relationship; move corrective staging")
split("His exhaustion and his sister's importance are private, not an opening-night\nconfession waiting for a considerate question.", "His exhaustion and his sister's importance are private.", "<portrayal character=\"Rashid\">\nHis exhaustion and his sister's importance are not an opening-night confession waiting for a considerate question.\n</portrayal>", "Privacy stays a canon knowledge boundary")
split("An author asking whether she enjoyed something can receive the part\nshe liked, a complaint, or a question about what comes next.", "", "<portrayal character=\"Caelindra\">\nAn author asking whether she enjoyed something can receive the part she liked, a complaint, or a question about what comes next.\n</portrayal>", "Move the hypothetical response menu; retain tastes, resistance, loneliness, and formative education")
split("Her binding prevents direct disclosure of the crime even in verse; an ordinary\nconversation need not contain a clue.", "Her binding prevents direct disclosure of the crime even in verse.", "<portrayal character=\"Seraphel\">\nAn ordinary conversation need not contain a clue.\n</portrayal>", "Keep the binding, verse interpretation, tastes, and private knowledge unchanged")
split("In company she generally\nspeaks casually: a short answer, something she wants, a complaint, or a joke\nthat fits what just happened.", "In company she generally\nspeaks casually.", "<portrayal character=\"Thessaly\">\nIn company she generally speaks casually: a short answer, something she wants, a complaint, or a joke that fits what just happened.\n</portrayal>", "Move illustrative delivery guidance, preserving social competence and ordinary casual speech")
split("She does not interpret the table's conversation for everyone else.\n", "", "<portrayal character=\"Thessaly\">\nShe does not interpret the table's conversation for everyone else.\n</portrayal>", "Narrative participation guidance, not a biographical incident")
split("When she has no reason to join an exchange, she can\nattend to her own business without supplying an aside.", "", "<portrayal character=\"Thessaly\">\nWhen she has no reason to join an exchange, she can attend to her own business without supplying an aside.\n</portrayal>", "Preserve permission for nonparticipation outside the biography")
split("Being sincere does\nnot make him the calmest or most reasonable person in every dispute.", "", "<portrayal character=\"Aldric\">\nBeing sincere does not make him the calmest or most reasonable person in every dispute.\n</portrayal>", "Retain sincerity, defensiveness, doubts and fallibility in canon")

merged_direction = []
for section in direction:
    match = re.match(r'(<portrayal [^>]+>)\n(.*?)\n</portrayal>', section, re.S)
    previous = next((i for i, s in enumerate(merged_direction) if match and s.startswith(match[1])), None)
    if previous is None:
        merged_direction.append(section)
    else:
        merged_direction[previous] = merged_direction[previous].removesuffix("\n</portrayal>") + "\n" + match[2] + "\n</portrayal>"

for name, source_name in [("covenant", "covenant"), ("breakwater", "second_story")]:
    folder = TARGET / "stories" / name
    folder.mkdir(parents=True, exist_ok=True)
    source = (SOURCE / f"candidate_41/{source_name}.txt").read_bytes()
    if name == "covenant":
        (folder / "canon.md").write_text(canon.strip() + "\n")
        (folder / "direction.md").write_text("\n\n".join(merged_direction).strip() + "\n")
    else:
        (folder / "canon.md").write_bytes(source)
        (folder / "direction.md").write_text("<direction>\nContemporary social drama at a coastal cinema preparing to reopen. Begin with\nRowan in the upstairs room before the staff supper.\n</direction>\n")
    player = json.loads((SOURCE / f"candidate_41/{name}_player.json").read_text())
    player.pop("opening")
    (folder / "player.json").write_text(json.dumps(player, ensure_ascii=False, indent=2) + "\n")

(TARGET / "prompts").mkdir(exist_ok=True)
for target, source in [("author.txt", "system.txt"), ("revision.txt", "revision_augmented.txt")]:
    (TARGET / "prompts" / target).write_bytes((SOURCE / "candidate_43" / source).read_bytes())

audit = {
    "source_commit": "d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b",
    "covenant_changes": changes,
    "breakwater_canon_byte_identical": (TARGET / "stories/breakwater/canon.md").read_bytes() == (SOURCE / "candidate_41/second_story.txt").read_bytes(),
    "notes": [
        "No biographies were summarized. Formative history, relationships, tastes, private knowledge, abilities and starting commitments survive.",
        "Illustrative capacities which also describe actual psychology remain canon; not every conditional sentence is a writing instruction.",
        "Covenant uses restored C41, not the older C42/C43 comparison brief. No original-chat-only material was newly imported.",
        "Breakwater direction is new genre/opening guidance; its entire source brief is unchanged canon.",
        "Default player descriptions are unchanged. Old opening submissions are supplied by story direction/canon plus the new explicit first player turn.",
    ],
    "output_sha256": {str(p.relative_to(TARGET)): sha(p.read_bytes()) for folder in [TARGET / "prompts", TARGET / "stories/covenant", TARGET / "stories/breakwater"] for p in sorted(folder.iterdir())},
}
(Path(__file__).parent / "source_migration.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_splits": len(changes), "breakwater_byte_identical": audit["breakwater_canon_byte_identical"], "outputs": len(audit["output_sha256"])}))
