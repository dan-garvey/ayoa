"""One-shot migration: split spoiler content from lore/facts into hidden_lore/hidden_facts.

Usage:
    .venv/bin/python scripts/split_lore.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT_PATH = "app/storage/saves/covenant_of_thrones/ckpt_0000.json"

# --- Facts classification ---
# Common knowledge: things the player character would know on arrival
COMMON_FACTS = [
    "The Covenant of Thrones ended the Sundering War and established a Council of Thrones with one seat per signatory race.",
    "Article Nineteen bans procreation between members of different signatory races; violation is punishable by death for parents and child.",
    "The Academy of the Covenant sits on the Nexus, a floating plateau above a perpetual storm, accessible by bridges of light from each race's territory.",
    "Magic exists for all races but each tradition has a unique cost; humans have no innate magic but can learn fragments of all traditions.",
    "Seraphel Dawnquill can only speak in verse due to a family curse that can be broken by revealing the truth before representatives of all seven races.",
]

# Hidden facts: conspiracy details to be discovered through play
HIDDEN_FACTS = [
    "Garvey House contains an abandoned wing with a bloodline-keyed ward sealing evidence of the human collapse conspiracy.",
    "The human collapse was caused by an engineered magical plague targeting the human adaptability trait, not a natural disease.",
    "The Inheritors are a secret network of descendants of the original conspirators who monitor and react to threats against their hidden agenda.",
    "Professor Kira vel Shaan is an Inheritor agent embedded in the Academy faculty.",
]

# --- Lore split ---
# Common lore: history, setting, institutions, magic — things a newcomer would learn in orientation
COMMON_LORE = """\
Three centuries ago the six major races -- demons, angels, elves, dragons, beastkin, and fae -- waged the Sundering War, a forty-year cataclysm that reshaped continents, destroyed cities, and scarred magic itself. The Covenant of Thrones, negotiated in the now-ruined neutral city of Valdris, ended the war and created a fragile peace built on the Council of Thrones, the Covenant Accords, the Academy of the Covenant, and Arbitration Protocols. The Covenant is imperfect: demons resent military limits, angels hoard institutional power, elves pursue long-term schemes, dragons dislike any sovereignty constraints, beastkin remain second-class signatories, and fae interpret obligations creatively.

Article Nineteen of the Accords bans procreation between members of different signatory races, punishable by death for both parents and the child. The law is actively enforced, yet cross-racial attraction is common and conducted in secrecy.

The Academy sits on the Nexus, a floating plateau at the intersection of all six territories, suspended above a perpetual storm and accessed by bridges of light. The campus is an architectural collage of demon black-stone halls, elven living-wood towers, angelic white-marble spires, dragon-carved caverns, and other racial styles. Roughly two thousand students study five tracks (Diplomacy, Military, Arcana, Commerce, and the Conclave). The Conclave -- about 60-80 future Council seat-holders -- live in Garvey House, a three-story mixed-style building with private rooms, a common area, a formal dining hall, seminar rooms, a restricted library, and ever-blooming gardens.

Faculty includes Chancellor Mordecai Ashworth (angel, weary neutral), Professor Vex Thorn (demon, reformist), Professor Elara Windwhisper (elf), Professor Gareth Stone (human, historian), and Professor Kira vel Shaan (demon, faculty member).

Magic exists in every race but each tradition has a distinct cost. Demon magic draws from internal life-force and ages the user; angel magic requires divine clarity; elven magic is slow and long-lasting; dragon magic is raw elemental power demanding control; beastkin magic enhances senses and physical traits; fae magic bends reality but is bound by contracts; humans have no innate magic but can learn fragments of all traditions, making them valuable translators. Magic is never free.

The "human collapse" sixty years ago was a devastating plague that wiped out ninety percent of humans and erased their political power. The cause remains officially unknown and is a subject of speculation and conspiracy theories.\
"""

# Hidden lore: the actual conspiracy, named conspirators, secret networks
HIDDEN_LORE = """\
The human collapse was not a natural plague but an engineered curse targeting the human adaptability that allowed them to learn other magics. The conspiracy was led by Dan Garvey (ancestor of the player character) and a coalition of angel, demon, elven, and dragon interests.

Dan Garvey sealed evidence of the conspiracy behind a bloodline-keyed ward in the abandoned wing of Garvey House; the ward is unknown to the Academy staff.

The Inheritors are the descendants of the original conspirators. Key members include Lord Verantus (angel seat holder, father of Aldric), Lady Ashira vel Kothren (demon heir, grandmother of Ashara), Toxicia Vaeyn (elf, mother of Caelindra), and Lady Coldpeak (dragon, aunt of Ysolde). They monitor threats through faculty agents and react with escalating measures when their secrets are probed.

Professor Kira vel Shaan is an Inheritor agent embedded in the Academy faculty. She covertly reports to the Inheritors.

Professor Elara Windwhisper covertly reports to the Silence (elven intelligence).

A side curse binds the Dawnquill angel family to speak only in verse; the curse can be broken only when the truth of its origin is spoken before representatives of all seven races.\
"""


def main():
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)

    ws = data["world_state"]

    # Replace facts
    ws["facts"] = COMMON_FACTS
    ws["hidden_facts"] = HIDDEN_FACTS

    # Replace lore
    ws["lore"] = COMMON_LORE
    ws["hidden_lore"] = HIDDEN_LORE

    # Also update the player character fact to use PLAYER_NAME consistently
    # (the personalize endpoint will replace it later)
    player_fact = "PLAYER_NAME Garvey is a human with no magical ability, the heir to the Garvey estate, and a newcomer to the Academy."
    if player_fact not in ws["facts"]:
        ws["facts"].append(player_fact)

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated {CHECKPOINT_PATH}")
    print(f"  Common facts: {len(ws['facts'])}")
    print(f"  Hidden facts: {len(ws['hidden_facts'])}")
    print(f"  Common lore: {len(ws['lore'])} chars")
    print(f"  Hidden lore: {len(ws['hidden_lore'])} chars")


if __name__ == "__main__":
    main()
