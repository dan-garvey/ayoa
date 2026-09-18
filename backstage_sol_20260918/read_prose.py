"""Print A/B prose without showing condition settings or private checkup notes."""

import argparse
import json
from pathlib import Path

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("story", choices=("covenant", "breakwater"))
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("--lane", choices=("A", "B", "seed"))
args = parser.parse_args()
lanes = (args.lane,) if args.lane else ("seed",) if args.end <= 4 else ("A", "B")
for label in lanes:
    path = here / "sessions" / f"{args.story}_{label}" / "state.json"
    if not path.exists():
        print(f"{args.story} {label}: not initialized")
        continue
    state = json.loads(path.read_text())
    print(f"\n# {args.story} {label}\n")
    for turn in state["turns"][args.start - 1 : args.end]:
        print(
            f"## Turn {turn['turn'] + 1}\n\nPlayer: {turn['input']}\n\n{turn['output']}\n"
        )
