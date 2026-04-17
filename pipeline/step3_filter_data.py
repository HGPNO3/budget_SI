"""
Step 3: Filter out redundant turns to create "budget" training data.

Input: info_gain JSON from step2
Output: filtered dialogue with redundant turns removed

This is the data you'd feed into RL training:
- Original dialogue = baseline
- Filtered dialogue = "budget" version (same outcome, fewer turns)

Usage:
    python step3_filter_data.py data/episode_XXXX_info_gain.json
"""
import json
import sys


def filter_redundant_turns(info_gain_path):
    with open(info_gain_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    turns = data["turns"]
    winner = data["winner_name"]

    print(f"[Step 3] Filtering redundant turns")
    print(f"  Winner: {winner}")
    print(f"  Original turns: {len(turns)}")
    print()

    # Original dialogue
    print("=== ORIGINAL DIALOGUE ===")
    for t in turns:
        marker = " " if t["label"] == "useful" else "X" if t["label"] == "redundant" else "~"
        print(f"  [{marker}] {t['role']}: {t['content'][:80]}...")
    print()

    # Filtered dialogue (remove redundant, keep useful + neutral)
    filtered = [t for t in turns if t["label"] != "redundant"]

    print(f"=== FILTERED DIALOGUE ({len(filtered)} turns) ===")
    for t in filtered:
        print(f"  [+] {t['role']}: {t['content'][:80]}...")
    print()

    reduction = (1 - len(filtered) / max(len(turns), 1)) * 100
    print(f"--- Result ---")
    print(f"  Original: {len(turns)} turns")
    print(f"  Filtered: {len(filtered)} turns")
    print(f"  Reduction: {reduction:.1f}%")

    # Save filtered version
    output = {
        "winner_name": winner,
        "goal_description": data["goal_description"],
        "original_turn_count": len(turns),
        "filtered_turn_count": len(filtered),
        "reduction_rate": reduction / 100,
        "original_dialogue": [
            {"role": t["role"], "content": t["content"], "label": t["label"]}
            for t in turns
        ],
        "filtered_dialogue": [
            {"role": t["role"], "content": t["content"]}
            for t in filtered
        ],
    }

    output_path = info_gain_path.replace("_info_gain.json", "_filtered.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Saved to: {output_path}")
    print()
    print("Next step: use filtered_dialogue as 'better' training data for RL.")
    print("The model should learn to produce dialogues that skip redundant turns.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        import glob
        print("Usage: python step3_filter_data.py data/episode_XXXX_info_gain.json")
        print("\nAvailable info_gain files:")
        for f in sorted(glob.glob("data/*_info_gain.json")):
            print(f"  {f}")
        sys.exit(1)

    filter_redundant_turns(sys.argv[1])
