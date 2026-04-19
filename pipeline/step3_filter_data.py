"""
Step 3: Filter out redundant rounds to create training data.

Input: info_gain JSON from step2
Output: filtered dialogue with redundant rounds removed

Usage:
    python step3_filter_data.py data/episode_XXXX_info_gain.json
    python step3_filter_data.py data/  # batch mode: process all info_gain files in dir
"""
import json
import sys
import os
import glob


def filter_one_episode(info_gain_path):
    """Filter redundant rounds from one episode."""
    with open(info_gain_path, "r") as f:
        data = json.load(f)

    rounds = data["rounds"]
    winner = data["winner_name"]

    # Separate useful/neutral vs redundant
    kept = [r for r in rounds if r["label"] != "redundant"]
    removed = [r for r in rounds if r["label"] == "redundant"]

    reduction = (1 - len(kept) / max(len(rounds), 1)) * 100

    print(f"  {os.path.basename(info_gain_path)}: {len(rounds)} → {len(kept)} rounds ({reduction:.0f}% reduced)")

    # Build training-ready filtered dialogue
    filtered_dialogue = []
    for r in kept:
        filtered_dialogue.append({
            "round_idx": r["round_idx"],
            "dialogue_text": r["dialogue_text"],
            "info_gain": r["info_gain"],
            "agent_messages": r["agent_messages"],
        })

    output = {
        "winner_name": winner,
        "goal_description": data["goal_description"],
        "gt_text": data.get("gt_text", ""),
        "original_round_count": len(rounds),
        "filtered_round_count": len(kept),
        "reduction_rate": reduction / 100,
        "filtered_dialogue": filtered_dialogue,
        "removed_rounds": [
            {"round_idx": r["round_idx"], "dialogue_text": r["dialogue_text"], "info_gain": r["info_gain"]}
            for r in removed
        ],
    }

    output_path = info_gain_path.replace("_info_gain.json", "_filtered.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python step3_filter_data.py data/episode_XXXX_info_gain.json")
        print("  python step3_filter_data.py data/  # batch mode")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_info_gain.json")))
        print(f"[Step 3] Batch filtering {len(files)} episodes\n")
    else:
        files = [path]
        print(f"[Step 3] Filtering 1 episode\n")

    total_original = 0
    total_filtered = 0

    for f in files:
        result = filter_one_episode(f)
        total_original += result["original_round_count"]
        total_filtered += result["filtered_round_count"]

    if len(files) > 1:
        reduction = (1 - total_filtered / max(total_original, 1)) * 100
        print(f"\n--- Batch Summary ---")
        print(f"  Episodes: {len(files)}")
        print(f"  Total rounds: {total_original} → {total_filtered} ({reduction:.1f}% reduced)")


if __name__ == "__main__":
    main()
