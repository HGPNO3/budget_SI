"""
Step 2: Compute information gain per round.

A round = one complete sotopia turn (A speaks + B responds).
For each round k:
  - Build context = all dialogue from rounds 0..k
  - Measure how much this round increased P(winner achieves goal)
  - IG = predicted_score[k] - predicted_score[k-1]

Current version: simple (ask model to predict 0-10).
TODO: upgrade to IGPO token log-prob method (see Task #13).

Usage:
    conda activate lx_sotopia
    export OPENAI_API_BASE="http://localhost:8001/v1"
    export OPENAI_API_KEY="EMPTY"
    python step2_compute_info_gain.py data/episode_XXXX.json

Output: data/episode_XXXX_info_gain.json
"""
import json
import re
import sys

from openai import OpenAI

import config


client = OpenAI(
    base_url=config.VLLM_BASE_URL,
    api_key=config.OPENAI_API_KEY,
)


def build_dialogue_up_to_round(rounds, up_to_idx):
    """
    Build cumulative dialogue text from round 0 to up_to_idx (inclusive).
    Only includes rounds that have actual speech.
    """
    parts = []
    for r in rounds[:up_to_idx + 1]:
        if r["dialogue_text"]:
            parts.append(r["dialogue_text"])
    return "\n".join(parts)


def predict_goal_achievement(dialogue_text, winner_name, goal_description):
    """
    Ask the model: given the dialogue so far, how likely is the winner
    to achieve their goal? Returns a float score (0-10).

    This is the SIMPLE version. Will be replaced by IGPO token log-prob.
    """
    prompt = (
        f"You are an expert evaluator of social conversations.\n\n"
        f"{winner_name}'s goal is: {goal_description}\n\n"
        f"Conversation so far:\n---\n{dialogue_text}\n---\n\n"
        f"Based ONLY on the conversation above, how likely is {winner_name} "
        f"to achieve their goal?\n"
        f"Rate from 0 (will definitely fail) to 10 (will definitely succeed).\n"
        f"Respond with ONLY a single number, nothing else."
    )

    try:
        response = client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return min(max(float(match.group(1)), 0.0), 10.0)
        return 5.0
    except Exception as e:
        print(f"  [WARN] API call failed: {e}")
        return 5.0


def compute_info_gain(episode_path):
    """Compute info gain for each round in an episode."""
    with open(episode_path, "r", encoding="utf-8") as f:
        episode = json.load(f)

    # Check if episode has a winner
    if not episode.get("has_winner"):
        print(f"[SKIP] No winner in {episode_path}. Cannot compute IG.")
        return None

    rounds = episode["rounds"]
    winner_name = episode.get("winner_name", "Unknown")
    gt_text = episode.get("gt_text", "")

    # Extract goal description from gt_text
    goal_description = gt_text.replace(f"{winner_name} successfully achieved their goal: ", "") if gt_text else "achieve their social goal"

    print(f"[Step 2] Computing info gain for: {episode_path}")
    print(f"  Winner: {winner_name}")
    print(f"  Goal: {goal_description[:80]}...")
    print(f"  Total rounds: {len(rounds)}")

    # Filter to rounds that have actual speech
    speech_rounds = [r for r in rounds if r["has_speech"]]
    print(f"  Rounds with speech: {len(speech_rounds)}")
    print()

    # Compute predicted score at each round
    scores = []
    info_gains = []

    for k in range(len(speech_rounds)):
        dialogue_text = build_dialogue_up_to_round(speech_rounds, k)
        score = predict_goal_achievement(dialogue_text, winner_name, goal_description)
        scores.append(score)

        ig = 0.0 if k == 0 else score - scores[k - 1]
        info_gains.append(ig)

        label = "USEFUL" if ig > 0 else ("NEUTRAL" if ig == 0 else "REDUNDANT")
        round_idx = speech_rounds[k]["round_idx"]

        # Show who spoke in this round
        speakers = [m["sender"] for m in speech_rounds[k]["agent_messages"]
                     if "did nothing" not in m["content"] and "left the conversation" not in m["content"]]
        speakers_str = " & ".join(speakers) if speakers else "(no speech)"
        preview = speech_rounds[k]["dialogue_text"][:60].replace("\n", " ")

        print(f"  Round {round_idx:2d} | {speakers_str:30s} | score={score:5.1f} | IG={ig:+6.2f} | {label:9s} | {preview}...")

    # Summary
    useful = sum(1 for ig in info_gains if ig > 0)
    redundant = sum(1 for ig in info_gains if ig < 0)
    neutral = sum(1 for ig in info_gains if ig == 0)

    print(f"\n--- Summary ---")
    print(f"  Speech rounds: {len(speech_rounds)}")
    print(f"  Useful (IG > 0):    {useful}")
    print(f"  Redundant (IG < 0): {redundant}")
    print(f"  Neutral (IG = 0):   {neutral}")
    print(f"  Redundancy rate: {redundant/max(len(speech_rounds),1)*100:.1f}%")

    # Build result
    result = {
        "episode_path": episode_path,
        "winner_name": winner_name,
        "goal_description": goal_description,
        "gt_text": gt_text,
        "rounds": [],
        "summary": {
            "total_rounds": len(rounds),
            "speech_rounds": len(speech_rounds),
            "useful": useful,
            "redundant": redundant,
            "neutral": neutral,
            "redundancy_rate": redundant / max(len(speech_rounds), 1),
        },
    }

    for k, sr in enumerate(speech_rounds):
        result["rounds"].append({
            "round_idx": sr["round_idx"],
            "agent_messages": sr["agent_messages"],
            "dialogue_text": sr["dialogue_text"],
            "predicted_score": scores[k],
            "info_gain": info_gains[k],
            "label": "useful" if info_gains[k] > 0 else ("neutral" if info_gains[k] == 0 else "redundant"),
        })

    output_path = episode_path.replace(".json", "_info_gain.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[Step 2] Results saved to: {output_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        import glob
        print("Usage: python step2_compute_info_gain.py data/episode_XXXX.json")
        print("\nAvailable episodes:")
        for f in sorted(glob.glob(f"{config.DATA_DIR}/episode_*.json")):
            if "info_gain" not in f and "filtered" not in f:
                print(f"  {f}")
        sys.exit(1)

    compute_info_gain(sys.argv[1])
