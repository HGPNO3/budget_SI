"""
Step 2: Compute information gain per turn.

For each turn k in the dialogue:
  - Build context = dialogue turns 1..k
  - Ask model: "Will [winner] achieve their goal?" → get predicted score
  - Info gain = predicted_score[k] - predicted_score[k-1]
  - Label turns with info_gain <= 0 as "redundant"

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


def build_dialogue_prefix(turns, up_to_index):
    """Build dialogue text from turn 0 to up_to_index (inclusive)."""
    lines = []
    for t in turns[:up_to_index + 1]:
        lines.append(f"{t['role']}: {t['content']}")
    return "\n".join(lines)


def predict_goal_score(dialogue_text, winner_name, goal_description):
    """
    Ask the model to predict the winner's goal achievement score
    given a partial dialogue.
    Returns a float score (0-10).
    """
    prompt = f"""You are an expert evaluator of social conversations.

Scenario context: Two agents are having a conversation.
{winner_name}'s goal is: {goal_description}

Here is the conversation so far:
---
{dialogue_text}
---

Based ONLY on the conversation above, how likely is {winner_name} to achieve their goal?
Rate from 0 (will definitely fail) to 10 (will definitely succeed).
Respond with ONLY a single number, nothing else."""

    try:
        response = client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Extract number from response
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 10.0)
        return 5.0  # default if parsing fails
    except Exception as e:
        print(f"  [WARN] API call failed: {e}")
        return 5.0


def compute_info_gain(episode_path):
    """Main function: compute info gain for each turn in an episode."""
    with open(episode_path, "r", encoding="utf-8") as f:
        episode = json.load(f)

    turns = episode["turns"]
    winner = episode.get("winner")
    agent_profiles = episode.get("agent_profiles", [])
    agent_goals = episode.get("agent_goals", [])

    # Determine winner's name and goal
    winner_name = "the winner"
    goal_description = "achieve their social goal"

    if agent_profiles:
        # winner is like "agent_0" or an agent pk
        for i, profile in enumerate(agent_profiles):
            if winner and (f"agent_{i}" == winner or profile.get("pk") == winner):
                winner_name = profile.get("name", winner_name)
                if agent_goals and i < len(agent_goals):
                    goal_description = agent_goals[i]
                break

    print(f"[Step 2] Computing info gain for: {episode_path}")
    print(f"  Winner: {winner_name}")
    print(f"  Goal: {goal_description[:80]}...")
    print(f"  Total turns: {len(turns)}")
    print()

    # Compute predicted score at each turn
    scores = []
    info_gains = []

    # Group turns into "rounds" — each round = one complete exchange
    # We evaluate after each individual turn (both A and B contributions matter)
    for k in range(len(turns)):
        dialogue_text = build_dialogue_prefix(turns, k)
        score = predict_goal_score(dialogue_text, winner_name, goal_description)
        scores.append(score)

        if k == 0:
            ig = 0.0  # no previous turn to compare
        else:
            ig = score - scores[k - 1]
        info_gains.append(ig)

        label = "USEFUL" if ig > 0 else ("NEUTRAL" if ig == 0 else "REDUNDANT")
        role = turns[k]["role"]
        content_preview = turns[k]["content"][:60]
        print(f"  Turn {k:2d} | {role:20s} | score={score:5.1f} | IG={ig:+6.2f} | {label:9s} | {content_preview}...")

    # Summary statistics
    useful = sum(1 for ig in info_gains if ig > 0)
    redundant = sum(1 for ig in info_gains if ig < 0)
    neutral = sum(1 for ig in info_gains if ig == 0)

    print(f"\n--- Summary ---")
    print(f"  Useful turns:    {useful}/{len(turns)}")
    print(f"  Redundant turns: {redundant}/{len(turns)}")
    print(f"  Neutral turns:   {neutral}/{len(turns)}")
    print(f"  Redundancy rate: {redundant/max(len(turns),1)*100:.1f}%")

    # Save results
    result = {
        "episode_path": episode_path,
        "winner_name": winner_name,
        "goal_description": goal_description,
        "turns": [],
        "summary": {
            "total_turns": len(turns),
            "useful": useful,
            "redundant": redundant,
            "neutral": neutral,
            "redundancy_rate": redundant / max(len(turns), 1),
        },
    }

    for k, turn in enumerate(turns):
        result["turns"].append({
            "turn_index": k,
            "role": turn["role"],
            "content": turn["content"],
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
        print("Usage: python step2_compute_info_gain.py data/episode_XXXX.json")
        print("\nAvailable episodes:")
        import glob
        for f in sorted(glob.glob(f"{config.DATA_DIR}/episode_*.json")):
            if "info_gain" not in f:
                print(f"  {f}")
        sys.exit(1)

    compute_info_gain(sys.argv[1])
