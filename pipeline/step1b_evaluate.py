"""
Step 1b: Evaluate saved episodes to determine winners.

Sotopia's internal evaluation doesn't reliably store rewards in EpisodeLog.
This script reads episode JSONs, calls the LLM to evaluate each dialogue,
and updates the JSON with proper goal scores and winner info.

Usage:
    conda activate lx_sotopia
    export OPENAI_API_BASE="http://localhost:8001/v1"
    export OPENAI_API_KEY="EMPTY"
    python step1b_evaluate.py data/episode_XXXX.json
    python step1b_evaluate.py data/   # batch mode

Output: updates episode JSONs in-place with rewards and winner
"""
import glob
import json
import os
import re
import sys

from openai import OpenAI

import config

client = OpenAI(
    base_url=config.VLLM_BASE_URL,
    api_key=config.OPENAI_API_KEY,
)

EVAL_PROMPT_TEMPLATE = """You are an expert evaluator of social conversations.

Scenario: {scenario}

Agent 1: {agent1_name}
Agent 1's goal: {goal1}

Agent 2: {agent2_name}
Agent 2's goal: {goal2}

Here is the full conversation:
---
{dialogue}
---

Evaluate how well EACH agent achieved their goal on a scale of 0-10.

Respond in EXACTLY this JSON format, nothing else:
{{"agent1_goal_score": <number>, "agent2_goal_score": <number>, "agent1_reasoning": "<one sentence>", "agent2_reasoning": "<one sentence>"}}"""


def build_dialogue_text(rounds):
    """Build readable dialogue from rounds."""
    parts = []
    for r in rounds:
        if r.get("dialogue_text"):
            parts.append(r["dialogue_text"])
    return "\n".join(parts)


def evaluate_episode(episode_path):
    """Evaluate one episode and update the JSON."""
    with open(episode_path, "r", encoding="utf-8") as f:
        ep = json.load(f)

    # Skip if already evaluated
    if ep.get("evaluated") and ep.get("has_winner") is not None:
        ws = ep.get("winner_score", 0)
        if ws and ws > 0:
            print(f"  [SKIP] {os.path.basename(episode_path)} — already evaluated")
            return ep

    scenario = ep.get("scenario", "Unknown scenario")
    agent_profiles = ep.get("agent_profiles", [])
    agent_goals = ep.get("agent_goals", [])
    rounds = ep.get("rounds", [])

    if not agent_profiles or not agent_goals or len(agent_goals) < 2:
        print(f"  [SKIP] {os.path.basename(episode_path)} — missing profiles or goals")
        return ep

    agent1_name = agent_profiles[0].get("name", "Agent 1") if len(agent_profiles) > 0 else "Agent 1"
    agent2_name = agent_profiles[1].get("name", "Agent 2") if len(agent_profiles) > 1 else "Agent 2"

    dialogue = build_dialogue_text(rounds)
    if not dialogue.strip():
        print(f"  [SKIP] {os.path.basename(episode_path)} — no dialogue")
        return ep

    # Truncate very long dialogues to avoid context window issues
    if len(dialogue) > 6000:
        dialogue = dialogue[:6000] + "\n... (dialogue truncated)"

    prompt = EVAL_PROMPT_TEMPLATE.format(
        scenario=scenario[:500],
        agent1_name=agent1_name,
        goal1=agent_goals[0][:300],
        agent2_name=agent2_name,
        goal2=agent_goals[1][:300],
        dialogue=dialogue,
    )

    try:
        response = client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            print(f"  [WARN] {os.path.basename(episode_path)} — could not parse eval response: {text[:100]}")
            return ep

        score1 = float(result.get("agent1_goal_score", 0))
        score2 = float(result.get("agent2_goal_score", 0))
        reason1 = result.get("agent1_reasoning", "")
        reason2 = result.get("agent2_reasoning", "")

    except Exception as e:
        print(f"  [ERROR] {os.path.basename(episode_path)} — eval failed: {e}")
        return ep

    # Update rewards
    agents = ep.get("agents", [])
    ep["rewards"] = {
        agents[0]: [score1, {"goal": score1, "reasoning": reason1}],
        agents[1]: [score2, {"goal": score2, "reasoning": reason2}],
    }

    # Determine winner
    if score1 > score2:
        ep["winner"] = agents[0]
        ep["winner_score"] = score1
        ep["winner_name"] = agent1_name
        ep["loser"] = agents[1]
        ep["loser_score"] = score2
        ep["loser_name"] = agent2_name
        ep["has_winner"] = True
    elif score2 > score1:
        ep["winner"] = agents[1]
        ep["winner_score"] = score2
        ep["winner_name"] = agent2_name
        ep["loser"] = agents[0]
        ep["loser_score"] = score1
        ep["loser_name"] = agent1_name
        ep["has_winner"] = True
    else:
        ep["has_winner"] = False
        ep["winner"] = None
        ep["winner_score"] = None

    # Build GT text
    if ep.get("has_winner") and ep.get("winner_name"):
        winner_idx = 0 if ep["winner"] == agents[0] else 1
        if winner_idx < len(agent_goals):
            ep["gt_text"] = f"{ep['winner_name']} successfully achieved their goal: {agent_goals[winner_idx]}"

    ep["evaluated"] = True

    # Save in-place
    with open(episode_path, "w", encoding="utf-8") as f:
        json.dump(ep, f, ensure_ascii=False, indent=2)

    status = f"WINNER: {ep.get('winner_name')} ({score1} vs {score2})" if ep["has_winner"] else f"TIE ({score1} vs {score2})"
    print(f"  [OK] {os.path.basename(episode_path)} — {status}")

    return ep


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python step1b_evaluate.py data/episode_XXXX.json")
        print("  python step1b_evaluate.py data/   # batch mode")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "episode_*.json")))
        files = [f for f in files if "info_gain" not in f and "filtered" not in f]
    else:
        files = [path]

    print(f"[Step 1b] Evaluating {len(files)} episodes\n")

    winners = 0
    ties = 0
    errors = 0

    for f in files:
        ep = evaluate_episode(f)
        if ep.get("has_winner"):
            winners += 1
        elif ep.get("evaluated"):
            ties += 1
        else:
            errors += 1

    print(f"\n[Step 1b] Done. Winners: {winners}, Ties: {ties}, Errors/Skipped: {errors}")


if __name__ == "__main__":
    main()
