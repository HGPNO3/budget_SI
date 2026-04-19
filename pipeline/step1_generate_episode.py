"""
Step 1: Generate sotopia episodes and save structured data.

Supports single run or batch mode with specific scene IDs.

Usage:
    # Single random episode
    python step1_generate_episode.py

    # Batch mode with filtered scenes
    python step1_generate_episode.py --scenes data/filtered_scenes.json --count 10

Output: data/episode_<timestamp>.json (per episode)
"""
import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime

from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

import config


def get_latest_episode_from_redis():
    """Query Redis for the most recently saved EpisodeLog."""
    episodes = EpisodeLog.find().all()
    if not episodes:
        return None
    episodes.sort(key=lambda e: e.pk, reverse=True)
    return episodes[0]


def extract_rounds(messages):
    """
    Convert raw sotopia messages into structured rounds.

    A round = one complete turn in sotopia (A speaks + B responds).
    Environment messages are stored as context but not counted as dialogue.

    Returns list of dicts:
    {
        "round_idx": int,
        "agent_messages": [{"sender": str, "content": str}, ...],
        "dialogue_text": str,  # A and B's actual speech concatenated
    }
    """
    rounds = []

    for turn_idx, turn_messages in enumerate(messages):
        agent_msgs = []
        env_context = []

        for msg in turn_messages:
            if len(msg) == 3:
                sender, receiver, content = msg
            elif len(msg) == 2:
                sender, content = msg
                receiver = ""
            else:
                continue

            if not content.strip():
                continue

            if sender == "Environment":
                env_context.append(content)
            else:
                agent_msgs.append({
                    "sender": sender,
                    "content": content,
                })

        # Build dialogue text for this round (only agent speech)
        dialogue_parts = []
        for am in agent_msgs:
            if "did nothing" not in am["content"] and "left the conversation" not in am["content"]:
                dialogue_parts.append(f"{am['sender']}: {am['content']}")

        rounds.append({
            "round_idx": turn_idx,
            "agent_messages": agent_msgs,
            "dialogue_text": "\n".join(dialogue_parts),
            "has_speech": len(dialogue_parts) > 0,
        })

    return rounds


def determine_winner(rewards, agents):
    """
    Determine winner by comparing goal scores.
    Returns (winner_key, winner_goal_score, loser_key, loser_goal_score) or Nones if tie/invalid.
    """
    if not rewards:
        return None, None, None, None

    scores = {}
    items = rewards.items() if isinstance(rewards, dict) else enumerate(rewards)
    for key, reward_entry in items:
        if isinstance(reward_entry, (list, tuple)) and len(reward_entry) == 2:
            overall_score, dims = reward_entry
            goal_score = dims.get("goal", overall_score) if isinstance(dims, dict) else overall_score
        elif isinstance(reward_entry, (int, float)):
            goal_score = reward_entry
        else:
            continue
        scores[key] = goal_score

    if len(scores) < 2:
        return None, None, None, None

    sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner_key, winner_score = sorted_agents[0]
    loser_key, loser_score = sorted_agents[1]

    # Require a gap to declare a winner (no ties)
    if winner_score <= loser_score:
        return None, None, None, None

    return winner_key, winner_score, loser_key, loser_score


def episode_to_dict(episode):
    """Convert an EpisodeLog to a clean dictionary with structured rounds."""
    rounds = extract_rounds(episode.messages)

    # Extract rewards
    rewards = {}
    if episode.rewards:
        for i, reward_list in enumerate(episode.rewards):
            agent_key = episode.agents[i] if hasattr(episode, 'agents') and episode.agents else f"agent_{i}"
            rewards[agent_key] = reward_list

    result = {
        "pk": episode.pk,
        "environment": episode.environment,
        "agents": episode.agents if hasattr(episode, 'agents') else [],
        "rounds": rounds,
        "rewards": rewards,
        "models": episode.models if hasattr(episode, 'models') else [],
        "timestamp": datetime.now().isoformat(),
    }

    # Get environment profile
    try:
        env_profile = EnvironmentProfile.get(episode.environment)
        result["scenario"] = env_profile.scenario
        result["agent_goals"] = env_profile.agent_goals if hasattr(env_profile, 'agent_goals') else []
    except Exception:
        pass

    # Get agent profiles
    try:
        agent_profiles = []
        for agent_pk in episode.agents:
            profile = AgentProfile.get(agent_pk)
            agent_profiles.append({
                "pk": profile.pk,
                "name": f"{profile.first_name} {profile.last_name}",
            })
        result["agent_profiles"] = agent_profiles
    except Exception:
        pass

    # Determine winner
    winner_key, winner_score, loser_key, loser_score = determine_winner(rewards, result.get("agents", []))
    result["winner"] = winner_key
    result["winner_score"] = winner_score
    result["loser"] = loser_key
    result["loser_score"] = loser_score
    result["has_winner"] = winner_key is not None

    # Add winner name
    if winner_key and "agent_profiles" in result:
        for p in result["agent_profiles"]:
            if p["pk"] == winner_key:
                result["winner_name"] = p["name"]
            elif loser_key and p["pk"] == loser_key:
                result["loser_name"] = p["name"]

    # Build GT text for info gain computation
    if result.get("winner_name") and result.get("agent_goals"):
        for i, agent_pk in enumerate(result.get("agents", [])):
            if agent_pk == winner_key and i < len(result["agent_goals"]):
                goal = result["agent_goals"][i]
                result["gt_text"] = f"{result['winner_name']} successfully achieved their goal: {goal}"
                break

    return result


async def run_single_episode():
    """Run one episode with random sampling."""
    await run_async_server(
        model_dict={
            "env": config.VLLM_MODEL_NAME,
            "agent1": config.VLLM_MODEL_NAME,
            "agent2": config.VLLM_MODEL_NAME,
        },
        sampler=UniformSampler(),
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", help="Path to filtered_scenes.json")
    parser.add_argument("--count", type=int, default=1, help="Number of episodes to generate")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    total = args.count
    saved = 0
    skipped = 0

    for i in range(total):
        print(f"\n{'='*60}")
        print(f"[Step 1] Episode {i+1}/{total}")
        print(f"{'='*60}")

        await run_single_episode()

        episode = get_latest_episode_from_redis()
        if episode is None:
            print("[WARN] No episode found, skipping.")
            skipped += 1
            continue

        episode_dict = episode_to_dict(episode)

        # Save
        timestamp = int(time.time())
        filepath = os.path.join(config.DATA_DIR, f"episode_{timestamp}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(episode_dict, f, ensure_ascii=False, indent=2)

        has_winner = episode_dict.get("has_winner", False)
        winner_name = episode_dict.get("winner_name", "None")
        n_rounds = len(episode_dict["rounds"])
        speech_rounds = sum(1 for r in episode_dict["rounds"] if r["has_speech"])

        status = "HAS WINNER" if has_winner else "NO WINNER (tie/fail)"
        print(f"  Saved: {filepath}")
        print(f"  Rounds: {n_rounds} total, {speech_rounds} with speech")
        print(f"  Status: {status}")
        if has_winner:
            print(f"  Winner: {winner_name} (goal={episode_dict['winner_score']})")
            print(f"  GT: {episode_dict.get('gt_text', 'N/A')[:80]}...")

        saved += 1

    print(f"\n{'='*60}")
    print(f"[Step 1] Done. Saved: {saved}, Skipped: {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
