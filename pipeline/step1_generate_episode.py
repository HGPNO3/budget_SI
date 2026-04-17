"""
Step 1: Generate one sotopia episode and save structured data to JSON.

Usage:
    conda activate lx_sotopia
    export REDIS_OM_URL="redis://localhost:6380"
    export OPENAI_API_BASE="http://localhost:8001/v1"
    export OPENAI_API_KEY="EMPTY"
    python step1_generate_episode.py

Output: data/episode_<timestamp>.json
"""
import asyncio
import json
import os
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
    # Sort by pk (ULID, lexicographically ordered by time)
    episodes.sort(key=lambda e: e.pk, reverse=True)
    return episodes[0]


def episode_to_dict(episode):
    """Convert an EpisodeLog to a clean dictionary."""
    # Extract turn-by-turn messages
    turns = []
    # episode.messages is a list of list of tuples: [[(role, content), ...], ...]
    for turn_idx, turn_messages in enumerate(episode.messages):
        for role, content in turn_messages:
            if content.strip():
                turns.append({
                    "turn": turn_idx,
                    "role": role,
                    "content": content,
                })

    # Extract rewards/scores
    rewards = {}
    if episode.rewards:
        for i, reward_list in enumerate(episode.rewards):
            agent_name = f"agent_{i}"
            if hasattr(episode, 'agents') and episode.agents:
                agent_name = episode.agents[i]
            rewards[agent_name] = reward_list

    result = {
        "pk": episode.pk,
        "environment": episode.environment,
        "agents": episode.agents if hasattr(episode, 'agents') else [],
        "turns": turns,
        "rewards": rewards,
        "models": episode.models if hasattr(episode, 'models') else [],
        "timestamp": datetime.now().isoformat(),
    }

    # Try to get agent profiles and environment profile for context
    try:
        env_profile = EnvironmentProfile.get(episode.environment)
        result["scenario"] = env_profile.scenario
        result["agent_goals"] = env_profile.agent_goals if hasattr(env_profile, 'agent_goals') else []
    except Exception:
        pass

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

    return result


def determine_winner(episode_dict):
    """
    Determine which agent 'won' based on goal scores from rewards.
    Returns the winner's index and name.
    """
    rewards = episode_dict.get("rewards", {})
    if not rewards:
        return None, None

    # rewards structure varies; try to extract goal scores
    best_score = -float('inf')
    winner_key = None

    for agent_key, reward_data in rewards.items():
        # reward_data might be a list of dimension scores
        # Goal is typically the last dimension evaluated
        if isinstance(reward_data, (list, tuple)):
            # Try to get the goal score (usually the last one or a specific index)
            score = sum(reward_data) if reward_data else 0
        elif isinstance(reward_data, (int, float)):
            score = reward_data
        else:
            continue

        if score > best_score:
            best_score = score
            winner_key = agent_key

    return winner_key, best_score


async def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Count episodes before running
    try:
        episodes_before = len(EpisodeLog.find().all())
    except Exception:
        episodes_before = 0

    print(f"[Step 1] Running one sotopia episode...")
    print(f"  Model: {config.VLLM_MODEL_NAME}")
    print(f"  Episodes in Redis before: {episodes_before}")

    # Run one episode
    await run_async_server(
        model_dict={
            "env": config.VLLM_MODEL_NAME,
            "agent1": config.VLLM_MODEL_NAME,
            "agent2": config.VLLM_MODEL_NAME,
        },
        sampler=UniformSampler(),
    )

    # Retrieve the episode from Redis
    episode = get_latest_episode_from_redis()
    if episode is None:
        print("[ERROR] No episode found in Redis after running.")
        return

    episode_dict = episode_to_dict(episode)
    winner, score = determine_winner(episode_dict)
    episode_dict["winner"] = winner
    episode_dict["winner_score"] = score

    # Save to file
    timestamp = int(time.time())
    filepath = os.path.join(config.DATA_DIR, f"episode_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(episode_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[Step 1] Episode saved to: {filepath}")
    print(f"  Turns: {len(episode_dict['turns'])}")
    print(f"  Winner: {winner} (score: {score})")

    # Print a preview of the dialogue
    print("\n--- Dialogue Preview ---")
    for turn in episode_dict["turns"][:6]:
        role = turn["role"]
        content = turn["content"][:100]
        print(f"  [{turn['turn']}] {role}: {content}...")

    if len(episode_dict["turns"]) > 6:
        print(f"  ... ({len(episode_dict['turns']) - 6} more turns)")

    return filepath


if __name__ == "__main__":
    asyncio.run(main())
