"""
Step 1: Generate sotopia episodes from filtered zero-sum scenarios.

Usage:
    # From filtered scenes (recommended)
    python step1_generate_episode.py --scenes data/filtered_scenes.json --count 15

    # Single random episode (fallback)
    python step1_generate_episode.py

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
    A round = one complete turn (A speaks + B responds).
    Environment messages are stored but not counted as dialogue.
    """
    rounds = []
    for turn_idx, turn_messages in enumerate(messages):
        agent_msgs = []
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
                continue
            agent_msgs.append({"sender": sender, "content": content})

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
    """Determine winner by comparing goal scores."""
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

    if winner_score <= loser_score:
        return None, None, None, None

    return winner_key, winner_score, loser_key, loser_score


def episode_to_dict(episode):
    """Convert an EpisodeLog to a clean dictionary with structured rounds."""
    rounds = extract_rounds(episode.messages)

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
            agent_profiles.append({"pk": profile.pk, "name": f"{profile.first_name} {profile.last_name}"})
        result["agent_profiles"] = agent_profiles
    except Exception:
        pass

    winner_key, winner_score, loser_key, loser_score = determine_winner(rewards, result.get("agents", []))
    result["winner"] = winner_key
    result["winner_score"] = winner_score
    result["loser"] = loser_key
    result["loser_score"] = loser_score
    result["has_winner"] = winner_key is not None

    if winner_key and "agent_profiles" in result:
        for p in result["agent_profiles"]:
            if p["pk"] == winner_key:
                result["winner_name"] = p["name"]
            elif loser_key and p["pk"] == loser_key:
                result["loser_name"] = p["name"]

    if result.get("winner_name") and result.get("agent_goals"):
        for i, agent_pk in enumerate(result.get("agents", [])):
            if agent_pk == winner_key and i < len(result["agent_goals"]):
                goal = result["agent_goals"][i]
                result["gt_text"] = f"{result['winner_name']} successfully achieved their goal: {goal}"
                break

    return result


def load_scene_ids(scenes_path):
    """Load environment PKs from filtered_scenes.json."""
    with open(scenes_path, "r") as f:
        scenes = json.load(f)
    return [s["pk"] for s in scenes]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", help="Path to filtered_scenes.json (recommended)")
    parser.add_argument("--count", type=int, default=1, help="Number of episodes to generate")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Load scene IDs if provided
    if args.scenes:
        scene_ids = load_scene_ids(args.scenes)
        random.shuffle(scene_ids)
        scene_ids = scene_ids[:args.count]
        print(f"[Step 1] Using {len(scene_ids)} scenes from {args.scenes}")
    else:
        scene_ids = [None] * args.count
        print(f"[Step 1] Random sampling (no --scenes provided)")

    total = len(scene_ids)
    saved = 0
    has_winner_count = 0

    for i, scene_id in enumerate(scene_ids):
        print(f"\n{'='*60}")
        print(f"[Step 1] Episode {i+1}/{total}")

        if scene_id:
            # Use UniformSampler with specific env_candidates
            print(f"  Scene: {scene_id}")
            sampler = UniformSampler(env_candidates=[scene_id])
        else:
            sampler = UniformSampler()

        print(f"{'='*60}")

        try:
            await run_async_server(
                model_dict={
                    "env": config.VLLM_MODEL_NAME,
                    "agent1": config.VLLM_MODEL_NAME,
                    "agent2": config.VLLM_MODEL_NAME,
                },
                sampler=sampler,
            )
        except Exception as e:
            print(f"  [ERROR] Episode failed: {e}")
            continue

        episode = get_latest_episode_from_redis()
        if episode is None:
            print("[WARN] No episode found, skipping.")
            continue

        episode_dict = episode_to_dict(episode)

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
            has_winner_count += 1

        saved += 1
        time.sleep(1)  # avoid timestamp collision

    print(f"\n{'='*60}")
    print(f"[Step 1] Done. Saved: {saved}, With winner: {has_winner_count}, No winner: {saved - has_winner_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
