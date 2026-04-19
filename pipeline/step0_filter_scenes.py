"""
Step 0: Filter sotopia scenes for zero-sum / negotiation scenarios.

Searches EnvironmentProfile descriptions for keywords indicating
competitive/negotiation scenarios where there's a clear winner.

Usage:
    conda activate lx_sotopia
    export REDIS_OM_URL="redis://localhost:6380"
    python step0_filter_scenes.py

Output: data/filtered_scenes.json
"""
import json
import os
import re

from sotopia.database import EnvironmentProfile

import config

# Keywords that suggest zero-sum / negotiation / competitive scenarios
KEYWORDS = [
    # English
    "negotiat", "bargain", "persuad", "convinc", "compet",
    "divide", "split", "share", "allocat",
    "price", "cost", "budget", "money", "pay", "sell", "buy",
    "win", "lose", "advantage", "disagree", "conflict",
    "demand", "refuse", "reject", "compromise",
    "limit", "scarce", "only one",
    # Goal-oriented
    "maximize", "minimize", "get more", "get the best",
    "your share", "fair", "unfair",
]


def search_scenes():
    """Search all EnvironmentProfiles for zero-sum scenarios."""
    print("Loading all EnvironmentProfiles from Redis...")
    all_profiles = EnvironmentProfile.find().all()
    print(f"Total scenes: {len(all_profiles)}")

    matched = []
    for profile in all_profiles:
        # Combine scenario text and agent goals for searching
        searchable_text = ""

        if hasattr(profile, 'scenario') and profile.scenario:
            searchable_text += profile.scenario.lower() + " "

        if hasattr(profile, 'agent_goals') and profile.agent_goals:
            for goal in profile.agent_goals:
                if goal:
                    searchable_text += goal.lower() + " "

        if not searchable_text.strip():
            continue

        # Check for keyword matches
        matched_keywords = []
        for kw in KEYWORDS:
            if kw.lower() in searchable_text:
                matched_keywords.append(kw)

        if matched_keywords:
            scene_info = {
                "pk": profile.pk,
                "scenario": profile.scenario if hasattr(profile, 'scenario') else "",
                "agent_goals": profile.agent_goals if hasattr(profile, 'agent_goals') else [],
                "matched_keywords": matched_keywords,
                "keyword_count": len(matched_keywords),
            }
            matched.append(scene_info)

    # Sort by number of keyword matches (more matches = more likely zero-sum)
    matched.sort(key=lambda x: x["keyword_count"], reverse=True)

    return matched


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    scenes = search_scenes()

    print(f"\n=== Found {len(scenes)} candidate scenes ===\n")

    # Print top matches
    for i, scene in enumerate(scenes[:20]):
        print(f"[{i+1}] pk={scene['pk']}")
        print(f"    Scenario: {scene['scenario'][:100]}...")
        if scene['agent_goals']:
            for j, goal in enumerate(scene['agent_goals']):
                if goal:
                    print(f"    Goal {j}: {goal[:80]}...")
        print(f"    Keywords: {', '.join(scene['matched_keywords'])}")
        print()

    if len(scenes) > 20:
        print(f"... and {len(scenes) - 20} more.\n")

    # Save all matched scenes
    output_path = os.path.join(config.DATA_DIR, "filtered_scenes.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(scenes)} scenes to: {output_path}")

    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Total scenes in DB: {len(EnvironmentProfile.find().all())}")
    print(f"Matched scenes: {len(scenes)}")
    print(f"Match rate: {len(scenes)/884*100:.1f}%")

    # Keyword distribution
    kw_counts = {}
    for scene in scenes:
        for kw in scene["matched_keywords"]:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
    print(f"\nTop keywords:")
    for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kw}: {count} scenes")


if __name__ == "__main__":
    main()
