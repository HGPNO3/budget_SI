"""
Step 0: Filter sotopia scenes for zero-sum / competitive scenarios.

Three-layer filtering:
  Layer 1: codename pattern matching (craigslist = competitive, mutual = cooperative)
  Layer 2: goal conflict detection (opposing intents in agent_goals)
  Layer 3: exclude known cooperative patterns

Usage:
    conda activate lx_sotopia
    export REDIS_OM_URL="redis://localhost:6380"

    # First run: diagnostic mode — see codename/source distribution
    python step0_filter_scenes.py --diagnose

    # Second run: actual filtering
    python step0_filter_scenes.py

Output: data/filtered_scenes.json
"""
import argparse
import json
import os
import re
from collections import Counter

from sotopia.database import EnvironmentProfile

import config


# Known competitive codename patterns (from sotopia benchmark.py)
COMPETITIVE_CODENAMES = [
    "craigslist_bargains",
    "craigslist",
    "borrow_money",
]

# Known cooperative codename patterns (exclude these)
COOPERATIVE_CODENAMES = [
    "mutual_friend",
    "mutual",
]

# Goal conflict indicators: if agent_goals[0] has word from column A
# AND agent_goals[1] has word from column B → likely zero-sum
CONFLICT_PAIRS = [
    # (agent A intent, agent B intent)
    (["sell", "seller", "maximize", "highest", "raise", "increase"],
     ["buy", "buyer", "minimize", "lowest", "reduce", "decrease"]),
    (["buy", "buyer"],
     ["sell", "seller"]),
    (["lend", "give", "offer"],
     ["borrow", "take", "receive"]),
    (["keep", "maintain", "protect", "defend"],
     ["get", "obtain", "acquire", "take", "convince"]),
    (["convince.*stop", "persuade.*stop", "prevent"],
     ["continue", "defend", "justify", "maintain"]),
    (["more", "larger share", "majority"],
     ["fair", "equal", "less", "minimize"]),
]

# Strong zero-sum signals in EITHER goal
STRONG_ZERO_SUM_KEYWORDS = [
    "penalty", "bonus",  # explicit reward structure
    "target price",  # price negotiation
    "budget constraint",
    "cannot afford",
    "competing for",
    "only one",
    "limited supply",
]


def diagnose_scenes():
    """Print distribution of codenames and sources for manual inspection."""
    all_profiles = EnvironmentProfile.find().all()
    print(f"Total scenes: {len(all_profiles)}\n")

    # Extract codename prefixes (strip trailing numbers/underscores)
    codename_prefixes = Counter()
    sources = Counter()
    relationships = Counter()

    for p in all_profiles:
        if hasattr(p, 'codename') and p.codename:
            # Get prefix: "craigslist_bargains_10" → "craigslist_bargains"
            prefix = re.sub(r'_?\d+$', '', p.codename)
            codename_prefixes[prefix] += 1
        if hasattr(p, 'source') and p.source:
            sources[p.source] += 1
        if hasattr(p, 'relationship') and p.relationship:
            rel = str(p.relationship) if not isinstance(p.relationship, str) else p.relationship
            relationships[rel] += 1

    print("=== Codename Prefixes (top 30) ===")
    for prefix, count in codename_prefixes.most_common(30):
        # Show an example goal for this prefix
        example = None
        for p in all_profiles:
            if hasattr(p, 'codename') and p.codename and p.codename.startswith(prefix):
                goals = p.agent_goals if hasattr(p, 'agent_goals') else []
                if goals:
                    example = [g[:60] for g in goals[:2]]
                break
        print(f"  {prefix}: {count} scenes")
        if example:
            for i, g in enumerate(example):
                print(f"    goal[{i}]: {g}...")
        print()

    print("=== Sources ===")
    for source, count in sources.most_common():
        print(f"  {source}: {count}")

    print(f"\n=== Relationships ===")
    for rel, count in relationships.most_common():
        print(f"  {rel}: {count}")


def has_goal_conflict(goals):
    """Check if two agent goals contain opposing intents."""
    if not goals or len(goals) < 2:
        return False, ""

    g0 = goals[0].lower()
    g1 = goals[1].lower()

    # Check conflict pairs
    for a_words, b_words in CONFLICT_PAIRS:
        a_in_0 = any(re.search(w, g0) for w in a_words)
        b_in_1 = any(re.search(w, g1) for w in b_words)
        a_in_1 = any(re.search(w, g1) for w in a_words)
        b_in_0 = any(re.search(w, g0) for w in b_words)

        if (a_in_0 and b_in_1) or (a_in_1 and b_in_0):
            return True, "goal_conflict"

    # Check strong zero-sum keywords in either goal
    combined = g0 + " " + g1
    for kw in STRONG_ZERO_SUM_KEYWORDS:
        if kw in combined:
            return True, f"keyword:{kw}"

    return False, ""


def filter_scenes():
    """Three-layer filtering for zero-sum scenarios."""
    all_profiles = EnvironmentProfile.find().all()
    print(f"Total scenes: {len(all_profiles)}\n")

    results = {
        "layer1_codename": [],
        "layer2_goal_conflict": [],
        "excluded_cooperative": 0,
    }

    all_matched = []

    for p in all_profiles:
        codename = p.codename if hasattr(p, 'codename') else ""
        scenario = p.scenario if hasattr(p, 'scenario') else ""
        goals = p.agent_goals if hasattr(p, 'agent_goals') else []
        source = p.source if hasattr(p, 'source') else ""

        codename_lower = codename.lower()

        # Layer 3 first: exclude known cooperative
        is_cooperative = any(cp in codename_lower for cp in COOPERATIVE_CODENAMES)
        if is_cooperative:
            results["excluded_cooperative"] += 1
            continue

        # Layer 1: codename pattern
        is_competitive_codename = any(cc in codename_lower for cc in COMPETITIVE_CODENAMES)

        # Layer 2: goal conflict analysis
        has_conflict, conflict_reason = has_goal_conflict(goals)

        if is_competitive_codename or has_conflict:
            match_reason = []
            if is_competitive_codename:
                match_reason.append(f"codename:{codename}")
            if has_conflict:
                match_reason.append(conflict_reason)

            scene = {
                "pk": p.pk,
                "codename": codename,
                "source": source,
                "scenario": scenario,
                "agent_goals": goals,
                "match_reason": match_reason,
            }
            all_matched.append(scene)

            if is_competitive_codename:
                results["layer1_codename"].append(p.pk)
            if has_conflict:
                results["layer2_goal_conflict"].append(p.pk)

    # Sort: codename matches first, then by number of reasons
    all_matched.sort(key=lambda x: len(x["match_reason"]), reverse=True)

    # Print results
    print(f"=== Filtering Results ===")
    print(f"  Excluded (cooperative): {results['excluded_cooperative']}")
    print(f"  Layer 1 (codename match): {len(results['layer1_codename'])}")
    print(f"  Layer 2 (goal conflict): {len(results['layer2_goal_conflict'])}")
    print(f"  Total matched (union): {len(all_matched)}")
    print()

    # Show top matches
    for i, scene in enumerate(all_matched[:15]):
        print(f"[{i+1}] {scene['codename']} (pk={scene['pk'][:12]}...)")
        print(f"    Reason: {', '.join(scene['match_reason'])}")
        print(f"    Scenario: {scene['scenario'][:80]}...")
        if scene['agent_goals']:
            for j, g in enumerate(scene['agent_goals'][:2]):
                print(f"    Goal[{j}]: {g[:80]}...")
        print()

    if len(all_matched) > 15:
        print(f"... and {len(all_matched) - 15} more.\n")

    # Save
    os.makedirs(config.DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.DATA_DIR, "filtered_scenes.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_matched, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_matched)} scenes to: {output_path}")
    return all_matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="Diagnostic mode: show codename/source distribution")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_scenes()
    else:
        filter_scenes()


if __name__ == "__main__":
    main()
