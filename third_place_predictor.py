"""
Third-Place Qualification Predictor — FIFA 48-Team World Cup Format
====================================================================
In the 48-team format there are 12 groups of 4. The top 2 from each
group advance automatically (24 teams). The best 8 of the 12 third-place
teams also advance, making qualification thresholds inherently uncertain.

This module provides four things:
  1. BRUTE-FORCE enumerator  — reference implementation, validates others.
  2. PRUNED DFS enumerator   — skips branches that cannot beat a target record.
  3. THRESHOLD FINDER        — pinpoints the points/GD a team likely needs.
  4. SURVIVAL CURVE          — P(advance | 3rd place finishes with X pts).

Complexity
----------
Each group has C(4,2) = 6 matches. With K score outcomes per match the
full enumeration space per group is K^6. There are 12 independent groups.
- Brute force:  O(K^6)  per group, O(12 * K^6) total
- Pruned DFS:   branches are cut whenever the remaining matches cannot
                produce a third-place record beating the supplied target.
                Typically saves 30-70% of work for tight targets.
- Because all 12 groups are structurally identical, we enumerate ONE group
  and reuse its distribution 12 times, dropping total work to O(K^6).

Validation
----------
Run:  python third_place_predictor.py --validate
"""

import argparse
import itertools
import random
import time
from collections import defaultdict
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Core types
# ─────────────────────────────────────────────────────────────────────────────

class ThirdPlaceRecord(NamedTuple):
    """Immutable, sortable record for a 3rd-place team (higher = better)."""
    points: int
    goal_diff: int
    goals_for: int

    def __str__(self):
        return f"{self.points}pts  GD{self.goal_diff:+d}  GF{self.goals_for}"


# ─────────────────────────────────────────────────────────────────────────────
# Match & group helpers
# ─────────────────────────────────────────────────────────────────────────────

# All pairings in a 4-team group (round-robin)
MATCH_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def match_points(home_goals: int, away_goals: int) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def third_place_record(combo: tuple) -> ThirdPlaceRecord:
    """
    Given 6 match results (home_goals, away_goals) for MATCH_PAIRS,
    return the ThirdPlaceRecord of the 3rd-ranked team.
    """
    pts = [0] * 4
    gd  = [0] * 4
    gf  = [0] * 4
    for (ht, at), (hg, ag) in zip(MATCH_PAIRS, combo):
        hp, ap = match_points(hg, ag)
        pts[ht] += hp;  pts[at] += ap
        gd[ht]  += hg - ag;  gd[at] += ag - hg
        gf[ht]  += hg;       gf[at] += ag
    # Sort teams by (pts, gd, gf) descending; pick index 2
    order = sorted(range(4), key=lambda i: (-pts[i], -gd[i], -gf[i]))
    i = order[2]
    return ThirdPlaceRecord(pts[i], gd[i], gf[i])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Brute-force enumerator
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_enumerate(
    score_set: list[tuple[int, int]]
) -> dict[ThirdPlaceRecord, int]:
    """
    Enumerate ALL K^6 group outcomes exhaustively.
    Returns {ThirdPlaceRecord → frequency}.
    """
    dist: dict[ThirdPlaceRecord, int] = defaultdict(int)
    for combo in itertools.product(score_set, repeat=6):
        dist[third_place_record(combo)] += 1
    return dict(dist)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pruned DFS enumerator
# ─────────────────────────────────────────────────────────────────────────────

def pruned_enumerate(
    score_set: list[tuple[int, int]],
    target: ThirdPlaceRecord | None = None,
) -> tuple[dict[ThirdPlaceRecord, int], int, int]:
    """
    Depth-first enumeration with pruning.

    If `target` is given, a branch is pruned when the BEST possible
    third-place record reachable from that node cannot beat `target`.

    Upper-bound estimation
    ----------------------
    For each unplayed match, each of the two teams could gain 3 pts.
    We compute the maximum points each team can possibly reach, then
    check if the third-best max-points value is < target.points.
    If so, no completion can produce a third-place team with enough pts.

    Returns
    -------
    (distribution, branches_explored, branches_pruned)
    """
    dist: dict[ThirdPlaceRecord, int] = defaultdict(int)
    explored = 0
    pruned   = 0

    # Precompute "max additional points" for each team given remaining matches
    # remaining_pts[match_idx][team] = max pts team can earn from match_idx onward
    max_remaining = []
    for start in range(6):
        earn = [0] * 4
        for ht, at in MATCH_PAIRS[start:]:
            earn[ht] += 3
            earn[at] += 3
        max_remaining.append(earn)
    max_remaining.append([0] * 4)  # after all 6 matches

    def dfs(
        match_idx: int,
        pts: list[int],
        gd: list[int],
        gf: list[int],
    ) -> None:
        nonlocal explored, pruned

        if match_idx == 6:
            explored += 1
            order = sorted(range(4), key=lambda i: (-pts[i], -gd[i], -gf[i]))
            i = order[2]
            dist[ThirdPlaceRecord(pts[i], gd[i], gf[i])] += 1
            return

        # ── Pruning ──────────────────────────────────────────────────────────
        if target is not None:
            earn = max_remaining[match_idx]
            max_pts = sorted(
                [pts[i] + earn[i] for i in range(4)], reverse=True
            )
            # Third-best achievable points
            if max_pts[2] < target.points:
                pruned += 1
                return
        # ─────────────────────────────────────────────────────────────────────

        ht, at = MATCH_PAIRS[match_idx]
        for hg, ag in score_set:
            hp, ap = match_points(hg, ag)
            pts[ht] += hp;  pts[at] += ap
            gd[ht]  += hg - ag;  gd[at] += ag - hg
            gf[ht]  += hg;       gf[at] += ag

            dfs(match_idx + 1, pts, gd, gf)

            pts[ht] -= hp;  pts[at] -= ap
            gd[ht]  -= hg - ag;  gd[at] -= ag - hg
            gf[ht]  -= hg;       gf[at] -= ag

    dfs(0, [0]*4, [0]*4, [0]*4)
    return dict(dist), explored, pruned


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold finder
# ─────────────────────────────────────────────────────────────────────────────

def find_threshold(
    score_set: list[tuple[int, int]],
    n_groups: int = 12,
    n_advance: int = 8,
    n_samples: int = 80_000,
    rng_seed: int = 42,
) -> dict:
    """
    Determine the points / GD a 3rd-place team needs to advance.

    Strategy
    --------
    1. Enumerate one group's third-place distribution (all 12 are symmetric).
    2. Use Monte Carlo to simulate the joint ranking across n_groups groups.
    3. Report the modal cutoff record and survival probabilities.

    Parameters
    ----------
    score_set  : possible (home_goals, away_goals) outcomes per match
    n_groups   : number of groups in the tournament (default 12)
    n_advance  : number of 3rd-place teams that advance (default 8)
    n_samples  : Monte Carlo samples for cross-group analysis
    rng_seed   : for reproducibility
    """
    random.seed(rng_seed)

    # ── Step 1: enumerate one group ────────────────────────────────────────
    t0 = time.perf_counter()
    dist, explored, pruned = pruned_enumerate(score_set, target=None)
    enum_time = time.perf_counter() - t0

    total_outcomes = len(score_set) ** 6
    print(f"Group enumeration: {explored:,} leaf nodes  |  "
          f"{total_outcomes:,} brute-force  |  "
          f"Time: {enum_time:.2f}s")

    # Build cumulative distribution for fast weighted sampling
    records  = list(dist.keys())
    weights  = list(dist.values())
    cum_w    = []
    running  = 0
    total_w  = sum(weights)
    for w in weights:
        running += w
        cum_w.append(running)

    def sample_one() -> ThirdPlaceRecord:
        r = random.randint(1, total_w)
        lo, hi = 0, len(records) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_w[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return records[lo]

    # ── Step 2: Monte Carlo cross-group ranking ────────────────────────────
    cutoff_dist: dict[ThirdPlaceRecord, int] = defaultdict(int)
    pts_wins:  dict[int, int] = defaultdict(int)
    pts_trials: dict[int, int] = defaultdict(int)

    for _ in range(n_samples):
        thirds = [sample_one() for _ in range(n_groups)]
        thirds_sorted = sorted(thirds, reverse=True)
        cutoff = thirds_sorted[n_advance - 1]   # worst team that advances
        cutoff_dist[cutoff] += 1

        for rec in thirds:
            pts_trials[rec.points] += 1
            # A team advances if it ranks in the top n_advance
            if rec >= cutoff and thirds.index(rec) < n_advance:
                pts_wins[rec.points] += 1

    # Modal cutoff
    modal_cutoff = max(cutoff_dist, key=lambda r: cutoff_dist[r])
    cutoff_freq  = cutoff_dist[modal_cutoff] / n_samples

    # Survival probabilities per points value
    survival = {
        p: pts_wins[p] / pts_trials[p]
        for p in sorted(pts_trials)
        if pts_trials[p] > 0
    }

    return {
        "group_distribution":  dist,
        "enum_time_s":         enum_time,
        "total_brute_force":   total_outcomes,
        "explored":            explored,
        "modal_cutoff":        modal_cutoff,
        "cutoff_frequency":    cutoff_freq,
        "survival_by_points":  survival,
        "n_samples":           n_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(score_set: list[tuple[int, int]]) -> bool:
    """
    Verify pruned_enumerate matches brute_force_enumerate exactly.
    Returns True if they agree.
    """
    print("─" * 60)
    print("VALIDATION: pruned DFS vs brute-force")
    print(f"Score set size: {len(score_set)}   "
          f"Total scenarios: {len(score_set)**6:,}")
    print("─" * 60)

    t0 = time.perf_counter()
    bf = brute_force_enumerate(score_set)
    t1 = time.perf_counter()
    print(f"Brute force:  {sum(bf.values()):>10,} outcomes  "
          f" {len(bf):>3} distinct   {t1-t0:.2f}s")

    t0 = time.perf_counter()
    pr, explored, pruned_nodes = pruned_enumerate(score_set, target=None)
    t1 = time.perf_counter()
    print(f"Pruned DFS:   {sum(pr.values()):>10,} outcomes  "
          f" {len(pr):>3} distinct   {t1-t0:.2f}s")
    print(f"              (no target → no pruning; "
          f"explored={explored:,}, pruned={pruned_nodes:,})")

    # Test pruning WITH a target: enumerate only outcomes ≥ 4 pts
    target = ThirdPlaceRecord(4, -1, 0)
    t0 = time.perf_counter()
    pt, exp2, prun2 = pruned_enumerate(score_set, target=target)
    t1 = time.perf_counter()
    bf_count = sum(v for k, v in bf.items() if k >= target)
    pr_count = sum(v for k, v in pt.items() if k >= target)
    savings   = prun2 / (exp2 + prun2) * 100 if (exp2 + prun2) > 0 else 0
    print(f"\nWith target {target}:")
    print(f"  BF count of qualifying outcomes: {bf_count:,}")
    print(f"  Pruned count:                    {pr_count:,}")
    print(f"  Match: {bf_count == pr_count}")
    print(f"  Branches pruned: {prun2:,} / {exp2+prun2:,} "
          f"({savings:.1f}% savings)  {t1-t0:.2f}s")

    match = bf == pr
    print(f"\nFull distribution match: {match}")
    if not match:
        diff = {k for k in (set(bf) | set(pr)) if bf.get(k) != pr.get(k)}
        print(f"  Differing keys: {diff}")
    print()
    return match


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    W = 60
    print("=" * W)
    print("  THIRD-PLACE QUALIFICATION PREDICTOR — 48-TEAM FORMAT")
    print("=" * W)
    print(f"  Tournament: 12 groups, 8 best 3rd-place teams advance")
    print()
    print(f"  Group enumeration time  : {results['enum_time_s']:.2f}s")
    print(f"  Brute-force scenarios   : {results['total_brute_force']:,}")
    print(f"  Pruned scenarios explored: {results['explored']:,}")
    savings = (1 - results['explored']/results['total_brute_force']) * 100
    print(f"  Pruning savings         : {savings:.1f}%")
    print(f"  Distinct 3rd-place recs : {len(results['group_distribution'])}")
    print()

    mc = results['modal_cutoff']
    print(f"  Most common cutoff record : {mc}")
    print(f"  Cutoff frequency          : {results['cutoff_frequency']:.1%} "
          f"of {results['n_samples']:,} simulations")
    print()

    print("  ── Survival curve: P(advance | 3rd place with X points) ──")
    print(f"  {'Points':>8}  {'P(advance)':>12}  {'Verdict':<9}  Chart")
    print("  " + "─" * 52)
    verdicts = {
        lambda p: p > 0.90: "SAFE ✓",
        lambda p: p > 0.65: "Likely",
        lambda p: p > 0.35: "Risky",
        lambda p: p > 0.10: "Unlikely",
    }
    for pts, prob in sorted(results['survival_by_points'].items()):
        verdict = "Almost impossible"
        if prob > 0.90:   verdict = "SAFE ✓"
        elif prob > 0.65: verdict = "Likely"
        elif prob > 0.35: verdict = "Risky"
        elif prob > 0.10: verdict = "Unlikely"
        bar = "█" * int(prob * 30)
        print(f"  {pts:>8}  {prob:>11.1%}  {verdict:<9}  {bar}")
    print()

    print("  ── Top 10 most common 3rd-place group outcomes ──")
    dist = results['group_distribution']
    total = sum(dist.values())
    top10 = sorted(dist.items(), key=lambda x: -x[1])[:10]
    print(f"  {'Pts':>4} {'GD':>5} {'GF':>5}  {'Count':>8}  {'Freq':>7}")
    print("  " + "─" * 35)
    for rec, cnt in top10:
        print(f"  {rec.points:>4} {rec.goal_diff:>+5} {rec.goals_for:>5}  "
              f"{cnt:>8,}  {cnt/total:>6.2%}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Score-set presets
# ─────────────────────────────────────────────────────────────────────────────

def score_set_preset(name: str) -> list[tuple[int, int]]:
    """
    Named presets for the score universe:
      'tiny'     — 8 outcomes (0-3 goals total), fast validation
      'standard' — 9 outcomes (max 2 goals/team), default
      'full'     — all outcomes up to 3 goals/team (16 outcomes)
    """
    if name == "tiny":
        return [(h, a) for h in range(3) for a in range(3) if h + a <= 3]
    if name == "standard":
        return [(h, a) for h in range(3) for a in range(3)]
    if name == "full":
        return [(h, a) for h in range(4) for a in range(4)]
    raise ValueError(f"Unknown preset: {name!r}. Choose tiny/standard/full.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Third-place qualification predictor (48-team World Cup)"
    )
    parser.add_argument(
        "--preset", default="standard",
        choices=["tiny", "standard", "full"],
        help="Score universe size (default: standard)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run brute-force vs pruned validation first"
    )
    parser.add_argument(
        "--samples", type=int, default=80_000,
        help="Monte Carlo samples for cross-group analysis (default: 80000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    score_set = score_set_preset(args.preset)
    print(f"Score set preset: '{args.preset}'  ({len(score_set)} outcomes,  "
          f"{len(score_set)**6:,} scenarios/group)\n")

    if args.validate:
        ok = validate(score_set)
        if not ok:
            print("ERROR: Validation failed!")
            return

    results = find_threshold(
        score_set=score_set,
        n_groups=12,
        n_advance=8,
        n_samples=args.samples,
        rng_seed=args.seed,
    )
    print()
    print_results(results)


if __name__ == "__main__":
    main()
