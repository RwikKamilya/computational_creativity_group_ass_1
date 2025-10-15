#!/usr/bin/env python3
"""
run_ga.py

Run the cookie Genetic Algorithm with the RecipeFitness evaluator.

Usage (defaults work):
    python run_ga.py \
        --init configs/init_recipes.json \
        --ingredients configs/ingredients.json \
        --overall configs/overall_configs.json \
        --pairings configs/flavor_pairings.json \
        --category cookie \
        --population 80 \
        --generations 3000 \
        --seed 7 \
        --save-csv run_output.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fitness_evaluator import RecipeFitness
from genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig


# ---------- Types ----------

Recipe = Dict[str, Any]


# ---------- Helpers ----------

def load_seed_recipes(path: str) -> List[Recipe]:
    with open(path, "r") as f:
        data = json.load(f)
    return list(data["seed_recipes"])

def load_ingredient_ids(path: str) -> Optional[List[str]]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    return None

def aspects_table(population: List[Recipe], evaluator: RecipeFitness) -> pd.DataFrame:
    rows = []
    for r in population:
        aspects = evaluator.compute_aspects(r)
        score = evaluator.fitness_from_aspects(aspects)
        rows.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "valid": aspects["valid"],
            "problems": "; ".join(aspects["problems"]),
            "pairing_score": round(aspects["pairing_score"], 3),
            "soft_rules_score": round(aspects["soft_rules_score"], 3),
            "flavor_score": round(aspects["flavor_score"], 3),
            "ratio_score": round(aspects["ratio_score"], 3),
            "novelty_score": round(aspects["novelty_score"], 3),
            "simplicity_score": round(aspects["simplicity_score"], 3),
            "fitness": round(score, 3),
        })
    return (pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True))



def select_diverse_topk(
    scored: List[Tuple[dict, float]],
    k: int = 5,
    lam: float = 0.7,
    all_ingredient_ids: List[str] | None = None,  # unused now, kept for signature compat
) -> List[Tuple[dict, float]]:
    """
    scored: list of (recipe_dict, fitness) sorted desc by fitness
    Returns k items (recipe, fitness) chosen by MMR with Jaccard(set-of-ingredients) diversity.
    MMR(r) = lam * fitness(r) - (1 - lam) * max_{s in chosen} Jaccard(set(r), set(s))
    """

    # --- 0) de-duplicate by id while preserving best score first ---
    seen_ids = set()
    dedup: List[Tuple[dict, float]] = []
    for r, s in scored:
        rid = r.get("id") or f"anon_{id(r)}"
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        dedup.append((r, float(s)))

    if not dedup:
        return []

    def ing_set(recipe: dict) -> set:
        return set(recipe.get("ingredients_g", {}).keys())

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b) or 1
        return inter / union

    # --- 1) greedy MMR selection ---
    chosen: List[Tuple[dict, float]] = []
    chosen_sets: List[set] = []

    # take the best by fitness first
    r0, s0 = dedup[0]
    chosen.append((copy.deepcopy(r0), s0))
    chosen_sets.append(ing_set(r0))

    candidates = dedup[1:]
    while len(chosen) < min(k, len(dedup)) and candidates:
        best_idx = None
        best_mmr = -1e18

        for idx, (r, s) in enumerate(candidates):
            rs = ing_set(r)
            max_sim = max((jaccard(rs, cs) for cs in chosen_sets), default=0.0)
            mmr = lam * s - (1.0 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        r_star, s_star = candidates.pop(best_idx)
        chosen.append((copy.deepcopy(r_star), s_star))
        chosen_sets.append(ing_set(r_star))

    return chosen


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evolve cookie recipes with a GA.")
    ap.add_argument("--init", default="configs/init_recipes.json")
    ap.add_argument("--ingredients", default="configs/ingredients.json")
    ap.add_argument("--overall", default="configs/overall_configs.json")
    ap.add_argument("--pairings", default="configs/flavor_pairings.json")
    ap.add_argument("--category", default="cookie")

    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--generations", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--save-csv", type=str, default="run_output.csv")
    return ap.parse_args()



# ---------- Main ----------

def main() -> None:
    args = parse_args()

    # Load data
    seeds = load_seed_recipes(args.init)
    all_ings = load_ingredient_ids(args.ingredients)

    # Build fitness (weights are tunable)
    evaluator = RecipeFitness.from_files(
        ingredients_path=args.ingredients,
        overall_cfg_path=args.overall,
        flavor_pairings_path=args.pairings,
        category_id=args.category,
        weights={
            "flavor_score": 0,  # Set to 0 as it is 0.4 pairing + 0.6 soft rules anyway
            "pairing_score": 0.5,
            "soft_rules_score": 1,  # up
            "ratio_score": 0.25,  # up
            "novelty_score": 0,
            "simplicity_score": 0.3,  # small nudge toward simpler recipes
        }
    )

    evaluator.set_reference(seeds)  # seed novelty

    # GA config
    cfg = GeneticAlgorithmConfig(
        max_generations=args.generations,
        population_size=args.population,
        random_seed=args.seed,

        tournament_k=3,
        survivor_fraction=0.5,
        elitism_n=3,

        crossover_prob=0.50,
        mutation_rate=0.2,
        mutation_strength=0.5,

        add_ingredient_prob=0.5,
        remove_ingredient_prob=0.3,

        alpha_low=0.20,
        alpha_high=0.80,

        max_ingredients=20,
        trim_below_grams=0.5,
        bloat_remove_boost=0.35,
    )

    # Construct GA
    ga = GeneticAlgorithm(
        initial_recipes=seeds,
        fitness_fn=evaluator,                # RecipeFitness is __call__-able
        ga_cfg=cfg,
        all_ingredient_ids=all_ings,
    )

    final_recipes = ga.run(verbose=True, top_k=args.population)
    top_k_recipes = select_diverse_topk(final_recipes, k=5)
    top_k_recipes = top_k_recipes[:5]
    # Run

    print(f"\n=== Top {len(top_k_recipes)} RECIPE ===")

    for rank, recipe_score_tuple in enumerate(top_k_recipes, start=1):
        recipe = recipe_score_tuple[0]
        fitness_score = recipe_score_tuple[1]
        print("Rank     :", rank)
        print("Name     :", recipe.get("name"))
        print("ID       :", recipe.get("id"))
        print("Category :", recipe.get("category"))
        print("Fitness  :", round(fitness_score, 4))
        print("Ingredients (g):")
        for k, v in sorted(recipe["ingredients_g"].items()):
            print(f"  - {k:20s} : {float(v):.1f} g")


if __name__ == "__main__":
    main()
