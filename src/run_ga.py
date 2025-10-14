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
from typing import Any, Dict, List, Optional

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


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evolve cookie recipes with a GA.")
    ap.add_argument("--init", default="../configs/init_recipes.json")
    ap.add_argument("--ingredients", default="../configs/ingredients.json")
    ap.add_argument("--overall", default="../configs/overall_configs.json")
    ap.add_argument("--pairings", default="../configs/flavor_pairings.json")
    ap.add_argument("--category", default="cookie")

    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--generations", type=int, default=300)
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
            "flavor_score":     1.30,   # blended flavor: 0.6*soft + 0.4*pairing
            "pairing_score":    0.20,   # keep 0 unless you want an extra term
            "soft_rules_score": 0.20,   # keep 0 unless you want separate accounting
            "ratio_score":      0.20,
            "novelty_score":    0.20,
            "simplicity_score": 0.00,
        },
    )

    evaluator.set_reference(seeds)  # seed novelty

    # GA config
    cfg = GeneticAlgorithmConfig(
        max_generations=args.generations,
        population_size=args.population,
        random_seed=args.seed,

        tournament_k=3,
        survivor_fraction=0.60,
        elitism_n=9,

        crossover_prob=0.90,
        mutation_rate=0.30,
        mutation_strength=0.15,

        add_ingredient_prob=0.15,
        remove_ingredient_prob=0.12,

        alpha_low=0.35,
        alpha_high=0.65,

        max_ingredients=18,
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

    # Run
    best_recipe, best_score = ga.run(verbose=True)

    # Report
    print("\n=== BEST RECIPE ===")
    print("Name     :", best_recipe.get("name"))
    print("ID       :", best_recipe.get("id"))
    print("Category :", best_recipe.get("category"))
    print("Fitness  :", round(best_score, 4))
    print("Ingredients (g):")
    for k, v in sorted(best_recipe["ingredients_g"].items()):
        print(f"  - {k:20s} : {float(v):.1f} g")

    # Final evaluation table
    try:
        final_pop = copy.deepcopy(ga.population)  # exposed by GA
        evaluator.set_reference(final_pop)
        df = aspects_table(final_pop, evaluator)
    except Exception:
        pool = seeds + [best_recipe]
        evaluator.set_reference(pool)
        df = aspects_table(pool, evaluator)

    print("\n=== TOP 5 (final eval) ===")
    print(df.head(5).to_string(index=False))

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved: {args.save_csv}")


if __name__ == "__main__":
    main()
