#!/usr/bin/env python3
"""
Run the cookie GA with default configs and the RecipeFitness evaluator.

Usage (defaults just work):
    python run_ga.py
Optional flags:
    --init configs/init_recipes.json
    --ingredients configs/ingredients.json
    --overall configs/overall_configs.json
    --pairings configs/flavor_pairings.json
    --category cookie
    --population 60
    --generations 40
    --seed 7
    --save-csv run_population_scores.csv
"""

import argparse
import json
import copy
import pandas as pd
from typing import List, Dict, Any

# --- import your classes ---
# Adjust these imports to match your file layout.
from fitness_evaluator import RecipeFitness
from ga import GeneticAlgorithm, GAConfig

Recipe = Dict[str, Any]


# ---------- small helpers ----------
def load_seed_recipes(path: str) -> List[Recipe]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["seed_recipes"]


def load_ingredient_ids(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    return None


def aspects_table(pop: List[Recipe], fit: RecipeFitness) -> pd.DataFrame:
    rows = []
    for r in pop:
        asp = fit.compute_aspects(r)
        score = fit.fitness_from_aspects(asp)
        rows.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "valid": asp["valid"],
            "problems": "; ".join(asp["problems"]),
            "pairing_score": round(asp["pairing_score"], 3),
            "soft_rules_score": round(asp["soft_rules_score"], 3),
            "flavor_score": round(asp["flavor_score"], 3),
            "ratio_score": round(asp["ratio_score"], 3),
            "novelty_score": round(asp["novelty_score"], 3),
            "simplicity_score": round(asp["simplicity_score"], 3),
            "fitness": round(score, 3),

        })

    return pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="configs/init_recipes.json")
    ap.add_argument("--ingredients", default="configs/ingredients.json")
    ap.add_argument("--overall", default="configs/overall_configs.json")
    ap.add_argument("--pairings", default="configs/flavor_pairings.json")
    ap.add_argument("--category", default="cookie")

    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--generations", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--save-csv", type=str, default="run_output.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    # --- load data ---
    seeds = load_seed_recipes(args.init)
    all_ings = load_ingredient_ids(args.ingredients)

    # --- build fitness (defaults can be tweaked here) ---
    fit = RecipeFitness.from_files(
        ingredients_path=args.ingredients,
        overall_cfg_path=args.overall,
        flavor_pairings_path=args.pairings,
        category_id=args.category,
        weights={"flavor_score": 1.30,  # blended flavor, OR set to 0 if you split
                 "pairing_score": 0.2,  # keep 0 if using blended flavor
                 "soft_rules_score": 0.2,  # set >0 if you want to weight separately
                 "ratio_score": 0.20,
                 "novelty_score": 0.02,
                 "simplicity_score": 0.00
                 },
    )

    # Optional: seed novelty with the initial set (the GA will still work without this)
    fit.set_reference(seeds)

    # --- GA config defaults (reasonable starting point) ---
    cfg = GAConfig(
        max_generations=args.generations,
        population_size=args.population,
        random_seed=args.seed,

        tournament_k=3,
        survivor_fraction=0.75,
        elitism_n=9,

        crossover_prob=0.9,
        mutation_rate=0.45,
        mutation_strength=0.25,

        add_ingredient_prob=0.30,
        remove_ingredient_prob=0.10,

        alpha_low=0.35,
        alpha_high=0.65,

        max_ingredients=18,
        trim_below_grams=0.5,
        bloat_remove_boost=0.35,
    )

    # --- construct GA ---
    ga = GeneticAlgorithm(
        initial_recipes=seeds,
        fitness_fn=fit,  # RecipeFitness is __call__-able
        ga_cfg=cfg,
        all_ingredient_ids=all_ings
    )

    # (Optional) If you added the 2-line hook in GA.run to call set_reference(self.population),
    # novelty will be based on the *current* population each generation.

    # --- run ---
    best_recipe, best_score = ga.run(verbose=True)

    # --- report ---
    print("\n=== BEST RECIPE ===")
    print("Name     :", best_recipe.get("name"))
    print("ID       :", best_recipe.get("id"))
    print("Category :", best_recipe.get("category"))
    print("Fitness  :", round(best_score, 4))
    print("Ingredients (g):")
    for k, v in sorted(best_recipe["ingredients_g"].items()):
        print(f"  - {k:20s} : {float(v):.1f} g")

    # Build a table of the final population for inspection/saving
    # (We don’t have GA.population here after run, so we’ll synthesize a last eval table:
    #  just evaluate the best and the seeds for a quick glance; or, if your GA exposes
    #  the final population (e.g., ga.population), use that directly.)
    try:
        final_pop = copy.deepcopy(ga.population)  # if your GA exposes it
        fit.set_reference(final_pop)
        df = aspects_table(final_pop, fit)
    except Exception:
        # fallback: seeds + winner
        pool = seeds + [best_recipe]
        fit.set_reference(pool)
        df = aspects_table(pool, fit)

    print("\n=== TOP 5 (final eval) ===")
    print(df.head(5).to_string(index=False))

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved: {args.save_csv}")


if __name__ == "__main__":
    main()
