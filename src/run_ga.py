import argparse
import copy
import json

import pandas as pd

from fitness_evaluator import RecipeFitness
from genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig

Recipe = dict


def load_seed_recipes(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return list(data["seed_recipes"])


def load_ingredient_ids(path: str) -> list | None:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    return None


def aspects_table(population: list, evaluator: RecipeFitness) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)


def ing_set(recipe: dict) -> set:
    return set(recipe.get("ingredients_g", {}).keys())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def select_diverse_topk(scored: list, k: int = 5, lam: float = 0.7, all_ingredient_ids=None) -> list:
    seen_ids = set()
    dedup = []
    for r, s in scored:
        rid = r.get("id") or f"anon_{id(r)}"
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        dedup.append((r, float(s)))

    if not dedup:
        return []

    chosen = []
    chosen_sets = []

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evolve cookie recipes with a GA.")
    ap.add_argument("--init", default="../configs/init_recipes.json")
    ap.add_argument("--ingredients", default="../configs/ingredients.json")
    ap.add_argument("--overall", default="../configs/overall_configs.json")
    ap.add_argument("--pairings", default="../configs/flavor_pairings.json")
    ap.add_argument("--category", default="cookie")

    ap.add_argument("--population", type=int, default=50)
    ap.add_argument("--generations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5)

    ap.add_argument("--save-csv", type=str, default="run_output.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    seeds = load_seed_recipes(args.init)
    all_ings = load_ingredient_ids(args.ingredients)

    evaluator = RecipeFitness.from_files(
        ingredients_path=args.ingredients,
        overall_cfg_path=args.overall,
        flavor_pairings_path=args.pairings,
        category_id=args.category,
        weights={
            "flavor_score": 0,
            "pairing_score": 0.5,
            "soft_rules_score": 1,
            "ratio_score": 0.25,
            "novelty_score": 0.25,
            "simplicity_score": 0.3,
        }
    )

    evaluator.set_reference(seeds)

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
        bloat_remove_boost=0.35
    )

    ga = GeneticAlgorithm(
        initial_recipes=seeds,
        fitness_fn=evaluator,
        ga_cfg=cfg,
        all_ingredient_ids=all_ings,
    )

    final_recipes = ga.run(verbose=True, top_k=args.population)
    top_k_recipes = select_diverse_topk(final_recipes, k=5)
    top_k_recipes = top_k_recipes[:5]

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
