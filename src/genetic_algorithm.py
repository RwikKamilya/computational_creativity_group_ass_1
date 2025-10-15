import copy
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypedDict


class Recipe(TypedDict, total=False):
    id: str
    name: str
    category: str
    ingredients_g: Dict[str, float]


Genome = Dict[str, float]
FitnessFn = Callable[[Recipe], float]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def total_mass(ingredients_g: Genome) -> float:
    return float(sum(float(v) for v in ingredients_g.values()))


def normalize_nonnegative(ingredients_g: Genome, min_grams: float) -> Genome:
    return {k: float(v) for k, v in ingredients_g.items() if float(v) > min_grams}


@dataclass
class GeneticAlgorithmConfig:
    max_generations: int = 3000
    population_size: int = 80
    random_seed: int = 7

    tournament_k: int = 3
    survivor_fraction: float = 0.75

    elitism_n: int = 9

    crossover_prob: float = 0.90
    mutation_rate: float = 0.40
    mutation_strength: float = 0.20

    add_ingredient_prob: float = 0.10
    remove_ingredient_prob: float = 0.10

    min_grams: float = 0.0
    max_grams: float = 5000.0

    alpha_low: float = 0.35
    alpha_high: float = 0.65

    max_ingredients: int = 18
    trim_below_grams: float = 0.5
    bloat_remove_boost: float = 0.25


class GeneticAlgorithm:

    def __init__(self, initial_recipes: List[Recipe], fitness_fn: Optional[FitnessFn],
                 ga_cfg: Optional[GeneticAlgorithmConfig] = None, all_ingredient_ids: Optional[List[str]] = None) -> None:
        if not initial_recipes:
            raise ValueError("initial_recipes cannot be empty.")

        self.fitness_fn = fitness_fn
        self.cfg = ga_cfg or GeneticAlgorithmConfig()
        self.rng = random.Random(self.cfg.random_seed)

        if all_ingredient_ids is None:
            all_set = set()
            for r in initial_recipes:
                all_set.update(r["ingredients_g"].keys())
            self.all_ingredient_ids = sorted(all_set)
        else:
            self.all_ingredient_ids = sorted(set(all_ingredient_ids))

        self.population = self._bootstrap_population(initial_recipes, self.cfg.population_size)


    def run(self, verbose: bool = True, top_k: int = 5) -> List[Tuple[Recipe, float]]:
        best: Optional[Recipe] = None
        best_fit: float = float("-inf")

        for gen in range(1, self.cfg.max_generations + 1):

            scored = [(r, self.fitness_fn(r)) for r in self.population]
            scored.sort(key=lambda x: x[1], reverse=True)

            if scored[0][1] > best_fit:
                best = copy.deepcopy(scored[0][0])
                best_fit = scored[0][1]

            if verbose:
                mean_score = sum(s for _, s in scored) / len(scored)
                print(f"[Gen {gen}] best={scored[0][1]:.4f}  mean={mean_score:.4f}")

            elites = [self._shallow_copy_with_fresh_ingredients(r) for r, _ in scored[: self.cfg.elitism_n]]

            n_survivors = max(self.cfg.elitism_n, int(self.cfg.survivor_fraction * len(self.population)))
            parents_scored = scored[:n_survivors]  # <— use the fitter slice only
            parents_pool = [r for r, _ in parents_scored]

            next_pop: List[Recipe] = elites
            while len(next_pop) < self.cfg.population_size:
                p1 = self._tournament_select(parents_scored, self.cfg.tournament_k)
                p2 = self._tournament_select(parents_scored, self.cfg.tournament_k)

                if self.rng.random() < self.cfg.crossover_prob:
                    child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(self.rng.choice(parents_pool))
                    child["id"] = f"child_{self.rng.randint(0, 10 ** 9)}"
                    child["name"] = f"cookie_{self.rng.randint(0, 10 ** 6)}"

                child = self._mutate(child)
                next_pop.append(child)

            self.population = next_pop

        final_scored = [(r, self.fitness_fn(r)) for r in self.population]
        final_scored.sort(key=lambda x: x[1], reverse=True)
        k = min(top_k, len(final_scored))

        top_k = [(copy.deepcopy(r), s) for (r, s) in final_scored[:k]]
        return top_k

    @staticmethod
    def _shallow_copy_with_fresh_ingredients(r: Recipe) -> Recipe:
        out = dict(r)
        out["ingredients_g"] = dict(r["ingredients_g"])
        return out

    def _tournament_select(self, scored_pop: List[Tuple[Recipe, float]], k: int) -> Recipe:
        contestants = self.rng.sample(scored_pop, k=min(k, len(scored_pop)))
        contestants.sort(key=lambda x: x[1], reverse=True)
        return contestants[0][0]

    def _bootstrap_population(self, seeds: List[Recipe], pop_size: int) -> List[Recipe]:
        out = []
        if len(seeds) >= pop_size:
            seeds_copy = copy.deepcopy(seeds)
            self.rng.shuffle(seeds_copy)
            return seeds_copy[:pop_size]

        out.extend(copy.deepcopy(seeds))
        while len(out) < pop_size:
            base = copy.deepcopy(self.rng.choice(seeds))
            out.append(self._mutate(base, weak=True))
        return out

    def _pick_random_ingredient(self, exclude: set) -> Optional[str]:
        if not self.all_ingredient_ids:
            return None
        candidates = [k for k in self.all_ingredient_ids if k not in exclude]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _renorm_total(self, g, target=800.0):
        tot = sum(g.values()) or 1.0
        scale = clamp(target / tot, 0.5, 2.0)
        for k in list(g.keys()):
            g[k] = clamp(g[k] * scale, self.cfg.min_grams, self.cfg.max_grams)
        return g

    def _crossover(self, p1: Recipe, p2: Recipe) -> Recipe:
        g1, g2 = p1["ingredients_g"], p2["ingredients_g"]
        alpha = self.rng.uniform(self.cfg.alpha_low, self.cfg.alpha_high)

        child_g: Genome = {}
        keys1, keys2 = set(g1.keys()), set(g2.keys())

        for k in sorted(keys1 & keys2):
            v = alpha * float(g1[k]) + (1.0 - alpha) * float(g2[k])
            v = clamp(v, self.cfg.min_grams, self.cfg.max_grams)
            if v > 0:
                child_g[k] = v

        for k in sorted(keys1 - keys2):
            if self.rng.random() < 0.5:
                v = clamp(float(g1[k]), self.cfg.min_grams, self.cfg.max_grams)
                if v > 0:
                    child_g[k] = v
        for k in sorted(keys2 - keys1):
            if self.rng.random() < 0.5:
                v = clamp(float(g2[k]), self.cfg.min_grams, self.cfg.max_grams)
                if v > 0:
                    child_g[k] = v

        child: Recipe = {
            "id": f"child_{self.rng.randint(0, 10 ** 9)}",
            "name": f"cookie_{self.rng.randint(0, 10 ** 6)}",
            "category": p1.get("category", "cookie"),
            "ingredients_g": normalize_nonnegative(child_g, self.cfg.trim_below_grams),
        }
        return child

    def _mutate(self, r: Recipe, weak: bool = False) -> Recipe:
        child = dict(r)
        g = dict(r["ingredients_g"])  # work on a copy

        total = max(1.0, total_mass(g))
        strength = self.cfg.mutation_strength * (0.5 if weak else 1.0)

        for k in list(g.keys()):
            if self.rng.random() < self.cfg.mutation_rate:
                qk = float(g[k])
                sigma_k = clamp(self.cfg.mutation_strength * qk, 0.05, 50.0)
                noise = self.rng.gauss(0.0, sigma_k)
                new_v = clamp(qk + noise, self.cfg.min_grams, self.cfg.max_grams)
                if new_v <= self.cfg.min_grams:
                    del g[k]
                else:
                    g[k] = new_v

        if self.rng.random() < self.cfg.add_ingredient_prob and self.all_ingredient_ids:
            cand = self._pick_random_ingredient(exclude=set(g.keys()))
            if cand:
                g[cand] = clamp(abs(self.rng.gauss(10.0, 10.0)), 1.0, 200.0)

        remove_p = self.cfg.remove_ingredient_prob
        if len(g) > self.cfg.max_ingredients:
            remove_p = min(1.0, remove_p + self.cfg.bloat_remove_boost)
        if self.rng.random() < remove_p and len(g) > 1:
            del g[self.rng.choice(sorted(g.keys()))]

        g = self._renorm_total(g, target=self.rng.uniform(700.0, 1000.0))

        T = sum(g.values()) or 1.0

        # Kludge: GA should not have recipe context
        ABS_CAP = {
            "salt_fine": 8.0,
            "vanilla_extract": 12.0,
            "almond_extract": 6.0,
            "yeast_dry": 0.0,
        }
        FRAC_CAP = {
            "salt_fine": 0.015,
            "vanilla_extract": 0.020,
        }

        for k in list(g.keys()):
            if k in ABS_CAP:
                if ABS_CAP[k] == 0.0:
                    del g[k]
                    continue
                g[k] = min(g[k], ABS_CAP[k])
            if k in FRAC_CAP:
                g[k] = min(g[k], FRAC_CAP[k] * T)

        if self.cfg.trim_below_grams > 0:
            g = {k: v for k, v in g.items() if v >= self.cfg.trim_below_grams}
        if len(g) > self.cfg.max_ingredients:
            keys = list(g.keys())
            self.rng.shuffle(keys)
            for k in keys[self.cfg.max_ingredients:]:
                del g[k]

        child["ingredients_g"] = g
        return child
