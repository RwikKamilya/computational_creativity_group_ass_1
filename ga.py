import json
import random
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional


# =========================
# Types
# =========================

Recipe = Dict[str, any]         # expects keys: id, name, category, ingredients_g
Genome = Dict[str, float]       # ingredient_id -> grams
FitnessFn = Callable[[Recipe], float]


# =========================
# Config
# =========================

@dataclass
class GAConfig:
    # Population / iterations
    max_generations: int = 50
    population_size: int = 50          # if initial set smaller, we'll sample with replacement
    random_seed: int = 42

    # Selection
    tournament_k: int = 3              # selection pressure: higher = more exploitation
    survivor_fraction: float = 0.5     # fraction of population to be parents for next gen

    # Elitism
    elitism_n: int = 2                 # number of top individuals carried unchanged

    # Crossover & mutation
    crossover_prob: float = 0.9        # probability two selected parents will crossover
    mutation_rate: float = 0.3         # per-ingredient mutation probability
    mutation_strength: float = 0.05    # relative noise magnitude (fraction of total mass)

    # Ingredient exploration (optional add/remove ops)
    add_ingredient_prob: float = 0.05  # chance to add a new ingredient not in genome
    remove_ingredient_prob: float = 0.02

    # Numeric safety
    min_grams: float = 0.0
    max_grams: float = 5000.0          # hard ceiling to prevent runaway values

    # Crossover mixing
    alpha_low: float = 0.25            # alpha ~ U(alpha_low, alpha_high) in arithmetic crossover
    alpha_high: float = 0.75

    max_ingredients: int = 18        # hard cap
    trim_below_grams: float = 0.5    # drop tiny amounts after mutation
    bloat_remove_boost: float = 0.25


# =========================
# Utility helpers
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def recipe_total_mass(ingredients_g: Genome) -> float:
    return sum(float(v) for v in ingredients_g.values())


def normalize_nonnegative(ingredients_g: Genome, min_grams: float = 0.0) -> Genome:
    """Ensure no negative grams; drop zero-and-below."""
    out = {}
    for k, v in ingredients_g.items():
        if v > min_grams:
            out[k] = v
    return out



# =========================
# Genetic Algorithm
# =========================

class GeneticAlgorithm:
    """
    A simple GA tailored to cookie recipes represented as:
        recipe = { "id", "name", "category", "ingredients_g": {ingredient_id: grams, ...} }

    You can plug in any fitness function: fitness(recipe) -> float (higher is better).
    By default we use a dummy fitness (random but recipe-dependent).
    """

    def __init__(
        self,
        initial_recipes: List[Recipe],
        fitness_fn: Optional[FitnessFn] = None,
        ga_cfg: Optional[GAConfig] = None,
        all_ingredient_ids: Optional[List[str]] = None,
    ):
        if not initial_recipes:
            raise ValueError("initial_recipes cannot be empty")

        self.cfg = ga_cfg or GAConfig()
        random.seed(self.cfg.random_seed)

        # If user didn't pass a fitness_fn, use dummy (random-ish + structure)
        self.fitness_fn: FitnessFn = fitness_fn or self._dummy_fitness

        # Full ingredient universe (for add/remove mutations).
        # If not provided, infer from all recipes' ingredient keys.
        if all_ingredient_ids is None:
            all_set = set()
            for r in initial_recipes:
                all_set.update(r["ingredients_g"].keys())
            self.all_ingredient_ids = sorted(all_set)
        else:
            self.all_ingredient_ids = list(all_ingredient_ids)

        # Build initial population to desired size
        self.population: List[Recipe] = self._bootstrap_population(initial_recipes, self.cfg.population_size)

    # -------- Core public API --------

    def run(self, verbose: bool = True) -> Tuple[Recipe, float]:
        """
        Run evolution for max_generations. Returns the best (recipe, fitness).
        """
        best = None
        best_fit = float("-inf")

        for gen in range(self.cfg.max_generations):

            if hasattr(self.fitness_fn, "set_reference"):
                self.fitness_fn.set_reference(self.population)

            # Evaluate
            scored = [(r, self.fitness_fn(r)) for r in self.population]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Track best
            if scored[0][1] > best_fit:
                best = copy.deepcopy(scored[0][0])
                best_fit = scored[0][1]

            if verbose:
                print(f"[Gen {gen+1:03d}] best={scored[0][1]:.4f}  mean={sum(s for _, s in scored)/len(scored):.4f}")

            # Build next generation
            next_pop = []

            # Elitism: carry top N unchanged
            # elites = [copy.deepcopy(r) for r, _ in scored[: self.cfg.elitism_n]]
            elites = []
            for r, _ in scored[: self.cfg.elitism_n]:
                e = dict(r)  # shallow copy
                e["ingredients_g"] = dict(r["ingredients_g"])  # copy only the heavy part once
                elites.append(e)
            next_pop.extend(elites)

            # Select survivors (parents pool)
            n_survivors = max(self.cfg.elitism_n, int(self.cfg.survivor_fraction * len(self.population)))
            parents_pool = [r for r, _ in scored[:n_survivors]]

            # Fill remainder with offspring
            while len(next_pop) < self.cfg.population_size:
                p1 = self._tournament_select(scored, k=self.cfg.tournament_k)
                p2 = self._tournament_select(scored, k=self.cfg.tournament_k)
                child = self._crossover(p1, p2) if random.random() < self.cfg.crossover_prob else copy.deepcopy(p1)
                child = self._mutate(child)
                next_pop.append(child)

            self.population = next_pop

        # Final evaluation
        final_scored = [(r, self.fitness_fn(r)) for r in self.population]
        final_scored.sort(key=lambda x: x[1], reverse=True)
        if final_scored[0][1] > best_fit:
            best, best_fit = final_scored[0]

        return best, best_fit

    # -------- Internals --------


    def pick_random_ingredient(self, exclude: set) -> Optional[str]:
        if not self.all_ingredient_ids:
            return None
        for _ in range(20):  # bounded retries
            cand = random.choice(self.all_ingredient_ids)
            if cand not in exclude:
                return cand
        return None

    def _bootstrap_population(self, seeds: List[Recipe], pop_size: int) -> List[Recipe]:
        """
        Start from the given seeds. If seeds < pop_size, sample with replacement and light-mutate
        to introduce diversity.
        """
        population: List[Recipe] = []
        if len(seeds) >= pop_size:
            # Shuffle and take first N
            shuffled = seeds[:]
            random.shuffle(shuffled)
            population = [copy.deepcopy(r) for r in shuffled[:pop_size]]
        else:
            # Copy seeds and clone/mutate to reach pop_size
            for r in seeds:
                population.append(copy.deepcopy(r))
            while len(population) < pop_size:
                base = copy.deepcopy(random.choice(seeds))
                population.append(self._mutate(base, weak=True))

        return population

    def _tournament_select(self, scored_pop: List[Tuple[Recipe, float]], k: int = 3) -> Recipe:
        """
        k-way tournament selection: pick k random individuals and return the fittest.
        """
        contestants = random.sample(scored_pop, k=min(k, len(scored_pop)))
        contestants.sort(key=lambda x: x[1], reverse=True)
        return contestants[0][0]

    def _crossover(self, p1: Recipe, p2: Recipe) -> Recipe:
        g1 = p1["ingredients_g"]
        g2 = p2["ingredients_g"]
        alpha = random.uniform(self.cfg.alpha_low, self.cfg.alpha_high)

        child_g: Genome = {}

        # Shared → blend; unique → optional inherit
        keys1, keys2 = set(g1.keys()), set(g2.keys())
        for k in (keys1 & keys2):
            v = alpha * float(g1[k]) + (1.0 - alpha) * float(g2[k])
            v = clamp(v, self.cfg.min_grams, self.cfg.max_grams)
            if v > 0:
                child_g[k] = v
        for k in (keys1 - keys2):
            if random.random() < 0.5:
                v = clamp(float(g1[k]), self.cfg.min_grams, self.cfg.max_grams)
                if v > 0:
                    child_g[k] = v
        for k in (keys2 - keys1):
            if random.random() < 0.5:
                v = clamp(float(g2[k]), self.cfg.min_grams, self.cfg.max_grams)
                if v > 0:
                    child_g[k] = v

        # _assert_ingredient_identity(keys1 | keys2, set(child_g.keys()))

        child = {
            "id": f"child_{random.randint(0, 10 ** 9)}",
            "name": f"cookie_{random.randint(0, 10 ** 6)}",  # short, non-recursive
            "category": p1.get("category", "cookie"),
            "ingredients_g": normalize_nonnegative(child_g, self.cfg.min_grams),
        }
        return child

    def _mutate(self, r: Recipe, weak: bool = False) -> Recipe:
        child = dict(r)
        g = dict(r["ingredients_g"])  # work on a copy
        before_keys = set(g.keys())

        total = max(1.0, recipe_total_mass(g))
        strength = self.cfg.mutation_strength * (0.5 if weak else 1.0)

        # Adjust amounts (add/subtract noise)
        for k in list(g.keys()):
            if random.random() < self.cfg.mutation_rate:
                noise = random.gauss(0.0, strength * total)
                new_v = clamp(float(g[k]) + noise, self.cfg.min_grams, self.cfg.max_grams)
                if new_v <= self.cfg.min_grams:
                    del g[k]
                else:
                    g[k] = new_v

        # Include: add a previously absent ingredient
        if random.random() < self.cfg.add_ingredient_prob and self.all_ingredient_ids:
            cand = self.pick_random_ingredient(exclude=set(g.keys()))
            if cand:
                g[cand] = clamp(abs(random.gauss(mu=10.0, sigma=10.0)), 1.0, 200.0)

        # Remove: delete an existing ingredient (no replacement)
        remove_p = self.cfg.remove_ingredient_prob
        if len(g) > self.cfg.max_ingredients:
            remove_p = min(1.0, remove_p + self.cfg.bloat_remove_boost)
        if random.random() < remove_p and len(g) > 1:
            del g[random.choice(list(g.keys()))]

        # Cleanup
        if self.cfg.trim_below_grams > 0:
            g = {k: v for k, v in g.items() if v >= self.cfg.trim_below_grams}
        if len(g) > self.cfg.max_ingredients:
            over = len(g) - self.cfg.max_ingredients
            keys = list(g.keys());
            random.shuffle(keys)
            for k in keys[:over]:
                del g[k]

        # _assert_ingredient_identity(before_keys, set(g.keys()))

        child["ingredients_g"] = g
        return child


def load_seed_recipes(path: str) -> List[Recipe]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["seed_recipes"]


def load_ingredient_ids(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    return None

def main():
    init_path = "configs/init_recipes.json"
    ingredients_path = "configs/ingredients.json"  # optional for add/remove variety

    # Load initial population and optional ingredient universe
    seeds = load_seed_recipes(init_path)
    all_ings = load_ingredient_ids(ingredients_path)

    # Configure GA
    cfg = GAConfig(
        max_generations=40,
        population_size=60,
        random_seed=7,
        tournament_k=3,
        survivor_fraction=0.6,
        elitism_n=3,
        crossover_prob=0.9,
        mutation_rate=0.25,
        mutation_strength=0.06,
        add_ingredient_prob=0.10,
        remove_ingredient_prob=0.10,
    )

    # Instantiate GA (using default dummy fitness for now)
    ga = GeneticAlgorithm(
        initial_recipes=seeds,
        fitness_fn=None,            # None → uses built-in dummy fitness
        ga_cfg=cfg,
        all_ingredient_ids=all_ings # enables add/remove ops
    )

    # Run evolution
    best_recipe, best_score = ga.run(verbose=True)

    # Show result
    print("\n=== BEST RECIPE ===")
    print("Name     :", best_recipe.get("name"))
    print("ID       :", best_recipe.get("id"))
    print("Category :", best_recipe.get("category"))
    print("Fitness  :", round(best_score, 4))
    print("Ingredients (g):")
    for k, v in sorted(best_recipe["ingredients_g"].items()):
        print(f"  - {k:20s} : {v:.1f} g")


if __name__ == "__main__":
    main()
