# fitness_evaluator.py
import json, math, itertools, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

Recipe = Dict[str, any]  # expects keys: id, name, category, ingredients_g


class RecipeFitness:
    """
    Domain-aware fitness evaluator for cookie recipes.
    Usage:
        fit = RecipeFitness.from_files(
            ingredients_path="configs/ingredients.json",
            overall_cfg_path="configs/overall_configs.json",
            flavor_pairings_path="configs/flavor_pairings.json",
            category_id="cookie",
            weights={
                "flavor_score":   0.75,
                "ratio_score":    0.75,
                "novelty_score":  0.50,
                "simplicity_score": 0.25
            },
        )
        fit.set_reference(initial_population)   # call once per generation
        score = fit(recipe)                     # __call__ returns scalar fitness
        aspects = fit.compute_aspects(recipe)   # optional detailed breakdown
    """

    # ---------- Constructors ----------
    def __init__(
            self,
            ING: Dict[str, dict],
            CFG: Dict[str, any],
            FP_LIST: List[dict],
            category_id: str = "cookie",
            weights: Optional[Dict[str, float]] = None,
    ):
        self.ING = ING
        self.CFG = CFG
        self.category_id = category_id

        # Flavor pairings lookup (bidirectional)
        self.FP = {}
        for row in FP_LIST:
            f1 = str(row.get("flavor", "")).strip().lower()
            f2 = str(row.get("flavor2", "")).strip().lower()
            try:
                suc = float(row.get("avg_success", 0))
            except:
                suc = 0.0
            try:
                rating = float(row.get("avg_rating", 0))
            except:
                rating = 0.0
            try:
                cnt = int(row.get("pairing_count", 0))
            except:
                cnt = 0
            self.FP[(f1, f2)] = {"avg_success": suc, "avg_rating": rating, "pairing_count": cnt}
            self.FP[(f2, f1)] = self.FP[(f1, f2)]

        # Category-specific config
        cookie_cfg = next(c for c in self.CFG["categories"] if c["id"] == self.category_id)
        self.ratio_constraints = cookie_cfg["ratio_constraints"]
        self.size_constraints = cookie_cfg["size_constraints"]["total_mass_g"]
        self.required_ing = set(cookie_cfg["required_ingredients"])
        self.required_roles = set(cookie_cfg["required_roles"])

        validity_cfg = self.CFG.get("validity", {})
        self.soft_rules = [
            r for r in validity_cfg.get("soft_rules", [])
            if r.get("category") == self.category_id
        ]

        # Fast maps
        self.ING_ROLES = {k: set(v.get("roles", [])) for k, v in self.ING.items()}
        self.ING_FLAVORS = {k: [str(s).lower() for s in v.get("flavor_notes", [])] for k, v in self.ING.items()}

        # Weights (0..1, normalized internally)
        self.weights = weights or {
            "flavor_score": 0.75,
            "ratio_score": 0.75,
            "novelty_score": 0.50,
            "simplicity_score": 0.25,
        }

        # Reference set for novelty scoring (set via set_reference)
        self._reference_recipes: List[Recipe] = []

    @classmethod
    def from_files(
            cls,
            ingredients_path: str,
            overall_cfg_path: str,
            flavor_pairings_path: str,
            category_id: str = "cookie",
            weights: Optional[Dict[str, float]] = None,
    ) -> "RecipeFitness":
        with open(ingredients_path) as f:
            ING = json.load(f)
        with open(overall_cfg_path) as f:
            CFG = json.load(f)
        with open(flavor_pairings_path) as f:
            FP_LIST = json.load(f)
        return cls(ING=ING, CFG=CFG, FP_LIST=FP_LIST, category_id=category_id, weights=weights)

    # ---------- Public API ----------
    def set_reference(self, recipes: List[Recipe]) -> None:
        """Provide a pool (e.g., current population) for novelty comparisons."""
        self._reference_recipes = list(recipes) if recipes else []

    def __call__(self, recipe: Recipe) -> float:
        """Scalar fitness used by the GA."""
        aspects = self.compute_aspects(recipe)
        return self.fitness_from_aspects(aspects)

    def compute_aspects(self, recipe: Recipe) -> dict:
        ing = recipe.get("ingredients_g", {})
        valid, problems = self.recipe_validity(ing)

        pairing = self.flavor_pairing_score(ing)
        soft = self.soft_rules_score(ing)
        flavor = 0.4 * pairing + 0.6 * soft  # let rules steer, pairings refine

        aspects = {
            "valid": valid,
            "problems": problems,
            "ratio_score": self.ratio_fit_score(ing),
            "pairing_score": pairing,
            "soft_rules_score": soft,
            "novelty_score": self.novelty_score(ing, self._reference_recipes),
            "simplicity_score": self.simplicity_score(ing),
        }
        aspects["flavor_score"] = flavor
        return aspects

    # ---------- Internals: scoring primitives ----------
    @staticmethod
    def recipe_total_mass(ingredients_g: dict) -> float:
        return float(sum(float(g) for g in ingredients_g.values()))

    def recipe_roles_breakdown(self, ingredients_g: dict):
        total = self.recipe_total_mass(ingredients_g)
        grams_by_role = defaultdict(float)
        for ing_id, grams in ingredients_g.items():
            for r in self.ING_ROLES.get(ing_id, []):
                grams_by_role[r] += float(grams)
        props_by_role = {r: (grams_by_role[r] / total if total > 0 else 0.0) for r in grams_by_role}
        return grams_by_role, props_by_role

    def recipe_validity(self, ingredients_g: dict) -> Tuple[bool, List[str]]:
        problems = []
        ing_ids = set(ingredients_g.keys())

        # required ingredients present?
        missing_ing = sorted(list(self.required_ing - ing_ids))
        if missing_ing:
            problems.append(f"Missing required ingredients: {', '.join(missing_ing)}")

        # required roles covered?
        grams_by_role, _ = self.recipe_roles_breakdown(ingredients_g)
        missing_roles = [r for r in self.required_roles if grams_by_role.get(r, 0) <= 0]
        if missing_roles:
            problems.append(f"Missing required roles: {', '.join(missing_roles)}")

        # total mass bounds?
        total = self.recipe_total_mass(ingredients_g)
        lo, hi = float(self.size_constraints["min"]), float(self.size_constraints["max"])
        if not (lo <= total <= hi):
            problems.append(f"Total mass {total:.1f}g outside [{lo}, {hi}] g")

        return (len(problems) == 0), problems

    @staticmethod
    def triangular_score(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        if value <= lo or value >= hi:
            return 0.0
        mid = (lo + hi) / 2.0
        if value == mid:
            return 1.0
        return (value - lo) / (mid - lo) if value < mid else (hi - value) / (hi - mid)

    def ratio_fit_score(self, ingredients_g: dict) -> float:
        _, props = self.recipe_roles_breakdown(ingredients_g)
        scores = [
            self.triangular_score(props.get(role, 0.0), float(rng["min"]), float(rng["max"]))
            for role, rng in self.ratio_constraints.items()
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def flavor_pairing_score(self, ingredients_g: dict) -> float:
        ing_list = list(ingredients_g.keys())
        character_roles = {"spice", "chocolate", "extract", "nut", "fruit", "inclusion"}

        pairs = []
        for i in range(len(ing_list)):
            for j in range(i + 1, len(ing_list)):
                ki, kj = ing_list[i], ing_list[j]
                roles_i = set(self.ING_ROLES.get(ki, []))
                roles_j = set(self.ING_ROLES.get(kj, []))
                w = 1.5 if ((roles_i | roles_j) & character_roles) else 1.0

                for a in self.ING_FLAVORS.get(ki, []):
                    for b in self.ING_FLAVORS.get(kj, []):
                        pairs.append((a.lower(), b.lower(), w))

        num, den = 0.0, 0.0
        for a, b, w in pairs:
            data = self.FP.get((a, b))
            if data:
                suc = max(0.0, min(1.0, float(data["avg_success"])))
                r_norm = max(0.0, min(1.0, float(data["avg_rating"]) / 5.0))
                s = (suc + r_norm) / 2.0
                num += s * w;
                den += w
        return (num / den) if den > 0 else 0.0

    # in RecipeFitness
    # put near the class top
    BASE_ROLES = {"base", "fat", "sweetener", "liquid", "binder"}

    def _filtered_set_for_novelty(self, ing_map: dict) -> set:
        keep = set()
        for k, g in ing_map.items():
            roles = set(self.ING_ROLES.get(k, []))
            # only keep character/non-base ingredients (spice, chocolate, extract, nuts, etc.)
            if roles - self.BASE_ROLES:
                keep.add(k)
        return keep

    def novelty_score(self, target_ing: dict, all_recipes: List[Recipe]) -> float:
        target_set = self._filtered_set_for_novelty(target_ing)
        if not target_set:
            return 0.5  # neutral if nothing but base roles

        best_sim = 0.0
        for r in all_recipes or []:
            s = self._filtered_set_for_novelty(r.get("ingredients_g", {}))
            # skip *exact* same set (use equality, not identity)
            if s == target_set:
                continue
            inter = len(target_set & s);
            union = len(target_set | s)
            sim = (inter / union) if union > 0 else 0.0
            if sim > best_sim:
                best_sim = sim
        return 1.0 - best_sim

    @staticmethod
    def simplicity_score(ingredients_g: dict, ideal_lo=6, ideal_hi=12, min_n=3, max_n=20) -> float:
        n = len([k for k, v in ingredients_g.items() if v > 0])
        if n < min_n or n > max_n:
            return 0.0
        if n <= ideal_lo:
            return (n - min_n) / max(1, (ideal_lo - min_n))
        if n >= ideal_hi:
            return (max_n - n) / max(1, (max_n - ideal_hi))
        return 1.0

    def fitness_from_aspects(self, aspects: dict) -> float:
        if not aspects.get("valid", False):
            return 0.0
        # sanitize & normalize weights
        used = []
        for k, w in (self.weights or {}).items():
            if k in aspects and isinstance(w, (int, float)) and w > 0:
                used.append((k, max(0.0, min(1.0, float(w)))))
        if not used:
            return 0.0
        num = sum(aspects[k] * w for k, w in used)
        den = sum(w for _, w in used)
        return float(num / den) if den > 0 else 0.0

    def _has_any(self, ing_ids: set, keys: List[str]) -> bool:
        return any(k in ing_ids for k in keys or [])

    def _grams_of_any(self, ing_g: dict, keys: List[str]) -> float:
        return sum(float(ing_g.get(k, 0.0)) for k in keys or [])

    def _role_grams(self, ing_g: dict, role: str) -> float:
        total = 0.0
        for k, g in ing_g.items():
            if role in self.ING_ROLES.get(k, []):
                total += float(g)
        return total

    def _ratio_of_roles(self, ing_g: dict, num_role: str, den_role: str) -> float:
        num = self._role_grams(ing_g, num_role)
        den = self._role_grams(ing_g, den_role)
        return (num / den) if den > 0 else 0.0

    def _score_threshold_preference(self, ing_g: dict, rule: dict) -> Optional[float]:
        any_of = rule.get("any_of", [])
        min_grams = float(rule.get("min_grams", 0.0))
        ok = any(float(ing_g.get(k, 0.0)) >= min_grams for k in any_of)
        return 1.0 if ok else 0.0

    def _score_diversity_preference(self, ing_g: dict, rule: dict) -> Optional[float]:
        roles_considered = set(rule.get("roles_considered", []))
        ignore_roles = set(rule.get("ignore_roles", []))
        lo, hi = map(float, rule.get("target_effective_count_range", [2, 4]))

        # collect roles present
        roles_present = set()
        for k, g in ing_g.items():
            if g <= 0:
                continue
            for r in self.ING_ROLES.get(k, []):
                if r in roles_considered and r not in ignore_roles:
                    roles_present.add(r)

        n = len(roles_present)
        # triangular “bell” peaking inside [lo, hi]
        if n < lo:
            return max(0.0, n / max(1.0, lo))
        if n > hi:
            return max(0.0, 1.0 - (n - hi) / max(1.0, hi))
        return 1.0

    def soft_rules_score(self, ingredients_g: dict) -> float:
        """
        Combine all applicable soft rules to a single 0..1 score.
        Rules with 'weight' contribute proportionally.
        Rules with 'penalty' reduce the local score when violated.
        Non-applicable rules don’t count toward the denominator.
        """
        if not getattr(self, "soft_rules", None):
            return 0.0

        ing_ids = set(ingredients_g.keys())
        per_rule_scores = []
        per_rule_weights = []

        for r in self.soft_rules:
            rtype = r.get("type")

            # default: rule contributes only when applicable
            local_score = None

            if rtype == "ingredient_preference":
                prefer = r.get("prefer", [])
                if prefer:  # applicable always; reward presence
                    local_score = 1.0 if self._has_any(ing_ids, prefer) else 0.0

            elif rtype == "ratio_preference":
                num = r.get("numerator_role")
                den = r.get("denominator_role")
                tr = r.get("target_range", [0.0, 0.0])
                if num and den and tr:
                    ratio = self._ratio_of_roles(ingredients_g, num, den)
                    local_score = self.triangular_score(ratio, float(tr[0]), float(tr[1]))


            elif rtype == "pair_preference":
                if_any = r.get("if_any", [])
                prefer = r.get("prefer", [])
                gated = True if not if_any else any(x in ing_ids for x in if_any)
                if gated:
                    local_score = 1.0 if any(x in ing_ids for x in prefer) else 0.0
                else:
                    local_score = 0.5  # neutral when not applicable



            elif rtype == "structure_preference":
                pen = float(r.get("penalty", 0.0))
                when_all = set(r.get("when_all", []))
                unless_any = set(r.get("unless_any", []))
                violates = False
                ing_ids = set(ingredients_g.keys())
                if when_all:
                    violates = set(when_all).issubset(ing_ids)
                else:
                    # consider either explicit ids or role==riser
                    leaveners_ids = {"baking_powder", "baking_soda"} & ing_ids
                    # plus anything tagged with the 'riser' role
                    leaveners_roles = {k for k in ing_ids if "riser" in self.ING_ROLES.get(k, [])}
                    leaveners = leaveners_ids | leaveners_roles
                    violates = (len(leaveners) >= 2)

                if unless_any and any(x in ing_ids for x in unless_any):
                    violates = False

                local_score = 1.0 - pen if violates else 1.0


            elif rtype == "tag_preference":
                # Simple interpretation: if the anchor ingredient is present,
                # reward presence of any ingredient whose role matches the 'note' hint set.
                ing = r.get("ingredient")
                note = (r.get("note") or "").lower()
                if ing and ing in ing_ids:
                    # crude mapping: warm_spice → any spice role
                    if note in ("warm_spice", "spice", "spices"):
                        has_spice = any("spice" in self.ING_ROLES.get(k, []) for k in ing_ids)
                        local_score = 1.0 if has_spice else 0.0

            elif rtype == "process_preference":
                # No process data in recipes; ignore (neutral / not applicable)
                local_score = None

            # ---- new rule types ----
            elif rtype == "threshold_preference":
                local_score = self._score_threshold_preference(ingredients_g, r)

            elif rtype == "diversity_preference":
                local_score = self._score_diversity_preference(ingredients_g, r)

            # Accumulate only if applicable / computed
            if local_score is not None:
                w = float(r.get("weight", 1.0))
                per_rule_scores.append(max(0.0, min(1.0, local_score)) * w)
                per_rule_weights.append(w)

        if not per_rule_weights:
            return 0.0
        return float(sum(per_rule_scores) / sum(per_rule_weights))

    def _role_grams(self, ing_g: dict, role: str) -> float:
        role = (role or "").lower()

        # Composite / pseudo roles used by soft rules
        if role == "sweetener_total":
            total = 0.0
            for k, g in ing_g.items():
                if "sweetener" in self.ING_ROLES.get(k, []):
                    total += float(g)
            # also count common syrups/molasses if not tagged in roles
            for k in ("molasses", "honey", "maple_syrup"):
                total += float(ing_g.get(k, 0.0))
            return total

        if role == "sweetener_brown":
            return float(ing_g.get("sugar_brown", 0.0)) + float(ing_g.get("molasses", 0.0))

        # Default: sum grams of any ingredient with that role
        total = 0.0
        for k, g in ing_g.items():
            if role in self.ING_ROLES.get(k, []):
                total += float(g)
        return total
