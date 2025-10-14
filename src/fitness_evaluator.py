from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

Recipe = Dict[str, Any]  # expects keys: id, name, category, ingredients_g


class RecipeFitness:

    BASE_ROLES = {"base", "fat", "sweetener", "liquid", "binder"}

    def __init__(
            self,
            ingredients_index: Dict[str, dict],
            overall_config: Dict[str, Any],
            flavor_pairings: List[dict],
            category_id: str = "cookie",
            weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.ING = ingredients_index
        self.CFG = overall_config
        self.category_id = category_id

        self.FP: Dict[Tuple[str, str], Dict[str, float]] = {}
        for row in flavor_pairings:
            a = str(row.get("flavor", "")).strip().lower()
            b = str(row.get("flavor2", "")).strip().lower()
            try:
                suc = float(row.get("avg_success", 0.0))
            except Exception:
                suc = 0.0
            try:
                rating = float(row.get("avg_rating", 0.0))
            except Exception:
                rating = 0.0
            try:
                cnt = float(row.get("pairing_count", 0))
            except Exception:
                cnt = 0.0
            self.FP[(a, b)] = {"avg_success": suc, "avg_rating": rating, "pairing_count": cnt}
            self.FP[(b, a)] = self.FP[(a, b)]

        # Category config
        cat_cfg = next(c for c in self.CFG["categories"] if c["id"] == self.category_id)
        self.ratio_constraints = cat_cfg["ratio_constraints"]
        self.size_constraints = cat_cfg["size_constraints"]["total_mass_g"]
        self.required_ing = set(cat_cfg["required_ingredients"])
        self.required_roles = set(cat_cfg["required_roles"])

        # Soft rules (pre-filtered)
        self.soft_rules = [
            r for r in self.CFG.get("validity", {}).get("soft_rules", [])
            if r.get("category") == self.category_id
        ]

        # Fast maps
        self.ING_ROLES = {k: set(v.get("roles", [])) for k, v in self.ING.items()}
        self.ING_FLAVORS = {k: [str(s).lower() for s in v.get("flavor_notes", [])] for k, v in self.ING.items()}

        # Aspect weights
        self.weights = weights or {
            "flavor_score": 1.30,  # blended: 0.6*soft + 0.4*pairing
            "pairing_score": 0.00,  # (keep 0 if using blended flavor_score)
            "soft_rules_score": 0.00,
            "ratio_score": 0.20,
            "novelty_score": 0.02,
            "simplicity_score": 0.00,
        }

        self._reference_recipes: List[Recipe] = []

    # ---------- Constructors ----------

    @classmethod
    def from_files(
            cls,
            ingredients_path: str,
            overall_cfg_path: str,
            flavor_pairings_path: str,
            category_id: str = "cookie",
            weights: Optional[Dict[str, float]] = None,
    ) -> "RecipeFitness":
        with open(ingredients_path, "r") as f:
            ING = json.load(f)
        with open(overall_cfg_path, "r") as f:
            CFG = json.load(f)
        with open(flavor_pairings_path, "r") as f:
            FP_LIST = json.load(f)
        return cls(ING, CFG, FP_LIST, category_id=category_id, weights=weights)

    # ---------- External API ----------

    def set_reference(self, recipes: List[Recipe]) -> None:
        self._reference_recipes = list(recipes) if recipes else []

    def __call__(self, recipe: Recipe) -> float:
        aspects = self.compute_aspects(recipe)
        return self.fitness_from_aspects(aspects)

    # ---------- Aspect computation ----------

    def compute_aspects(self, recipe: Recipe) -> Dict[str, Any]:
        ing = recipe.get("ingredients_g", {})
        valid, problems = self.recipe_validity(ing)

        pairing = self.flavor_pairing_score(ing)
        soft = self.soft_rules_score(ing)
        flavor = 0.4 * pairing + 0.6 * soft  # let soft rules steer, pairings refine

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

    # ---------- Validity & ratios ----------

    @staticmethod
    def recipe_total_mass(ingredients_g: Dict[str, float]) -> float:
        return float(sum(float(g) for g in ingredients_g.values()))

    def recipe_roles_breakdown(self, ingredients_g: Dict[str, float]):
        total = self.recipe_total_mass(ingredients_g)
        grams_by_role = defaultdict(float)
        for ing_id, grams in ingredients_g.items():
            for r in self.ING_ROLES.get(ing_id, []):
                grams_by_role[r] += float(grams)
        props_by_role = {r: (grams_by_role[r] / total if total > 0 else 0.0) for r in grams_by_role}
        return grams_by_role, props_by_role

    def recipe_validity(self, ingredients_g: Dict[str, float]) -> Tuple[bool, List[str]]:
        problems: List[str] = []
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

    def ratio_fit_score(self, ingredients_g: Dict[str, float]) -> float:
        _, props = self.recipe_roles_breakdown(ingredients_g)
        scores = [
            self.triangular_score(props.get(role, 0.0), float(rng["min"]), float(rng["max"]))
            for role, rng in self.ratio_constraints.items()
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    # ---------- Flavor pairing ----------

    def flavor_pairing_score(self, ingredients_g: Dict[str, float]) -> float:
        ing_list = sorted(ingredients_g.keys())
        character_roles = {"flavor", "inclusion"}

        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(ing_list)):
            for j in range(i + 1, len(ing_list)):
                ki, kj = ing_list[i], ing_list[j]
                roles_i = set(self.ING_ROLES.get(ki, []))
                roles_j = set(self.ING_ROLES.get(kj, []))
                #
                w = 1.5 if ((roles_i | roles_j) & character_roles) else 0.1
                # w = 1.5 if ((roles_i | roles_j) & character_roles) else 1.0

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
                num += s * w
                den += w
        return (num / den) if den > 0 else 0.0

    # ---------- Novelty ----------

    def _filtered_set_for_novelty(self, ing_map: Dict[str, float]) -> set:
        keep = set()
        for k, _ in ing_map.items():
            roles = set(self.ING_ROLES.get(k, []))
            if roles - self.BASE_ROLES:
                keep.add(k)
        return keep

    def novelty_score(self, target_ing: Dict[str, float], all_recipes: List[Recipe]) -> float:
        target_set = self._filtered_set_for_novelty(target_ing)
        if not target_set:
            return 0.5  # neutral if nothing but base roles

        best_sim = 0.0
        for r in all_recipes or []:
            s = self._filtered_set_for_novelty(r.get("ingredients_g", {}))
            if s == target_set:
                continue
            inter = len(target_set & s)
            union = len(target_set | s)
            sim = (inter / union) if union > 0 else 0.0
            best_sim = max(best_sim, sim)
        return 1.0 - best_sim

    # ---------- Soft rules ----------

    def _has_any(self, ing_ids: set, keys: List[str]) -> bool:
        return any(k in ing_ids for k in (keys or []))

    def _grams_of_any(self, ing_g: Dict[str, float], keys: List[str]) -> float:
        return sum(float(ing_g.get(k, 0.0)) for k in keys or [])

    def _role_grams(self, ing_g: Dict[str, float], role: str) -> float:
        total = 0.0
        for k, g in ing_g.items():
            if role in self.ING_ROLES.get(k, []):
                total += float(g)
        return total

    def _ratio_of_roles(self, ing_g: Dict[str, float], num_role: str, den_role: str) -> float:
        num = self._role_grams(ing_g, num_role)
        den = self._role_grams(ing_g, den_role)
        return (num / den) if den > 0 else 0.0

    def _score_threshold_preference(self, ing_g: Dict[str, float], rule: dict) -> Optional[float]:
        any_of = rule.get("any_of", [])
        min_grams = float(rule.get("min_grams", 0.0))
        ok = any(float(ing_g.get(k, 0.0)) >= min_grams for k in any_of)
        return 1.0 if ok else 0.0

    def _score_diversity_preference(self, ing_g: Dict[str, float], rule: dict) -> Optional[float]:
        roles_considered = set(rule.get("roles_considered", []))
        ignore_roles = set(rule.get("ignore_roles", []))
        lo, hi = map(float, rule.get("target_effective_count_range", [2, 4]))

        roles_present = set()
        for k, g in ing_g.items():
            if g <= 0:
                continue
            for r in self.ING_ROLES.get(k, []):
                if r in roles_considered and r not in ignore_roles:
                    roles_present.add(r)

        n = len(roles_present)
        if n < lo:
            return max(0.0, n / max(1.0, lo))
        if n > hi:
            return max(0.0, 1.0 - (n - hi) / max(1.0, hi))
        return 1.0

    def soft_rules_score(self, ingredients_g: Dict[str, float]) -> float:
        """Weighted aggregation of all applicable soft rules for the category."""
        if not self.soft_rules:
            return 0.0

        ing_ids = set(ingredients_g.keys())
        total_score, total_weight = 0.0, 0.0

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        for r in self.soft_rules:
            rtype = r.get("type")
            w = float(r.get("weight", 0.0))
            if w <= 0:
                continue

            local_score: Optional[float] = None

            if rtype == "ingredient_preference":
                prefer = r.get("prefer", [])
                local_score = 1.0 if self._has_any(ing_ids, prefer) else 0.0

            elif rtype == "ratio_preference":
                num = r.get("numerator_role")
                den = r.get("denominator_role")
                lo, hi = [float(x) for x in r.get("target_range", [0.0, 1.0])]
                ratio = self._ratio_of_roles(ingredients_g, num, den) if num and den else 0.0
                local_score = self.triangular_score(ratio, lo, hi)

            elif rtype == "pair_preference":
                if_any = r.get("if_any", [])
                prefer = r.get("prefer", [])
                gated = True if not if_any else self._has_any(ing_ids, if_any)
                if gated:
                    local_score = 1.0 if self._has_any(ing_ids, prefer) else 0.0

            elif rtype == "structure_preference":
                pen = float(r.get("penalty", 0.0))
                when_all = set(r.get("when_all", []))
                unless_any = set(r.get("unless_any", []))
                violates = False
                if when_all:
                    violates = set(when_all).issubset(ing_ids)
                else:
                    leaveners = {k for k in ing_ids if "leavener" in self.ING_ROLES.get(k, [])}
                    violates = (len(leaveners) >= 2)
                if unless_any and self._has_any(ing_ids, list(unless_any)):
                    violates = False
                local_score = 1.0 - pen if violates else 1.0

            elif rtype == "tag_preference":
                ing = r.get("ingredient")
                note = (r.get("note") or "").lower()
                if ing and ing in ing_ids:
                    if note in ("warm_spice", "spice", "spices"):
                        has_spice = any("spice" in self.ING_ROLES.get(k, []) for k in ing_ids)
                        local_score = 1.0 if has_spice else 0.0

            elif rtype == "process_preference":
                local_score = None  # no process data available

            elif rtype == "threshold_preference":
                local_score = self._score_threshold_preference(ingredients_g, r)

            elif rtype == "diversity_preference":
                local_score = self._score_diversity_preference(ingredients_g, r)

            if local_score is not None:
                total_score += clamp01(local_score) * w
                total_weight += w

        return float(total_score / total_weight) if total_weight > 0 else 0.0

    # ---------- Simplicity & final aggregation ----------

    @staticmethod
    def simplicity_score(ingredients_g: Dict[str, float],
                         ideal_lo: int = 6,
                         ideal_hi: int = 12,
                         min_n: int = 3,
                         max_n: int = 20) -> float:
        n = len([k for k, v in ingredients_g.items() if v > 0])
        if n < min_n or n > max_n:
            return 0.0
        if n <= ideal_lo:
            return (n - min_n) / max(1, (ideal_lo - min_n))
        if n >= ideal_hi:
            return (max_n - n) / max(1, (max_n - ideal_hi))
        return 1.0

    def fitness_from_aspects(self, aspects: Dict[str, Any]) -> float:
        if not aspects.get("valid", False):
            return 0.0
        used = []
        for k, w in (self.weights or {}).items():
            if k in aspects and isinstance(w, (int, float)) and w > 0:
                used.append((k, max(0.0, min(1.0, float(w)))))
        if not used:
            return 0.0
        num = sum(aspects[k] * w for k, w in used)
        den = sum(w for _, w in used)
        return float(num / den) if den > 0 else 0.0
