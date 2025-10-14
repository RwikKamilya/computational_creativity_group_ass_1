import json, math, itertools, statistics
import pandas as pd
from collections import defaultdict

# ---------- Load data ----------
with open("configs/ingredients.json") as f:
    ING = json.load(f)

with open("configs/overall_configs.json") as f:
    CFG = json.load(f)

with open("configs/init_recipes.json") as f:
    REC = json.load(f)["seed_recipes"]

with open("configs/flavor_pairings.json") as f:
    FP_LIST = json.load(f)

# Build fast lookup for flavor pairings (both orders)
FP = {}
for row in FP_LIST:
    f1 = str(row["flavor"]).strip().lower()
    f2 = str(row["flavor2"]).strip().lower()
    try: suc = float(row.get("avg_success", 0))
    except: suc = 0.0
    try: rating = float(row.get("avg_rating", 0))
    except: rating = 0.0
    try: cnt = int(row.get("pairing_count", 0))
    except: cnt = 0
    FP[(f1, f2)] = {"avg_success": suc, "avg_rating": rating, "pairing_count": cnt}
    FP[(f2, f1)] = FP[(f1, f2)]

# Config helpers
cookie_cfg = next(c for c in CFG["categories"] if c["id"] == "cookie")
ratio_constraints = cookie_cfg["ratio_constraints"]
size_constraints = cookie_cfg["size_constraints"]["total_mass_g"]
required_ing = set(cookie_cfg["required_ingredients"])
required_roles = set(cookie_cfg["required_roles"])

ING_ROLES   = {k: set(v.get("roles", [])) for k, v in ING.items()}
ING_FLAVORS = {k: [s.lower() for s in v.get("flavor_notes", [])] for k, v in ING.items()}

# ---------- Scoring utilities ----------
def recipe_total_mass(ingredients_g: dict) -> float:
    return float(sum(float(g) for g in ingredients_g.values()))

def recipe_roles_breakdown(ingredients_g: dict):
    total = recipe_total_mass(ingredients_g)
    grams_by_role = defaultdict(float)
    for ing_id, grams in ingredients_g.items():
        for r in ING_ROLES.get(ing_id, []):
            grams_by_role[r] += float(grams)
    props_by_role = {r: (grams_by_role[r] / total if total > 0 else 0.0) for r in grams_by_role}
    return grams_by_role, props_by_role

def recipe_validity(ingredients_g: dict):
    problems = []
    ing_ids = set(ingredients_g.keys())

    missing_ing = sorted(list(required_ing - ing_ids))
    if missing_ing:
        problems.append(f"Missing required ingredients: {', '.join(missing_ing)}")

    grams_by_role, _ = recipe_roles_breakdown(ingredients_g)
    missing_roles = [r for r in required_roles if grams_by_role.get(r, 0) <= 0]
    if missing_roles:
        problems.append(f"Missing required roles: {', '.join(missing_roles)}")

    total = recipe_total_mass(ingredients_g)
    if not (size_constraints["min"] <= total <= size_constraints["max"]):
        problems.append(f"Total mass {total:.1f}g outside [{size_constraints['min']}, {size_constraints['max']}] g")

    return (len(problems) == 0), problems

def triangular_score(value: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    if value <= lo or value >= hi: return 0.0
    mid = (lo + hi) / 2.0
    if value == mid: return 1.0
    return (value - lo) / (mid - lo) if value < mid else (hi - value) / (hi - mid)

def ratio_fit_score(ingredients_g: dict) -> float:
    _, props = recipe_roles_breakdown(ingredients_g)
    scores = [triangular_score(props.get(role, 0.0), float(rng["min"]), float(rng["max"]))
              for role, rng in ratio_constraints.items()]
    return float(sum(scores)/len(scores)) if scores else 0.0

def flavor_pairing_score(ingredients_g: dict) -> float:
    ing_list = list(ingredients_g.keys())
    pairs = []
    for i in range(len(ing_list)):
        for j in range(i+1, len(ing_list)):
            fi = ING_FLAVORS.get(ing_list[i], [])
            fj = ING_FLAVORS.get(ing_list[j], [])
            for a in fi:
                for b in fj:
                    pairs.append((a, b))
    scores = []
    for a, b in pairs:
        data = FP.get((a, b))
        if data:
            suc = max(0.0, min(1.0, float(data["avg_success"])))
            r_norm = max(0.0, min(1.0, float(data["avg_rating"]) / 5.0))
            scores.append((suc + r_norm) / 2.0)
    return float(sum(scores)/len(scores)) if scores else 0.0

def novelty_score(target_ing: dict, all_recipes: list[dict]) -> float:
    target_set = set(target_ing.keys())
    best_sim = 0.0
    for r in all_recipes:
        s = set(r["ingredients_g"].keys())
        if s is target_set: continue
        inter = len(target_set & s)
        union = len(target_set | s)
        sim = (inter/union) if union > 0 else 0.0
        best_sim = max(best_sim, sim)
    return 1.0 - best_sim

def simplicity_score(ingredients_g: dict, ideal_lo=6, ideal_hi=12, min_n=3, max_n=20) -> float:
    n = len([k for k,v in ingredients_g.items() if v > 0])
    if n < min_n or n > max_n: return 0.0
    if n <= ideal_lo: return (n - min_n) / max(1, (ideal_lo - min_n))
    if n >= ideal_hi: return (max_n - n) / max(1, (max_n - ideal_hi))
    return 1.0

def compute_aspects(recipe: dict, all_recipes: list[dict]) -> dict:
    ing = recipe["ingredients_g"]
    valid, problems = recipe_validity(ing)
    aspects = {
        "valid": valid,
        "problems": problems,
        "ratio_score": ratio_fit_score(ing),
        "pairing_score": flavor_pairing_score(ing),
        "novelty_score": novelty_score(ing, all_recipes),
        "simplicity_score": simplicity_score(ing)
    }
    # For now, "flavor_score" == pairing-based score
    aspects["flavor_score"] = aspects["pairing_score"]
    return aspects

def fitness_from_aspects(aspects: dict, weights: dict) -> float:
    if not aspects["valid"]:
        return 0.0
    used = [(k, max(0.0, min(1.0, float(w)))) for k, w in weights.items()
            if k in aspects and isinstance(w, (int, float)) and w > 0]
    if not used: return 0.0
    num = sum(aspects[k]*w for k, w in used)
    den = sum(w for _, w in used)
    return float(num/den) if den > 0 else 0.0

# ---- Tune your weights here (all 0..1) ----
weights = {
    "flavor_score":   0.75,
    "ratio_score":    0.75,
    "novelty_score":  0.50,
    "simplicity_score": 0.25
}

# ---------- Run scoring ----------
rows = []
for r in REC:
    asp = compute_aspects(r, REC)
    fit = fitness_from_aspects(asp, weights)
    rows.append({
        "id": r["id"],
        "name": r["name"],
        "valid": asp["valid"],
        "problems": "; ".join(asp["problems"]),
        "flavor_score": round(asp["flavor_score"], 3),
        "ratio_score": round(asp["ratio_score"], 3),
        "novelty_score": round(asp["novelty_score"], 3),
        "simplicity_score": round(asp["simplicity_score"], 3),
        "fitness": round(fit, 3)
    })

df = pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)
print(df.head(10))
df.to_csv("recipe_aspect_scores.csv", index=False)
print("Saved: recipe_aspect_scores.csv")
