# 🍪 Survival of the Sweetest
**Computational Creativity – Assignment 1**

This project evolves cookie recipes using a **Genetic Algorithm (GA)** to explore *computational creativity*.

---

## 🧩 Files Overview
| File | Description |
|------|--------------|
| `ingredients.json` | Ingredient ontology (roles, category, flavour notes). |
| `init_recipes.json` | Starting inspiration set (seed recipes). |
| `flavor_pairings.json` | Empirical flavour compatibility data. |
| `overall_configs.json` | Global category rules and ratio constraints. |
| `fitness_evaluator.py` | Computes fitness from structural and flavour aspects. |
| `genetic_algorithm.py` | Core GA implementation (selection, crossover, mutation). |
| `run_ga.py` | Entry point for running the full experiment. |
| `result.txt` | Example run output with best evolved recipes. |

---

## ⚙️ How to Run
```bash
# 1. Ensure dependencies (Python ≥3.10, pandas, matplotlib)
pip install -r requirements.txt

# 2. Run the evolutionary process
python run_ga.py

