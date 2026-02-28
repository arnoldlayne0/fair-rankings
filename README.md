# Fair Rankings

Algorithms for fairness-constrained ranking maximisation, originally developed as part of an dissertation at the London School of Economics (MSc Operations Research, 2018).

## Problem

Given a set of items with quality scores and ranked positions with varying exposure (e.g. news articles in a feed), how do we produce a ranking that maximises total utility while ensuring fair representation of different groups?

This package implements two formulations:

- **Top-k Constrained Ranking Maximisation**: bounds on the number of items from each group in the top-k positions, solved via a greedy algorithm (optimal under the Monge property with disjoint groups).
- **Parity-Constrained Ranking Maximisation (PCRM)**: bounds on total exposure allocated to each group, solved exactly via integer programming (OR-Tools CBC) and approximately via a greedy heuristic.

Two fairness notions are supported:
- **Demographic parity**: exposure proportional to group size.
- **Disparate treatment**: exposure proportional to group quality.

## Structure

```
fair_rankings/
├── __init__.py
├── data.py          # Data generation, YOW dataset loading, fairness bounds
├── algorithms.py    # Ranking algorithms and fairness metrics
└── simulations.py   # Simulation harness for empirical analysis
tests/
└── test_algorithms.py
```

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from fair_rankings import data as dg, algorithms as ra

# Generate a random instance
pos_imp, item_qual, properties = dg.sim_data(m=20, n=10, p=3, distinct=True)
prop_list = dg.get_prop_list(properties)

# Compute demographic parity bounds (with 20% slack)
parity = dg.parity_pcrm(prop_list, item_qual, "demographic", viol_coeff=1.2)

# Solve exactly via IP
result_ip = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
print(f"IP objective: {result_ip.obj_value:.4f}")

# Solve approximately via greedy
result_greedy = ra.greedy_parity(item_qual, pos_imp, properties, parity)
print(f"Greedy objective: {result_greedy.value:.4f}")
```

## Running tests

```bash
pytest
```

## Simulation experiments

The `simulations` module reproduces the dissertation experiments:

```python
from fair_rankings import simulations as sim

# Runtime benchmarks
df_time = sim.sim_pcrm_time(times=100, notion="demographic")

# Price of fairness
df_price = sim.price_of_fairness(times=100, notion="demographic")

# Two-group analysis
df_twogroup = sim.twogroup_sim_mult_uniform(m=30, n=20, times=100, notion="demographic", viol=1.2)
```

## Key references

- Celis, Straszak, Vishnoi (2018). "Ranking with Fairness Constraints"
- Singh, Joachims (2018). "Fairness of Exposure in Rankings"
- Biega, Gummadi, Weikum (2018). "Equity of Attention"
- Zehlike et al. (2017). "FA*IR: A Fair Top-k Ranking Algorithm"
