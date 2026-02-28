"""Ranking algorithms for constrained ranking maximisation.

Implements:
- Top-k constrained ranking (greedy)
- Parity-constrained ranking maximisation (IP, LP relaxation, greedy)
- Unconstrained ranking (max-weight matching, trivial)
- Fairness metrics (demographic parity ratio, disparate treatment ratio)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import networkx as nx

from . import data as dg


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingResult:
    """Result of a ranking algorithm.

    Attributes
    ----------
    ranking : list[int] | None
        Item indices in rank order, or None if infeasible.
    value : float
        Objective value (0 if infeasible).
    feasible : bool
        Whether a feasible ranking was found.
    """

    ranking: list[int] | None
    value: float
    feasible: bool


@dataclass(frozen=True)
class LPResult:
    """Result of an LP/IP ranking solver.

    Attributes
    ----------
    success : bool
        Whether the solver found an optimal solution.
    feasible : bool
        Whether the solution is feasible (all positions filled).
    obj_value : float
        Objective function value.
    solution : list[float]
        Raw solution vector.
    desc_sol : list[list]
        Labelled solution vector (variable name, value).
    n_vars : int
        Number of variables in the formulation.
    n_constraints : int
        Number of constraints.
    """

    success: bool
    feasible: bool
    obj_value: float
    solution: list[float]
    desc_sol: list[list]
    n_vars: int
    n_constraints: int


# ---------------------------------------------------------------------------
# Top-k constrained ranking
# ---------------------------------------------------------------------------

def greedy_topk(
    properties: np.ndarray | pd.DataFrame,
    up_bounds: np.ndarray,
    pos_imp: np.ndarray,
    item_qual: np.ndarray,
) -> RankingResult:
    """Greedy algorithm for top-k constrained ranking maximisation.

    Solves the constrained ranking problem where bounds limit the number
    of items with a given property in the top-k positions.

    Assumes items are sorted in decreasing quality and the Monge property holds.

    Parameters
    ----------
    properties : array-like of shape (m, p)
        Binary property matrix.
    up_bounds : array-like of shape (p, n)
        ``up_bounds[l][k]`` = max items with property ``l`` in top ``k+1``.
    pos_imp : array-like of shape (n,)
        Position importance values.
    item_qual : array-like of shape (m,)
        Item quality values (sorted descending).

    Returns
    -------
    RankingResult
    """
    if isinstance(properties, pd.DataFrame):
        properties = properties.to_numpy()

    no_pos = len(pos_imp)
    no_items = len(item_qual)
    ranking: list[int] = []
    up_bounds = np.array(up_bounds).T
    curr_sat = np.zeros(properties.shape[1])

    for pos in range(no_pos):
        considered = 0
        placed = False
        while not placed and considered < no_items:
            if considered in ranking:
                considered += 1
            elif (curr_sat + properties[considered] <= up_bounds[pos]).all():
                ranking.append(considered)
                curr_sat += properties[considered]
                placed = True
            else:
                considered += 1

        if not placed:
            return RankingResult(ranking=None, value=0.0, feasible=False)

    total_value = sum(
        pos_imp[i] * item_qual[ranking[i]] for i in range(len(ranking))
    )
    return RankingResult(ranking=ranking, value=total_value, feasible=True)


# ---------------------------------------------------------------------------
# Parity-constrained ranking maximisation (IP / LP)
# ---------------------------------------------------------------------------

def _solve_parity(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
    prop_list: list[list[int]],
    parity: np.ndarray,
    relaxation: bool = False,
) -> LPResult:
    """Solve the parity-constrained ranking problem via IP or LP relaxation.

    Parameters
    ----------
    item_qual : array-like
        Item quality values.
    pos_imp : array-like
        Position importance values.
    prop_list : list of list of int
        Groups of item indices.
    parity : array-like
        Maximum exposure fractions per group.
    relaxation : bool
        If True, solve the LP relaxation. Otherwise solve the IP.

    Returns
    -------
    LPResult
    """
    from ortools.linear_solver import pywraplp

    if relaxation:
        solver = pywraplp.Solver(
            "FairRankingLP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
        )
    else:
        solver = pywraplp.Solver(
            "FairRankingIP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )

    m, n, p = len(item_qual), len(pos_imp), len(prop_list)
    val_list, bounds, A = dg.gen_data_lp(item_qual, pos_imp, prop_list, parity)

    # Decision variables
    variables = []
    objective = solver.Objective()
    for i in range(m * n):
        if relaxation:
            var = solver.NumVar(0.0, 1.0, val_list[i][0])
        else:
            var = solver.IntVar(0.0, solver.infinity(), val_list[i][0])
        variables.append(var)
        objective.SetCoefficient(var, val_list[i][1])
    objective.SetMaximization()

    # Constraints
    constraints = []

    # Each item used at most once
    for i in range(m):
        ct = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(m * n):
            ct.SetCoefficient(variables[j], A[i][j])
        constraints.append(ct)

    # Each position filled exactly once
    for i in range(m, m + n):
        ct = solver.Constraint(bounds[i][1], bounds[i][1])
        for j in range(m * n):
            ct.SetCoefficient(variables[j], A[i][j])
        constraints.append(ct)

    # Parity constraints
    for i in range(m + n, m + n + p):
        ct = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(m * n):
            ct.SetCoefficient(variables[j], A[i][j])
        constraints.append(ct)

    result_status = solver.Solve()

    success = result_status == solver.OPTIMAL
    solution = [var.solution_value() for var in variables]
    feasible = sum(solution) >= n
    obj_value = solver.Objective().Value()
    desc_sol = [[var.name(), var.solution_value()] for var in variables]

    return LPResult(
        success=success,
        feasible=feasible,
        obj_value=obj_value,
        solution=solution,
        desc_sol=desc_sol,
        n_vars=solver.NumVariables(),
        n_constraints=solver.NumConstraints(),
    )


def ip_parity(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
    prop_list: list[list[int]],
    parity: np.ndarray,
) -> LPResult:
    """Solve parity-constrained ranking as an Integer Program (exact)."""
    return _solve_parity(item_qual, pos_imp, prop_list, parity, relaxation=False)


def lp_parity(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
    prop_list: list[list[int]],
    parity: np.ndarray,
) -> LPResult:
    """Solve the LP relaxation of parity-constrained ranking."""
    return _solve_parity(item_qual, pos_imp, prop_list, parity, relaxation=True)


# ---------------------------------------------------------------------------
# Parity-constrained ranking (greedy approximation)
# ---------------------------------------------------------------------------

def greedy_parity(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
    properties: np.ndarray | pd.DataFrame,
    parity: np.ndarray,
) -> RankingResult:
    """Greedy approximation for parity-constrained ranking maximisation.

    Uses break-based inner loop (faster empirically).
    Items must be sorted in decreasing quality order.

    Parameters
    ----------
    item_qual : array-like of shape (m,)
        Item qualities (sorted descending).
    pos_imp : array-like of shape (n,)
        Position importance values.
    properties : array-like of shape (m, p)
        Binary property matrix.
    parity : array-like of shape (p,)
        Maximum exposure fraction per group.

    Returns
    -------
    RankingResult
    """
    if isinstance(properties, pd.DataFrame):
        properties = properties.to_numpy()

    no_pos = len(pos_imp)
    no_items = len(item_qual)
    total_expo = sum(pos_imp)

    ranking = [None] * no_pos
    curr_sat = np.zeros(properties.shape[1])
    item_list = list(range(no_items))

    for pos in range(no_pos):
        placed = False
        for idx, item in enumerate(item_list):
            if (curr_sat + properties[item] * pos_imp[pos] <= parity * total_expo).all():
                ranking[pos] = item
                curr_sat += properties[item] * pos_imp[pos]
                item_list.pop(idx)
                placed = True
                break

        if not placed:
            return RankingResult(ranking=None, value=0.0, feasible=False)

    total_value = sum(
        pos_imp[i] * item_qual[ranking[i]] for i in range(no_pos)
    )
    return RankingResult(ranking=ranking, value=total_value, feasible=True)


# ---------------------------------------------------------------------------
# Unconstrained ranking
# ---------------------------------------------------------------------------

def unconstrained_ranking_matching(
    val_mat: list[list[float]],
) -> tuple[list[list[int]], float]:
    """Solve unconstrained ranking via max-weight bipartite matching.

    Parameters
    ----------
    val_mat : list of list of float
        Value matrix of shape (n, m) where ``val_mat[i][j] = pos_imp[i] * item_qual[j]``.

    Returns
    -------
    ranking : list of [position, item] pairs
    value : float
        Total ranking value.
    """
    n = len(val_mat)
    m = len(val_mat[0])
    G = nx.Graph()
    G.add_nodes_from(range(n + m))

    for pos in range(n):
        for item in range(n, n + m):
            G.add_edge(pos, item, weight=val_mat[pos][item - n])

    matching = list(nx.max_weight_matching(G))
    ranking = [sorted(pair) for pair in matching]

    value = sum(val_mat[pair[0]][pair[1] - n] for pair in ranking)

    return ranking, value


def easy_unconstrained_rank(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Trivial unconstrained ranking: assign item i to position i.

    Only correct when both ``item_qual`` and ``pos_imp`` are sorted descending.
    """
    n = len(pos_imp)
    ranking = np.column_stack([np.arange(n), np.arange(n)])
    value = sum(item_qual[i] * pos_imp[i] for i in range(n))
    return ranking, value


# ---------------------------------------------------------------------------
# Ranking format conversions
# ---------------------------------------------------------------------------

def rank_mat_from_lp(rank_list: list[float], m: int, n: int) -> np.ndarray:
    """Reshape a flat LP solution vector into an (m, n) assignment matrix."""
    return np.reshape(rank_list, (m, n))


def rank_mat_from_greedy(rank_list: list[int], m: int, n: int) -> np.ndarray:
    """Convert a greedy ranking (list of item indices) to an (m, n) matrix."""
    rank_mat = np.zeros((m, n), dtype=int)
    for pos, item in enumerate(rank_list):
        rank_mat[item][pos] = 1
    return rank_mat


def rank_mat_from_matching(
    rank_list: list[list[int]], m: int, n: int,
) -> np.ndarray:
    """Convert a matching result to an (m, n) assignment matrix."""
    rank_mat = np.zeros((m, n), dtype=int)
    for pair in rank_list:
        item = pair[0]
        pos = pair[1] - n
        rank_mat[item][pos] = 1
    return rank_mat


def rank_list_from_mat(rank_mat: np.ndarray) -> np.ndarray:
    """Extract a position → item mapping from an assignment matrix."""
    m, n = rank_mat.shape
    rank_list = np.zeros(n, dtype=int)
    for i in range(m):
        for j in range(n):
            if rank_mat[i][j] == 1:
                rank_list[j] = i
    return rank_list


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

def max_unfairness(
    rank_list: np.ndarray,
    pos_imp: np.ndarray,
    parity: np.ndarray,
    prop_list: list[list[int]],
    viol: float,
) -> float:
    """Compute the largest violation factor of fairness constraints.

    Returns the maximum ratio of actual group exposure to its allowed bound.
    """
    p = len(prop_list)
    group_exp = np.zeros(p)
    total_exp = sum(pos_imp)

    for i in range(len(rank_list)):
        for l in range(p):
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i] / total_exp

    # Small epsilon to avoid division by zero
    eps = np.full(p, 0.001)
    return float(np.max(group_exp / (parity / viol + eps)))


def demographic_parity_ratio(
    rank_list: np.ndarray,
    pos_imp: np.ndarray,
    prop_list: list[list[int]],
) -> float:
    """Compute the demographic parity ratio between two groups.

    Returns ``exposure_per_item[0] / exposure_per_item[1]``.
    A value of 1.0 indicates perfect parity.
    """
    p = len(prop_list)
    group_exp = np.zeros(p)

    for i in range(len(rank_list)):
        for l in range(p):
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i] / len(prop_list[l])

    if group_exp[1] == 0:
        return float("inf")
    return float(group_exp[0] / group_exp[1])


def disparate_treatment_ratio(
    rank_list: np.ndarray,
    pos_imp: np.ndarray,
    item_qual: np.ndarray,
    prop_list: list[list[int]],
) -> float:
    """Compute the disparate treatment ratio between two groups.

    Normalises each group's exposure by its total quality.
    A value of 1.0 indicates perfect treatment parity.
    """
    p = len(prop_list)
    group_exp = np.zeros(p)
    group_util = np.zeros(p)

    for l in range(p):
        for i in prop_list[l]:
            group_util[l] += item_qual[i]

    for i in range(len(rank_list)):
        for l in range(p):
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i] / group_util[l]

    if group_exp[1] == 0:
        return float("inf")
    return float(group_exp[0] / group_exp[1])
