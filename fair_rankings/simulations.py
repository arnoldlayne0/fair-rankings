"""Simulation harness for studying empirical behaviour of ranking algorithms.

Provides functions to repeatedly solve ranking problems on random instances
and collect summary statistics on runtime, approximation quality, ranking
similarity, and the price of fairness.
"""

from __future__ import annotations

import math
import timeit
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from . import data as dg
from . import algorithms as ra


def sim_pcrm_time(
    times: int,
    notion: Literal["demographic", "utilitarian"],
) -> pd.DataFrame:
    """Benchmark IP solver runtime on random PCRM instances.

    Returns a DataFrame with columns:
    ``time``, ``value``, ``bound``, ``no_items``, ``no_pos``, ``no_prop``.
    """
    records = []
    for _ in range(times):
        p = np.random.randint(2, 20)
        n = np.random.randint(3, 100)
        m = math.ceil(np.random.uniform(1, 1.5) * n)
        viol = 1.5

        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list(properties)
        parity = dg.parity_pcrm(prop_list, item_qual, notion, viol)

        start = timeit.default_timer()
        result = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        elapsed = timeit.default_timer() - start

        records.append({
            "time": elapsed,
            "value": result.obj_value,
            "bound": viol,
            "no_items": m,
            "no_pos": n,
            "no_prop": p,
        })

    return pd.DataFrame(records)


def sim_topk_time(
    times: int,
    notion: Literal["demographic", "utilitarian"],
) -> pd.DataFrame:
    """Benchmark greedy top-k solver runtime on random instances.

    Returns a DataFrame with columns:
    ``time``, ``value``, ``bound``, ``no_items``, ``no_pos``, ``no_prop``.
    """
    records = []
    for _ in range(times):
        p = np.random.randint(2, 20)
        n = np.random.randint(3, 100)
        m = math.ceil(np.random.uniform(1, 1.5) * n)
        viol = 1.5

        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list(properties)
        parity = dg.parity_topk(prop_list, item_qual, n, notion)

        start = timeit.default_timer()
        result = ra.greedy_topk(properties, parity, pos_imp, item_qual)
        elapsed = timeit.default_timer() - start

        records.append({
            "time": elapsed,
            "value": result.value,
            "bound": viol,
            "no_items": m,
            "no_pos": n,
            "no_prop": p,
        })

    return pd.DataFrame(records)


def pcrm_vs_topk(
    times: int,
    notion: Literal["demographic", "utilitarian"],
) -> pd.DataFrame:
    """Compare PCRM (IP) and top-k (greedy) rankings on the same instances.

    Returns a DataFrame with Spearman correlations and value ratios.
    """
    records = []
    for i in range(times):
        p = np.random.randint(2, 5)
        n = np.random.randint(10, 20)
        m = math.ceil(np.random.uniform(1, 1.5) * n)
        viol = np.random.uniform(1.1, 1.2)

        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list(properties)
        parity_tk = dg.parity_topk(prop_list, item_qual, n, notion)
        parity_pc = dg.parity_pcrm(prop_list, item_qual, notion, viol)

        result_topk = ra.greedy_topk(properties, parity_tk, pos_imp, item_qual)
        result_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pc)

        rank_list_pcrm = ra.rank_list_from_mat(
            ra.rank_mat_from_lp(result_pcrm.solution, m, n)
        )

        if result_topk.feasible and result_topk.value != 0:
            value_factor = result_pcrm.obj_value / result_topk.value
        elif result_pcrm.obj_value == 0:
            value_factor = 1.0
        else:
            value_factor = 0.0

        rho, p_val = stats.spearmanr(result_topk.ranking, rank_list_pcrm)

        records.append({
            "value_factor": value_factor,
            "rho": rho,
            "p_val": p_val,
            "no_items": m,
            "no_pos": n,
            "no_prop": p,
        })

    df = pd.DataFrame(records)
    df["p_bonferroni"] = df["p_val"] * times
    return df


def greedy_approx_quality(
    times: int,
    notion: Literal["demographic", "utilitarian"],
    disjoint: bool,
) -> pd.DataFrame:
    """Compare greedy PCRM approximation quality against exact IP.

    Returns a DataFrame with approximation factors and feasibility flags.
    """
    records = []
    for i in range(times):
        p = np.random.randint(2, 5)
        n = np.random.randint(10, 20)
        m = math.ceil(np.random.uniform(1, 1.5) * n)
        viol = np.random.uniform(1.1, 1.2)

        pos_imp, item_qual, properties = dg.sim_data(m, n, p, disjoint)
        prop_list = dg.get_prop_list(properties)
        parity = dg.parity_pcrm(prop_list, item_qual, notion, viol)

        start = timeit.default_timer()
        result_ip = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        time_ip = timeit.default_timer() - start

        start = timeit.default_timer()
        result_greedy = ra.greedy_parity(item_qual, pos_imp, properties, parity)
        time_greedy = timeit.default_timer() - start

        feas_ip = result_ip.obj_value > 0
        feas_greedy = result_greedy.feasible and result_greedy.value > 0

        if feas_ip and feas_greedy:
            val_fact = result_ip.obj_value / result_greedy.value
        else:
            val_fact = np.nan

        records.append({
            "value_factor": val_fact,
            "feas_ip": int(feas_ip),
            "feas_greedy": int(feas_greedy),
            "time_ip": time_ip,
            "time_greedy": time_greedy,
            "viol_coeff": viol,
        })

    return pd.DataFrame(records)


def price_of_fairness(
    times: int,
    notion: Literal["demographic", "utilitarian"],
) -> pd.DataFrame:
    """Measure the utility loss from imposing fairness constraints.

    Returns a DataFrame with price of fairness and unfairness of the
    unconstrained ranking.
    """
    records = []
    for i in range(times):
        p = np.random.randint(2, 5)
        n = np.random.randint(10, 20)
        m = math.ceil(np.random.uniform(1, 1.5) * n)
        viol = np.random.uniform(1.15, 1.2)

        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list(properties)
        parity = dg.parity_pcrm(prop_list, item_qual, notion, viol)

        result_fair = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        val_mat = dg.get_val_mat(pos_imp, item_qual)
        result_unc = ra.unconstrained_ranking_matching(val_mat)

        unc_rank_mat = ra.rank_mat_from_matching(result_unc[0], m, n)
        unc_rank_list = ra.rank_list_from_mat(unc_rank_mat)

        unfair = ra.max_unfairness(
            unc_rank_list, pos_imp, parity / viol, prop_list, viol
        )
        opt = result_unc[1]
        fair = result_fair.obj_value
        price = (opt - fair) / opt if opt > 0 else 0.0

        records.append({
            "price_of_fairness": price,
            "viol_coeff": viol,
            "max_unfairness": unfair,
            "no_items": m,
            "no_pos": n,
            "no_prop": p,
        })

    return pd.DataFrame(records)


def twogroup_sim(
    one: tuple,
    two: tuple,
    n: int,
    notion: Literal["demographic", "utilitarian"],
    viol: float,
) -> dict:
    """Run a single two-group fairness experiment.

    Parameters
    ----------
    one, two : tuple of (mean, sd, size, property_probs)
        Group parameters.
    n : int
        Number of positions.
    notion : str
        Fairness notion.
    viol : float
        Violation coefficient.

    Returns
    -------
    dict with keys: price, ratio_unfair, ratio_fair, max_unfair_unfair,
    max_unfair_fair, approx_quality.
    """
    item_qual, prop_list, properties = dg.sim_twogroup_data(one, two)
    m = len(item_qual)
    exposure = dg.dcg_exposure(n)

    # Unconstrained
    val_mat = dg.get_val_mat(item_qual, exposure)
    unc_result = ra.unconstrained_ranking_matching(val_mat)
    unc_rank_list = np.arange(n)
    unc_val = unc_result[1]

    # Fair (exact IP)
    parity = dg.parity_pcrm(prop_list, item_qual, notion, viol)
    fair_result = ra.ip_parity(item_qual, exposure, prop_list, parity)
    fair_rank_mat = ra.rank_mat_from_lp(fair_result.solution, m, n)
    fair_rank_list = ra.rank_list_from_mat(fair_rank_mat)
    fair_val = fair_result.obj_value

    # Fair (greedy approximation)
    fair_approx = ra.greedy_parity(item_qual, exposure, properties, parity)
    approx_quality = (
        fair_val / fair_approx.value
        if fair_approx.feasible and fair_approx.value > 0
        else np.nan
    )

    # Fairness ratios
    if notion == "demographic":
        r_unfair = ra.demographic_parity_ratio(unc_rank_list, exposure, prop_list)
        r_fair = ra.demographic_parity_ratio(fair_rank_list, exposure, prop_list)
    else:
        r_unfair = ra.disparate_treatment_ratio(unc_rank_list, exposure, item_qual, prop_list)
        r_fair = ra.disparate_treatment_ratio(fair_rank_list, exposure, item_qual, prop_list)

    mu_unfair = ra.max_unfairness(unc_rank_list, exposure, parity, prop_list, viol)
    mu_fair = ra.max_unfairness(fair_rank_list, exposure, parity, prop_list, viol)

    price = (unc_val - fair_val) / unc_val if unc_val > 0 else 0.0

    return {
        "price": price,
        "ratio_unfair": r_unfair,
        "ratio_fair": r_fair,
        "max_unfair_unfair": mu_unfair,
        "max_unfair_fair": mu_fair,
        "approx_quality": approx_quality,
    }


def twogroup_sim_mult_uniform(
    m: int,
    n: int,
    times: int,
    notion: Literal["demographic", "utilitarian"],
    viol: float,
) -> pd.DataFrame:
    """Run multiple two-group experiments with uniform parameter sampling.

    Group 1: high quality (mean ∈ [0.6, 0.8]), property probs [0.8, 0.2].
    Group 2: low quality (mean ∈ [0.2, 0.4]), property probs [0.2, 0.8].
    """
    records = []
    for i in range(times):
        size_one = np.random.randint(0, m)
        one = [
            np.random.uniform(0.6, 0.8),
            np.random.uniform(0.1, 0.3),
            size_one,
            [0.8, 0.2],
        ]
        two = [
            np.random.uniform(0.2, 0.4),
            np.random.uniform(0.1, 0.3),
            m - size_one,
            [0.2, 0.8],
        ]
        result = twogroup_sim(one, two, n, notion, viol)
        records.append(result)

    return pd.DataFrame(records)
