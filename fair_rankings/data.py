"""Data generation and manipulation for fair ranking experiments.

Provides functions for:
- Generating simulated ranking instances (items, positions, properties)
- Loading and processing the YOW news recommendation dataset
- Computing fairness bounds (demographic parity, disparate treatment)
- Constructing LP/IP constraint matrices
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import math
import numpy as np
import pandas as pd
import scipy.stats


# ---------------------------------------------------------------------------
# Simulated data
# ---------------------------------------------------------------------------

def gen_distinct_properties(m: int, p: int) -> np.ndarray:
    """Generate ``p`` mutually exclusive random properties for ``m`` items.

    Each item is assigned exactly one property uniformly at random,
    so the resulting property columns are disjoint.
    """
    properties = np.zeros((m, p), dtype=int)
    for item in range(m):
        prop = np.random.randint(low=0, high=p)
        properties[item][prop] = 1
    return properties


def gen_properties(m: int, p: int) -> np.ndarray:
    """Generate ``p`` independent random properties for ``m`` items.

    Each (item, property) entry is drawn from Bernoulli(0.5),
    so properties are *not* necessarily disjoint.
    """
    properties = np.zeros((m, p), dtype=int)
    for item in range(m):
        properties[item] = np.random.binomial(1, 0.5, p)
    return properties


def sim_data(
    m: int,
    n: int,
    p: int,
    distinct: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random ranking instance.

    Parameters
    ----------
    m : int
        Number of items.
    n : int
        Number of positions.
    p : int
        Number of properties.
    distinct : bool
        If True, properties are mutually exclusive (disjoint groups).

    Returns
    -------
    pos_imp : ndarray of shape (n,)
        Position importance values, sorted descending.
    item_qual : ndarray of shape (m,)
        Item quality values, sorted descending.
    properties : ndarray of shape (m, p)
        Binary property matrix.
    """
    pos_imp = np.sort(np.random.uniform(0.0, 1.0, size=n))[::-1]
    item_qual = np.sort(np.random.uniform(0.0, 1.0, size=m))[::-1]
    if distinct:
        properties = gen_distinct_properties(m, p)
    else:
        properties = gen_properties(m, p)
    return pos_imp, item_qual, properties


# ---------------------------------------------------------------------------
# Property helpers
# ---------------------------------------------------------------------------

def get_prop_list(properties: np.ndarray | pd.DataFrame) -> list[list[int]]:
    """Return a list of item indices for each property column.

    Parameters
    ----------
    properties : array-like of shape (m, p)
        Binary property matrix.

    Returns
    -------
    prop_list : list of list of int
        ``prop_list[l]`` contains the indices of items possessing property ``l``.
    """
    if isinstance(properties, pd.DataFrame):
        properties = properties.to_numpy()
    prop_list = []
    for col in range(properties.shape[1]):
        prop_list.append(
            [i for i, val in enumerate(properties[:, col]) if val == 1]
        )
    return prop_list


# ---------------------------------------------------------------------------
# Value matrix
# ---------------------------------------------------------------------------

def get_val_mat(
    pos_imp: np.ndarray,
    item_qual: np.ndarray,
) -> list[list[float]]:
    """Compute the value matrix ``w[i][j] = pos_imp[i] * item_qual[j]``."""
    n = len(pos_imp)
    m = len(item_qual)
    return [
        [pos_imp[i] * item_qual[j] for j in range(m)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fairness bounds
# ---------------------------------------------------------------------------

def topk_upper_bounds(
    n: int,
    p: int,
    prob: float,
) -> list[list[int]]:
    """Generate random upper bounds for top-k ranking maximisation."""
    up_bounds = [[0] * n for _ in range(p)]
    for pr in range(p):
        for pos in range(n):
            up_bounds[pr][pos] = (
                up_bounds[pr][pos - 1]
                + int(np.random.binomial(1, prob, 1))
            )
    return up_bounds


def parity_pcrm(
    prop_list: list[list[int]],
    item_qual: np.ndarray,
    notion: Literal["demographic", "utilitarian"],
    viol_coeff: float,
) -> np.ndarray:
    """Compute parity bounds for the PCRM model.

    Parameters
    ----------
    prop_list : list of list of int
        Groups of item indices.
    item_qual : array-like
        Item quality scores.
    notion : {"demographic", "utilitarian"}
        Fairness notion.
    viol_coeff : float
        Violation coefficient (≥ 1) controlling how much the bound
        is relaxed relative to exact parity.

    Returns
    -------
    parity : ndarray
        Maximum fraction of total exposure each group may receive.
    """
    m = len(item_qual)
    p = len(prop_list)

    if notion == "demographic":
        parity = np.array([len(prop_list[i]) / m for i in range(p)])
    elif notion == "utilitarian":
        avg_qual = sum(item_qual) / m
        group_qual = np.zeros(p)
        for i in range(p):
            for j in prop_list[i]:
                group_qual[i] += item_qual[j]
            group_qual[i] /= len(prop_list[i])
        parity = group_qual / (avg_qual * p)
    else:
        raise ValueError(f"Unknown fairness notion: {notion!r}")

    return parity * viol_coeff


def parity_topk(
    prop_list: list[list[int]],
    item_qual: np.ndarray,
    n: int,
    notion: Literal["demographic", "utilitarian"],
) -> np.ndarray:
    """Compute parity bounds for the Top-k ranking model.

    Returns an integer matrix of shape ``(p, n)`` where entry ``[l, k]``
    is the maximum number of items from group ``l`` in the top ``k+1``
    positions.
    """
    m = len(item_qual)
    p = len(prop_list)

    if notion == "demographic":
        parity = np.zeros((p, n), dtype=int)
        for l in range(p):
            for k in range(n):
                parity[l][k] = math.ceil(len(prop_list[l]) / m * (k + 1))
    elif notion == "utilitarian":
        avg_qual = sum(item_qual) / m
        group_qual = np.zeros(p)
        for i in range(p):
            for j in prop_list[i]:
                group_qual[i] += item_qual[j]
            group_qual[i] /= len(prop_list[i])
        parity = np.zeros((p, n), dtype=int)
        for l in range(p):
            for k in range(n):
                parity[l][k] = math.ceil(
                    group_qual[l] / (avg_qual * p) * (k + 1)
                )
    else:
        raise ValueError(f"Unknown fairness notion: {notion!r}")

    return parity


# ---------------------------------------------------------------------------
# LP / IP data transformation
# ---------------------------------------------------------------------------

def gen_data_lp(
    item_qual: np.ndarray,
    pos_imp: np.ndarray,
    prop_list: list[list[int]],
    parity: np.ndarray,
) -> tuple[list, list, list]:
    """Transform ranking data into LP/IP standard form.

    Returns ``(val_list, bounds, A)`` where:
    - ``val_list`` is the objective coefficient vector (with labels),
    - ``bounds`` is the constraint RHS vector (with labels),
    - ``A`` is the constraint matrix.
    """
    m, n, p = len(item_qual), len(pos_imp), len(prop_list)
    total_imp = sum(pos_imp)

    # Objective: c vector with labels
    val_list = []
    for i in range(m):
        for j in range(n):
            val = item_qual[i] * pos_imp[j]
            label = f"item{i},position{j}"
            val_list.append([label, val])

    # Constraint bounds: b vector
    bounds = []
    for i in range(m):
        bounds.append([f"item {i}", 1])
    for j in range(n):
        bounds.append([f"position {j}", 1])
    for l in range(p):
        bounds.append([f"parity {l}", parity[l] * total_imp])

    # Constraint matrix A
    A = [[0] * (n * m) for _ in range(m + n + p)]
    for i in range(m):
        start = i * n
        A[i][start : start + n] = [1] * n
    for j in range(n):
        start = j
        A[m + j][start::n] = [1] * m
    for l in range(p):
        items_w_prop = prop_list[l]
        for k in items_w_prop:
            start = k * n
            stop = k * n + n
            A[m + n + l][start:stop] = list(pos_imp)

    return val_list, bounds, A


# ---------------------------------------------------------------------------
# Two-group simulation data
# ---------------------------------------------------------------------------

def sim_twogroup_data(
    one: tuple[float, float, int, list[float]],
    two: tuple[float, float, int, list[float]],
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Generate a two-group ranking instance with different quality distributions.

    Parameters
    ----------
    one, two : tuple of (mean, sd, size, property_probs)
        Parameters for each group. Quality is drawn from a truncated
        normal on [0, 1]. ``property_probs`` is a list of Bernoulli
        probabilities for each property.

    Returns
    -------
    item_qual : ndarray
        Sorted item qualities (descending).
    prop_list : list of list of int
        Groups of item indices.
    properties : ndarray
        Binary property matrix.
    """
    m = one[2] + two[2]
    p = len(one[3])
    sigma = 0.1

    item_qual_1 = scipy.stats.truncnorm.rvs(
        (0 - one[0]) / sigma,
        (1 - one[0]) / sigma,
        loc=one[0],
        scale=one[1],
        size=one[2],
    )
    item_qual_2 = scipy.stats.truncnorm.rvs(
        (0 - two[0]) / sigma,
        (1 - two[0]) / sigma,
        loc=two[0],
        scale=two[1],
        size=two[2],
    )

    properties = np.zeros((m, p), dtype=int)
    for item in range(one[2]):
        for l in range(p):
            properties[item][l] = np.random.binomial(1, one[3][l], 1)
    for item in range(one[2], m):
        for l in range(p):
            properties[item][l] = np.random.binomial(1, two[3][l], 1)

    item_qual = np.concatenate([item_qual_1, item_qual_2])

    # Sort everything by descending quality
    order = np.argsort(-item_qual)
    item_qual = item_qual[order]
    properties = properties[order]
    prop_list = get_prop_list(properties)

    return item_qual, prop_list, properties


# ---------------------------------------------------------------------------
# YOW dataset loading
# ---------------------------------------------------------------------------

def load_yow_data(path: str | Path) -> pd.DataFrame:
    """Load the raw YOW user study dataset.

    Parameters
    ----------
    path : str or Path
        Path to ``yow_userstudy_raw.xls``.
    """
    return pd.read_excel(path)


def _expand_classes(dat: pd.DataFrame) -> pd.DataFrame:
    """Split the pipe-delimited ``classes`` column into separate columns."""
    dat_exp = dat.classes.str.split("|", expand=True)
    dat_exp = dat_exp.drop(0, axis=1)
    dat_exp.columns = [f"class_{i}" for i in range(1, dat_exp.shape[1] + 1)]
    return pd.concat([dat, dat_exp], axis=1)


def topic_data(
    path: str | Path,
    topic: str,
    threshold: int,
) -> pd.DataFrame:
    """Extract a topic-specific subset from the YOW dataset.

    Parameters
    ----------
    path : str or Path
        Path to the raw Excel file.
    topic : str
        Topic string to filter on.
    threshold : int
        Minimum number of articles per RSS source to include.

    Returns
    -------
    DataFrame with columns: ``relevant`` (quality) + one-hot RSS source columns.
    """
    dat = pd.read_excel(path)[["RSS_ID", "classes", "relevant"]]
    dat = _expand_classes(dat)

    class_cols = [c for c in dat.columns if c.startswith("class_")]
    mask = dat[class_cols].eq(topic).any(axis=1)
    dat = dat.loc[mask, ["RSS_ID", "relevant"]]
    dat = dat.groupby("RSS_ID").filter(lambda x: len(x) >= threshold)
    dat.index = range(len(dat))

    # Add small noise to break ties
    dat["relevant"] = dat["relevant"] + np.random.normal(0, 0.05, len(dat))
    dat = pd.concat([dat["relevant"], pd.get_dummies(dat["RSS_ID"])], axis=1)
    dat = dat.sort_values("relevant", ascending=False).reset_index(drop=True)

    return dat


def user_data(path: str | Path, user_id: int) -> pd.DataFrame:
    """Extract a user-specific subset from the YOW dataset.

    Parameters
    ----------
    path : str or Path
        Path to the raw Excel file.
    user_id : int
        User identifier.

    Returns
    -------
    DataFrame with columns: ``user_like`` (quality) + one-hot topic columns.
    """
    dat = pd.read_excel(path)[["user_id", "classes", "user_like"]]
    dat = _expand_classes(dat)

    class_cols = [c for c in dat.columns if c.startswith("class_")]
    dat = dat[dat.user_id == user_id]
    dat.fillna(value=np.nan, inplace=True)

    dat_stack = dat[class_cols].stack()
    all_classes = list(dat_stack.value_counts().index)
    dat.index = range(len(dat))

    dat_mat = dat[class_cols].to_numpy()
    props = np.zeros((len(dat), len(all_classes)), dtype=int)
    for i in range(len(props)):
        for j, cls in enumerate(all_classes):
            if (dat_mat[i, :] == cls).any():
                props[i, j] = 1

    props = pd.DataFrame(props, columns=all_classes)
    dat["user_like"] = dat["user_like"] + np.random.normal(0, 0.05, len(dat))
    dat = pd.concat([dat["user_like"], props], axis=1)
    dat = dat.sort_values("user_like", ascending=False).reset_index(drop=True)

    return dat


def dcg_exposure(n: int) -> np.ndarray:
    """Return DCG-style position exposure: ``1 / log2(j + 2)``."""
    return np.array([1.0 / math.log(j + 2) for j in range(n)])
