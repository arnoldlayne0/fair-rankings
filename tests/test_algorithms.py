"""Tests for fair ranking algorithms.

Uses small instances with known solutions to verify correctness.
"""

import math
import numpy as np
import pytest

from fair_rankings import data as dg
from fair_rankings import algorithms as ra


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_instance():
    """A minimal 4-item, 3-position, 2-group instance."""
    item_qual = np.array([1.0, 0.8, 0.6, 0.4])
    pos_imp = np.array([1.0, 0.5, 0.25])
    properties = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    prop_list = dg.get_prop_list(properties)
    return item_qual, pos_imp, properties, prop_list


# ---------------------------------------------------------------------------
# Data module tests
# ---------------------------------------------------------------------------

class TestDataGeneration:

    def test_gen_distinct_properties_are_disjoint(self):
        props = dg.gen_distinct_properties(100, 5)
        assert props.shape == (100, 5)
        # Each item has exactly one property
        assert (props.sum(axis=1) == 1).all()

    def test_gen_properties_shape(self):
        props = dg.gen_properties(50, 3)
        assert props.shape == (50, 3)
        assert set(np.unique(props)).issubset({0, 1})

    def test_sim_data_sorted_descending(self):
        pos_imp, item_qual, props = dg.sim_data(10, 5, 3, True)
        assert len(pos_imp) == 5
        assert len(item_qual) == 10
        assert all(pos_imp[i] >= pos_imp[i + 1] for i in range(4))
        assert all(item_qual[i] >= item_qual[i + 1] for i in range(9))

    def test_get_prop_list_roundtrip(self):
        props = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        prop_list = dg.get_prop_list(props)
        assert prop_list[0] == [0, 2]
        assert prop_list[1] == [1, 2]

    def test_get_val_mat_values(self):
        pos_imp = np.array([1.0, 0.5])
        item_qual = np.array([0.8, 0.4, 0.2])
        val_mat = dg.get_val_mat(pos_imp, item_qual)
        assert len(val_mat) == 2
        assert len(val_mat[0]) == 3
        assert val_mat[0][0] == pytest.approx(0.8)
        assert val_mat[1][2] == pytest.approx(0.1)

    def test_parity_pcrm_demographic_sums_to_viol(self):
        prop_list = [[0, 1], [2, 3]]
        item_qual = np.array([1.0, 0.8, 0.6, 0.4])
        parity = dg.parity_pcrm(prop_list, item_qual, "demographic", 1.0)
        assert parity.sum() == pytest.approx(1.0)

    def test_parity_pcrm_invalid_notion_raises(self):
        with pytest.raises(ValueError, match="Unknown fairness notion"):
            dg.parity_pcrm([[0]], np.array([1.0]), "invalid", 1.0)

    def test_dcg_exposure_decreasing(self):
        exp = dg.dcg_exposure(10)
        assert len(exp) == 10
        assert all(exp[i] >= exp[i + 1] for i in range(9))


# ---------------------------------------------------------------------------
# Algorithm tests
# ---------------------------------------------------------------------------

class TestGreedyTopk:

    def test_unconstrained_picks_best_items(self):
        """With no binding constraints, greedy should pick items 0,1,2."""
        item_qual = np.array([1.0, 0.8, 0.6, 0.4])
        pos_imp = np.array([1.0, 0.5, 0.25])
        properties = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        # Very loose bounds
        up_bounds = np.array([[3, 3, 3], [3, 3, 3]])

        result = ra.greedy_topk(properties, up_bounds, pos_imp, item_qual)
        assert result.feasible
        assert result.ranking == [0, 1, 2]

    def test_constraints_force_reranking(self):
        """With tight bounds, group 0 can have at most 1 item in top 2."""
        item_qual = np.array([1.0, 0.9, 0.5, 0.4])
        pos_imp = np.array([1.0, 0.5])
        properties = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        # Group 0: at most 1 in top-1, at most 1 in top-2
        up_bounds = np.array([[1, 1], [2, 2]])

        result = ra.greedy_topk(properties, up_bounds, pos_imp, item_qual)
        assert result.feasible
        # Item 0 goes to pos 0, but item 1 can't go to pos 1
        # so item 2 (group 1) takes pos 1
        assert result.ranking[0] == 0
        assert result.ranking[1] == 2

    def test_infeasible_instance(self):
        """Impossible constraints → infeasible."""
        item_qual = np.array([1.0, 0.8])
        pos_imp = np.array([1.0, 0.5])
        properties = np.array([[1, 0], [1, 0]])
        # Group 0 can have 0 items at any position
        up_bounds = np.array([[0, 0], [0, 0]])

        result = ra.greedy_topk(properties, up_bounds, pos_imp, item_qual)
        assert not result.feasible
        assert result.ranking is None


class TestParityConstrained:

    def test_ip_returns_result(self, simple_instance):
        item_qual, pos_imp, properties, prop_list = simple_instance
        parity = dg.parity_pcrm(prop_list, item_qual, "demographic", 1.5)
        result = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        assert result.success
        assert result.obj_value > 0

    def test_lp_relaxation_geq_ip(self, simple_instance):
        """LP relaxation should have objective ≥ IP (it's a relaxation)."""
        item_qual, pos_imp, properties, prop_list = simple_instance
        parity = dg.parity_pcrm(prop_list, item_qual, "demographic", 1.2)
        ip_result = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        lp_result = ra.lp_parity(item_qual, pos_imp, prop_list, parity)
        assert lp_result.obj_value >= ip_result.obj_value - 1e-10

    def test_greedy_parity_feasible(self, simple_instance):
        item_qual, pos_imp, properties, prop_list = simple_instance
        parity = dg.parity_pcrm(prop_list, item_qual, "demographic", 1.5)
        result = ra.greedy_parity(item_qual, pos_imp, properties, parity)
        assert result.feasible
        assert result.value > 0

    def test_greedy_within_bound_of_ip(self, simple_instance):
        """Greedy should not be too far from IP optimum."""
        item_qual, pos_imp, properties, prop_list = simple_instance
        parity = dg.parity_pcrm(prop_list, item_qual, "demographic", 1.5)
        ip_result = ra.ip_parity(item_qual, pos_imp, prop_list, parity)
        greedy_result = ra.greedy_parity(item_qual, pos_imp, properties, parity)
        if greedy_result.feasible and ip_result.obj_value > 0:
            ratio = ip_result.obj_value / greedy_result.value
            # Greedy should be within 2x of optimal (very loose bound)
            assert ratio < 2.0


class TestUnconstrained:

    def test_matching_optimal_for_sorted(self):
        """For sorted inputs, matching should assign item i to position i."""
        pos_imp = np.array([1.0, 0.5, 0.25])
        item_qual = np.array([0.9, 0.6, 0.3])
        val_mat = dg.get_val_mat(pos_imp, item_qual)

        _, value = ra.unconstrained_ranking_matching(val_mat)
        expected = sum(pos_imp[i] * item_qual[i] for i in range(3))
        assert value == pytest.approx(expected, abs=1e-6)

    def test_easy_unconstrained_correct(self):
        pos_imp = np.array([1.0, 0.5])
        item_qual = np.array([0.8, 0.4])
        ranking, value = ra.easy_unconstrained_rank(item_qual, pos_imp)
        assert value == pytest.approx(0.8 * 1.0 + 0.4 * 0.5)


class TestFormatConversions:

    def test_greedy_mat_roundtrip(self):
        rank_list = [2, 0, 3]
        mat = ra.rank_mat_from_greedy(rank_list, 4, 3)
        recovered = ra.rank_list_from_mat(mat)
        np.testing.assert_array_equal(recovered, rank_list)


class TestFairnessMetrics:

    def test_dp_ratio_equal_groups(self):
        """Equal-size groups with equal exposure → ratio ≈ 1."""
        rank_list = np.array([0, 1])  # item 0 at pos 0, item 1 at pos 1
        pos_imp = np.array([1.0, 1.0])
        prop_list = [[0], [1]]
        ratio = ra.demographic_parity_ratio(rank_list, pos_imp, prop_list)
        assert ratio == pytest.approx(1.0)

    def test_dt_ratio_equal_quality(self):
        """Equal quality items → disparate treatment ratio ≈ 1 with equal exposure."""
        rank_list = np.array([0, 1])
        pos_imp = np.array([1.0, 1.0])
        item_qual = np.array([0.5, 0.5])
        prop_list = [[0], [1]]
        ratio = ra.disparate_treatment_ratio(rank_list, pos_imp, item_qual, prop_list)
        assert ratio == pytest.approx(1.0)
