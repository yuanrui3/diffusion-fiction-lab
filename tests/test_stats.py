import pytest

from dfl.stats import by_item, diff_independent, diff_of_diffs, diff_paired, mean_ci


def test_mean_is_pooled_over_defined_values():
    data = {"a": [1, 1, 0, 0], "b": [1, None, None, None], "c": [0, 0]}
    r = mean_ci(data, n_boot=500)
    assert r["mean"] == pytest.approx(3 / 7)
    assert r["n_items"] == 3
    assert r["lo"] <= r["mean"] <= r["hi"]


def test_all_undefined_gives_no_estimate():
    r = mean_ci({"a": [None, None], "b": []}, n_boot=100)
    assert r == {"mean": None, "lo": None, "hi": None, "n_items": 0}


def test_identical_items_have_a_zero_width_interval():
    r = mean_ci({f"i{k}": [1, 0, 1, 0] for k in range(20)}, n_boot=300)
    assert r["mean"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5


def test_interval_comes_from_items_not_samples():
    # Two kinds of item, each internally constant: all the variation is between items.
    # Treating the 800 samples as independent would give a far narrower interval.
    data = {f"x{k}": [1] * 100 for k in range(4)} | {f"y{k}": [0] * 100 for k in range(4)}
    r = mean_ci(data, n_boot=2000)
    assert r["mean"] == 0.5
    assert r["hi"] - r["lo"] > 0.4


def test_bootstrap_is_reproducible():
    data = {f"i{k}": [k % 2, 1, 0] for k in range(15)}
    assert mean_ci(data, n_boot=400) == mean_ci(data, n_boot=400)


def test_diff_independent():
    high = {f"h{k}": [1, 1, 1, 0] for k in range(30)}
    low = {f"l{k}": [0, 0, 0, 1] for k in range(40)}
    r = diff_independent(high, low, n_boot=500)
    assert r["diff"] == pytest.approx(0.5)
    assert r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)
    assert diff_independent(high, {}, n_boot=10)["diff"] is None


def test_paired_difference_cancels_shared_item_difficulty():
    # Model a beats model b by exactly one sample in four on every item, easy or hard.
    easy = {f"e{k}": ([1, 1, 1, 1], [1, 1, 1, 0]) for k in range(10)}
    hard = {f"d{k}": ([1, 0, 0, 0], [0, 0, 0, 0]) for k in range(10)}
    a = {k: v[0] for k, v in (easy | hard).items()}
    b = {k: v[1] for k, v in (easy | hard).items()}
    r = diff_paired(a, b, n_boot=1000)
    assert r["diff"] == pytest.approx(0.25)
    assert r["lo"] == pytest.approx(0.25) and r["hi"] == pytest.approx(0.25)
    assert r["n_items"] == 20


def test_paired_difference_uses_only_shared_items():
    r = diff_paired({"a": [1], "b": [1]}, {"b": [0], "c": [0]}, n_boot=50)
    assert r["n_items"] == 1 and r["diff"] == 1.0
    assert diff_paired({"a": [1]}, {"c": [0]}, n_boot=10)["diff"] is None


def test_diff_of_diffs():
    a1 = {f"u{k}": [1, 1] for k in range(8)}
    b1 = {f"u{k}": [1, 0] for k in range(8)}     # gap 0.5 on the first set
    a2 = {f"r{k}": [1, 0] for k in range(12)}
    b2 = {f"r{k}": [1, 0] for k in range(12)}    # gap 0 on the second set
    r = diff_of_diffs(a1, b1, a2, b2, n_boot=300)
    assert r["diff"] == pytest.approx(0.5)
    assert r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)


def test_by_item_groups_rows_and_keeps_none():
    rows = [
        {"item_id": "a", "rule": True, "pos": 0.5},
        {"item_id": "a", "rule": False, "pos": None},
        {"item_id": "b", "rule": True, "pos": 1.0},
    ]
    assert by_item(rows, "rule") == {"a": [1.0, 0.0], "b": [1.0]}
    assert by_item(rows, "pos") == {"a": [0.5, None], "b": [1.0]}
