import numpy as np
import pytest

from core.drift.statistical_tests import DriftStatistics


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture(scope="module")
def stable_data():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 2000)
    cur = np.random.normal(0, 1, 2000)
    return ref, cur


@pytest.fixture(scope="module")
def drifted_data():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 2000)
    cur = np.random.normal(3, 1.8, 2000)
    return ref, cur


@pytest.fixture(scope="module")
def nan_data():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)

    ref[0:10] = np.nan
    cur[5:15] = np.nan
    return ref, cur


# ==========================================================
# STRUCTURE VALIDATION
# ==========================================================

def assert_metric_structure(result):
    assert isinstance(result, dict)
    assert "metric" in result
    assert "score" in result
    assert "severity" in result


# ==========================================================
# PSI
# ==========================================================

def test_psi_stable(stable_data):
    ref, cur = stable_data
    result = DriftStatistics.psi(ref, cur)

    assert_metric_structure(result)
    assert result["score"] < 0.1
    assert result["severity"] == "NO_DRIFT"


def test_psi_drifted(drifted_data):
    ref, cur = drifted_data
    result = DriftStatistics.psi(ref, cur)

    assert_metric_structure(result)
    assert result["score"] > 0.25
    assert result["severity"] == "SIGNIFICANT_DRIFT"


# ==========================================================
# KS
# ==========================================================

def test_ks_stable(stable_data):
    ref, cur = stable_data
    result = DriftStatistics.ks_statistic(ref, cur)

    assert_metric_structure(result)
    assert "p_value" in result
    assert result["score"] < 0.1
    assert result["severity"] == "NO_DRIFT"


def test_ks_drifted(drifted_data):
    ref, cur = drifted_data
    result = DriftStatistics.ks_statistic(ref, cur)

    assert_metric_structure(result)
    assert result["score"] > 0.2
    assert result["severity"] == "DRIFT"


# ==========================================================
# KL
# ==========================================================

def test_kl_stable(stable_data):
    ref, cur = stable_data
    result = DriftStatistics.kl_divergence(ref, cur)

    assert_metric_structure(result)
    assert result["score"] < 0.1
    assert result["severity"] == "NO_DRIFT"


def test_kl_drifted(drifted_data):
    ref, cur = drifted_data
    result = DriftStatistics.kl_divergence(ref, cur)

    assert_metric_structure(result)
    assert result["score"] > 0.1
    assert result["severity"] == "DRIFT"


# ==========================================================
# JS
# ==========================================================

def test_js_stable(stable_data):
    ref, cur = stable_data
    result = DriftStatistics.js_divergence(ref, cur)

    assert_metric_structure(result)
    assert result["score"] < 0.1
    assert result["severity"] == "NO_DRIFT"


def test_js_drifted(drifted_data):
    ref, cur = drifted_data
    result = DriftStatistics.js_divergence(ref, cur)

    assert_metric_structure(result)
    assert result["score"] > 0.1
    assert result["severity"] == "DRIFT"


# ==========================================================
# NaN Handling
# ==========================================================

def test_metrics_handle_nan(nan_data):
    ref, cur = nan_data

    psi = DriftStatistics.psi(ref, cur)
    ks = DriftStatistics.ks_statistic(ref, cur)
    kl = DriftStatistics.kl_divergence(ref, cur)
    js = DriftStatistics.js_divergence(ref, cur)

    assert psi["score"] >= 0
    assert ks["score"] >= 0
    assert kl["score"] >= 0
    assert js["score"] >= 0


# ==========================================================
# Input Validation
# ==========================================================

def test_empty_input_raises():
    with pytest.raises(ValueError):
        DriftStatistics.psi([], [])


def test_none_input_raises():
    with pytest.raises(ValueError):
        DriftStatistics.ks_statistic(None, None)


# ==========================================================
# Full Report Integration
# ==========================================================

def test_full_report_structure(stable_data):
    ref, cur = stable_data
    report = DriftStatistics.full_report(ref, cur)

    assert isinstance(report, dict)
    assert "psi" in report
    assert "ks" in report
    assert "kl" in report
    assert "js" in report

    assert report["psi"]["metric"] == "PSI"
    assert report["ks"]["metric"] == "KS"
    assert report["kl"]["metric"] == "KL"
    assert report["js"]["metric"] == "JS"