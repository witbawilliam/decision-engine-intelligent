import numpy as np
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from typing import Dict, Any


class DriftStatistics:
    """
    

    Supports:
        - PSI
        - KS Statistic
        - KL Divergence
        - JS Divergence

    Designed for:
        - Production ML Monitoring
        - Feature-level Drift Detection
        - Numerical Stability
    """

    EPSILON = 1e-10

    

    

    @staticmethod
    def _validate_inputs(reference: np.ndarray, current: np.ndarray):
        if reference is None or current is None:
            raise ValueError("Reference and current arrays must not be None.")

        if len(reference) == 0 or len(current) == 0:
            raise ValueError("Reference and current arrays must not be empty.")

    @staticmethod
    def _clean_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        return arr

    @staticmethod
    def _numeric_distribution(reference, current, bins=10):
        """
        Uses SAME bin edges for reference and current.
        Critical for PSI/KL/JS correctness.
        """

        combined = np.concatenate([reference, current])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)

        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        ref_hist = ref_hist.astype(float)
        cur_hist = cur_hist.astype(float)

        ref_dist = ref_hist / (ref_hist.sum() + DriftStatistics.EPSILON)
        cur_dist = cur_hist / (cur_hist.sum() + DriftStatistics.EPSILON)

        # Stability
        ref_dist = np.clip(ref_dist, DriftStatistics.EPSILON, None)
        cur_dist = np.clip(cur_dist, DriftStatistics.EPSILON, None)

        return ref_dist, cur_dist

    
    # PSI
    

    @staticmethod
    def psi(reference, current, bins=10) -> Dict[str, Any]:
        """
        Population Stability Index
        """

        DriftStatistics._validate_inputs(reference, current)

        reference = DriftStatistics._clean_array(reference)
        current = DriftStatistics._clean_array(current)

        ref_dist, cur_dist = DriftStatistics._numeric_distribution(
            reference, current, bins
        )

        psi_values = (ref_dist - cur_dist) * np.log(ref_dist / cur_dist)
        psi_score = float(np.sum(psi_values))

        return {
            "metric": "PSI",
            "score": psi_score,
            "severity": DriftStatistics._psi_severity(psi_score)
        }

    @staticmethod
    def _psi_severity(score: float) -> str:
        if score < 0.1:
            return "NO_DRIFT"
        elif score < 0.25:
            return "MODERATE_DRIFT"
        return "SIGNIFICANT_DRIFT"

    
    # KS
    

    @staticmethod
    def ks_statistic(reference, current) -> Dict[str, Any]:
        DriftStatistics._validate_inputs(reference, current)

        reference = DriftStatistics._clean_array(reference)
        current = DriftStatistics._clean_array(current)

        statistic, p_value = ks_2samp(reference, current)

        return {
            "metric": "KS",
            "score": float(statistic),
            "p_value": float(p_value),
            "severity": "DRIFT" if statistic > 0.2 else "NO_DRIFT"
        }

    
    # KL Divergence
    

    @staticmethod
    def kl_divergence(reference, current, bins=10) -> Dict[str, Any]:
        DriftStatistics._validate_inputs(reference, current)

        reference = DriftStatistics._clean_array(reference)
        current = DriftStatistics._clean_array(current)

        ref_dist, cur_dist = DriftStatistics._numeric_distribution(
            reference, current, bins
        )

        kl_score = float(np.sum(rel_entr(ref_dist, cur_dist)))

        return {
            "metric": "KL",
            "score": kl_score,
            "severity": "DRIFT" if kl_score > 0.1 else "NO_DRIFT"
        }

    
    # JS Divergence
    

    @staticmethod
    def js_divergence(reference, current, bins=10) -> Dict[str, Any]:
        DriftStatistics._validate_inputs(reference, current)

        reference = DriftStatistics._clean_array(reference)
        current = DriftStatistics._clean_array(current)

        ref_dist, cur_dist = DriftStatistics._numeric_distribution(
            reference, current, bins
        )

        m = 0.5 * (ref_dist + cur_dist)

        kl_ref = np.sum(rel_entr(ref_dist, m))
        kl_cur = np.sum(rel_entr(cur_dist, m))

        js_score = float(0.5 * (kl_ref + kl_cur))

        return {
            "metric": "JS",
            "score": js_score,
            "severity": "DRIFT" if js_score > 0.1 else "NO_DRIFT"
        }

    
    # MASTER DRIFT REPORT


    @staticmethod
    def full_report(reference, current, bins=10) -> Dict[str, Any]:
        """
        Generates full drift analysis for a single feature.
        """

        return {
            "psi": DriftStatistics.psi(reference, current, bins),
            "ks": DriftStatistics.ks_statistic(reference, current),
            "kl": DriftStatistics.kl_divergence(reference, current, bins),
            "js": DriftStatistics.js_divergence(reference, current, bins),
        }