from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftReport:
    is_drifted: bool
    drift_scores: Dict[str, float]
    flagged_features: List[str]
    threshold: float


class DriftDetector:

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def check_drift(
        self,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:

        features = features or reference_df.columns
        drift_scores: Dict[str, float] = {}
        flagged: List[str] = []

        for col in features:

            if col not in current_df.columns:
                continue

            # Only numeric columns
            if reference_df[col].dtype not in (
                pl.Float32, pl.Float64, pl.Int32, pl.Int64
            ):
                continue

            ref_data = reference_df[col].drop_nulls().to_numpy()
            curr_data = current_df[col].drop_nulls().to_numpy()

            # Skip if empty
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue

            try:
                _, p_value = ks_2samp(ref_data, curr_data)
                p_value = float(p_value)
            except Exception as e:
                logger.warning(f"Failed KS test on {col}: {e}")
                continue

            drift_scores[col] = p_value

            if p_value < self.threshold:
                flagged.append(col)
                logger.warning(
                    f"Drift detected in '{col}': p={p_value:.5f}"
                )

        return DriftReport(
            is_drifted=len(flagged) > 0,
            drift_scores=drift_scores,
            flagged_features=flagged,
            threshold=self.threshold,
        )