import numpy as np
import polars as pl

class BinaryRefinementSearch:
    """
    Finds the exact 'Lever' value required to hit a 'Target' outcome
    without retraining the model.
    """
    
    def __init__(self, model, precision: float = 0.01, max_iters: int = 15):
        self.model = model
        self.precision = precision
        self.max_iters = max_iters

    def optimize(self, 
                 base_row: pl.DataFrame, 
                 lever_col: str, 
                 target_goal: float, 
                 bounds: tuple):
        """
        base_row: The original data point (constant features)
        lever_col: The feature we are tweaking (e.g., 'price')
        target_goal: The desired prediction (e.g., 1000000)
        bounds: (min_allowable, max_allowable)
        """
        low, high = bounds
        best_lever_value = low
        
        #  Determine direction: Does increasing lever increase target?
        test_low = base_row.with_columns(pl.lit(low).alias(lever_col))
        test_high = base_row.with_columns(pl.lit(high).alias(lever_col))
        
        pred_low = self.model.predict(test_low)[0]
        pred_high = self.model.predict(test_high)[0]
        
        # If the goal is outside the model's current range, return the boundary
        if target_goal <= min(pred_low, pred_high): return low
        if target_goal >= max(pred_low, pred_high): return high

        #  Binary Search Loop
        for _ in range(self.max_iters):
            mid = (low + high) / 2
            current_row = base_row.with_columns(pl.lit(mid).alias(lever_col))
            current_pred = self.model.predict(current_row)[0]

            if abs(current_pred - target_goal) < self.precision:
                return round(mid, 4)

            # Adjust bounds based on directionality
            if pred_high > pred_low:  # Positive correlation
                if current_pred < target_goal: low = mid
                else: high = mid
            else:  # Negative correlation
                if current_pred < target_goal: high = mid
                else: low = mid
                
        return round((low + high) / 2, 4)