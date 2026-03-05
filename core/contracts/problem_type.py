from enum import Enum

class ProblemType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    BINARY = "binary_classification"
    UNKNOWN = "unknown"