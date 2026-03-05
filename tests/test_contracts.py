import pytest
import polars as pl
from core.contracts.schema_validator import (
    DatasetSchema, 
    SchemaValidator, 
    ProblemType, 
    ColumnContract, 
    SchemaViolationError
)

# --- Fixtures: Reusable Data Components ---

@pytest.fixture
def valid_df():
    """A standard healthy dataset."""
    return pl.DataFrame({
        "timestamp": [1715644800, 1715731200], # Unix dates
        "feature_a": [1.0, 2.5],
        "target": [0, 1]
    }).with_columns(pl.from_epoch("timestamp", time_unit="s"))

@pytest.fixture
def sales_schema():
    """A valid enterprise schema contract."""
    return {
        "dataset_name": "Sales_Data",
        "target_column": "target",
        "columns": [
            {"name": "feature_a", "dtype": "float"},
            {"name": "target", "dtype": "int"}
        ]
    }

# --- Test Cases ---

def test_contract_normalization(sales_schema):
    """Verify that List[ColumnContract] is correctly mapped to a Dictionary."""
    schema = DatasetSchema(**sales_schema)
    assert isinstance(schema.columns, dict)
    assert "feature_a" in schema.columns
    assert schema.columns["feature_a"].name == "feature_a"

def test_validation_success(valid_df, sales_schema):
    """Test the 'Happy Path' where data matches the contract."""
    schema = DatasetSchema(**sales_schema)
    validator = SchemaValidator(valid_df, schema, ProblemType.CLASSIFICATION)
    
    result = validator.validate()
    assert result.valid is True
    assert len(result.errors) == 0

def test_missing_target_column(valid_df, sales_schema):
    """Test semantic failure: Target defined in schema but missing in DataFrame."""
    schema = DatasetSchema(**sales_schema)
    # Drop target from the data
    df_missing_target = valid_df.drop("target")
    
    validator = SchemaValidator(df_missing_target, schema, ProblemType.CLASSIFICATION)
    result = validator.validate()
    
    assert result.valid is False
    assert any("Missing Column" in err for err in result.errors)

def test_zero_variance_target(valid_df, sales_schema):
    """Test the math: Target with only one unique value should fail."""
    schema = DatasetSchema(**sales_schema)
    # Make target constant
    df_constant = valid_df.with_columns(pl.lit(1).alias("target"))
    
    validator = SchemaValidator(df_constant, schema, ProblemType.REGRESSION)
    result = validator.validate()
    
    assert result.valid is False
    assert any("Zero Variance" in err for err in result.errors)

def test_forecasting_requires_date(valid_df, sales_schema):
    """Verify that forecasting fails if no temporal axis is present."""
    schema = DatasetSchema(**sales_schema)
    # Remove the date column to trigger failure
    df_no_date = valid_df.drop("timestamp")
    
    validator = SchemaValidator(df_no_date, schema, ProblemType.FORECASTING)
    result = validator.validate()
    
    assert result.valid is False
    # Use 'any' with a substring check for professional resilience
    expected_msg = "Forecasting requires at least one Date/Datetime column"
    assert any(expected_msg in err for err in result.errors), f"Expected message '{expected_msg}' not found in {result.errors}"

def test_exception_raising(valid_df, sales_schema):
    """Verify the validator correctly raises SchemaViolationError when requested."""
    schema = DatasetSchema(**sales_schema)
    df_empty = pl.DataFrame()
    
    validator = SchemaValidator(df_empty, schema, ProblemType.REGRESSION)
    
    with pytest.raises(SchemaViolationError) as excinfo:
        validator.validate(raise_on_failure=True)
    
    assert "Critical: Input DataFrame is empty." in str(excinfo.value)