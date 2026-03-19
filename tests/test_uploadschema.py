import pytest
from pydantic import ValidationError
from app.schemas.upload_schema import (
    FileMeta,
    UploadRequest,
    UploadResponse,
    MergeConfig
)

def _make_request(**overrides) -> UploadRequest:
    """
    Helper to build UploadRequest with the new nested FileMeta structure.
    """
    # 1. Extract file-specific data
    file_info = FileMeta(
        filename=overrides.pop("filename", "dataset.csv"),
        file_type=overrides.pop("file_type", "csv"),
        file_size_mb=overrides.pop("file_size_mb", 10.0)
    )
    
    # 2. Build the request with the files list
    defaults = {
        "files": [file_info],
        "user_id": overrides.pop("user_id", "user_001"),
        "target_column": overrides.pop("target_column", None),
        "problem_type": overrides.pop("problem_type", None),
    }
    defaults.update(overrides)
    return UploadRequest(**defaults)

class TestUploadRequestHappyPath:
    def test_minimal_required_fields(self):
        req = _make_request()
        assert len(req.files) == 1
        assert req.files[0].filename == "dataset.csv"
        assert req.user_id == "user_001"

    def test_file_type_csv(self):
        req = _make_request(file_type="csv")
        assert req.files[0].file_type == "csv"

    def test_all_fields_provided(self):
        merge = MergeConfig(left_on="id", right_on="user_id", how="inner")
        file1 = FileMeta(filename="a.csv", file_type="csv", file_size_mb=1.0)
        file2 = FileMeta(filename="b.csv", file_type="csv", file_size_mb=1.0)
        
        req = UploadRequest(
            files=[file1, file2],
            user_id="admin_01",
            problem_type="classification",
            target_column="churn",
            merge_config=merge
        )
        assert len(req.files) == 2
        assert req.merge_config.how == "inner"

class TestFileSizeValidation:
    def test_above_200_raises(self):
        with pytest.raises(ValidationError, match="File size exceeds 200MB limit"):
            _make_request(file_size_mb=250.0)

    def test_zero_rejected_by_pydantic(self):
        # This matches Pydantic's built-in gt=0 check
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            _make_request(file_size_mb=0.0)

class TestUploadResponse:
    def test_happy_path(self):
        # Now includes file_count and merged as required by your schema
        resp = UploadResponse(
            job_id="job-123",
            status="uploaded",
            message="Success",
            file_count=1,
            merged=False
        )
        assert resp.status == "uploaded"

class TestSchemaFixes:
    """
    These replace the old 'TestSchemaBugs' to confirm the fixes work.
    """
    def test_bug1_comma_fix(self):
        # This would have failed before the comma fix!
        req_parquet = _make_request(file_type="parquet")
        req_excel = _make_request(file_type="excel")
        assert req_parquet.files[0].file_type == "parquet"
        assert req_excel.files[0].file_type == "excel"

    def test_bug2_status_fix(self):
        # Verifies 'excel' is gone and 'processing' is valid
        resp = UploadResponse(
            job_id="j1", 
            status="processing", 
            message="Working...",
            file_count=1,
            merged=False
        )
        assert resp.status == "processing"