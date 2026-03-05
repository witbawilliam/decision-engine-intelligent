import pytest
from app.schemas.upload_schema import UploadRequest


def test_valid_upload():
    request = UploadRequest(
        filename="sales.csv",
        file_type="csv",
        file_size_mb=10,
        target_column="sales",
        problem_type="regression",
        user_id="user123",
    )

    assert request.filename == "sales.csv"


def test_large_file_rejected():
    with pytest.raises(ValueError):
        UploadRequest(
            filename="big.csv",
            file_type="csv",
            file_size_mb=500,
            user_id="user123",
        )


def test_invalid_filename():
    with pytest.raises(ValueError):
        UploadRequest(
            filename="invalidfile",
            file_type="csv",
            file_size_mb=10,
            user_id="user123",
        )
