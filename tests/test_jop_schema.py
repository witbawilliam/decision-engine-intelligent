from datetime import datetime
from app.schemas.job_schema import JobMetadata


def test_job_metadata_creation():
    job = JobMetadata(
        job_id="job_123",
        user_id="user_1",
        filename="sales.csv",
        status="uploaded",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert job.status == "uploaded"
    assert job.job_id == "job_123"
