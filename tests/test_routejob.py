import sys
from unittest.mock import MagicMock, patch


mock_celery_app = MagicMock()
mock_workers_module = MagicMock()
mock_workers_module.celery_app = mock_celery_app
sys.modules["workers.celery_app"] = mock_workers_module


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.routes_jobs import router 
from app.schemas.job_schema import V1

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)

class TestModelTrainingRoutes:

    @pytest.fixture
    def valid_payload(self):
        return {
            # Must be a valid UUID format
            "user_id": "550e8400-e29b-41d4-a716-446655440000", 
            "filename": "dataset.csv",
            "target_column": "target",
            "problem_type": "classification",
            # Must be 16-128 characters (printable ASCII)
            "idempotency_key": "standard-length-idempotency-key-v1" 
        }

    

    def test_validate_dataset_success(self, client, valid_payload):
        with patch.object(V1, "validate_filename"), \
             patch.object(V1, "validate_column_name"):
            
            response = client.post("/v1/train/validate", json=valid_payload)
            assert response.status_code == 202
            assert response.json()["job_id"] == valid_payload["idempotency_key"]

    def test_train_tabular_success(self, client, valid_payload):
        with patch.object(V1, "validate_filename"), \
             patch.object(V1, "validate_column_name"):
            
            response = client.post("/v1/train/tabular", json=valid_payload)
            assert response.status_code == 202
            mock_celery_app.dispatch_automl_task.assert_called_with("train", valid_payload)

    

    def test_train_tabular_rejects_forecasting(self, client, valid_payload):
        """Now passes validation (422) and triggers your custom 400 logic."""
        payload = {**valid_payload, "problem_type": "forecasting"}
        response = client.post("/v1/train/tabular", json=payload)
        
        assert response.status_code == 400
        assert "forecast" in response.json()["detail"].lower()

    def test_validation_error_returns_422(self, client, valid_payload):
        """Tests that a ValueError in the internal route logic returns 422."""
        with patch.object(V1, "validate_filename", side_effect=ValueError("Invalid file extension")):
            response = client.post("/v1/train/tabular", json=valid_payload)
            
        assert response.status_code == 422
        assert "Invalid file extension" in response.json()["detail"]

    def test_broker_exception_returns_500(self, client, valid_payload):
        """Verify 500 status when the Celery dispatcher fails."""
        mock_celery_app.dispatch_automl_task.side_effect = Exception("Internal Connection Error")
        
        with patch.object(V1, "validate_filename"), \
             patch.object(V1, "validate_column_name"):
            
            response = client.post("/v1/train/tabular", json=valid_payload)
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]


   