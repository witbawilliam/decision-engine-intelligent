from __future__ import annotations

import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Any, Dict, Optional



# Model Registry


class ModelRegistry:

    def __init__(self, base_path: str = "ml_registry"):

        self.base_path = base_path
        self.artifacts_path = os.path.join(base_path, "artifacts")
        self.metadata_path = os.path.join(base_path, "metadata")

        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        self.index_file = os.path.join(self.metadata_path, "models.json")

        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                json.dump({}, f)


    # REGISTER MODEL
    

    def register(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        problem_type: str,
        stage: str = "staging",
    ) -> Dict[str, Any]:

        version = self._generate_version(model_name)

        model_dir = os.path.join(
            self.artifacts_path,
            model_name,
            f"v{version}"
        )

        os.makedirs(model_dir, exist_ok=True)

        # Save model artifact
        model_path = os.path.join(model_dir, "model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        
        metadata = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "problem_type": problem_type,
            "metrics": metrics,
            "parameters": parameters,
            "artifact_path": model_path,
            "created_at": datetime.utcnow().isoformat(),
            "run_id": str(uuid.uuid4()),
        }

        metadata_path = os.path.join(model_dir, "metadata.json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Update global index
        self._update_index(model_name, metadata)

        return metadata


    

    def load(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Any:

        index = self._load_index()

        if model_name not in index:
            raise ValueError("Model not found")

        versions = index[model_name]

        if stage:
            versions = [
                v for v in versions
                if v["stage"] == stage
            ]

        if not versions:
            raise ValueError("No matching model version found")

        if version:
            metadata = next(
                (v for v in versions if v["version"] == version),
                None
            )
        else:
            metadata = sorted(
                versions,
                key=lambda x: x["version"],
                reverse=True
            )[0]

        if metadata is None:
            raise ValueError("Version not found")

        with open(metadata["artifact_path"], "rb") as f:
            model = pickle.load(f)

        return model

    

    

    def promote(
        self,
        model_name: str,
        version: int,
        new_stage: str,
    ) -> None:

        index = self._load_index()

        if model_name not in index:
            raise ValueError("Model not found")

        for entry in index[model_name]:
            if entry["version"] == version:
                entry["stage"] = new_stage

        self._save_index(index)


    # INTERNAL METHODS
    

    def _generate_version(self, model_name: str) -> int:

        index = self._load_index()

        if model_name not in index:
            return 1

        existing_versions = [
            entry["version"]
            for entry in index[model_name]
        ]

        return max(existing_versions) + 1

    def _update_index(self, model_name: str, metadata: Dict[str, Any]):

        index = self._load_index()

        if model_name not in index:
            index[model_name] = []

        index[model_name].append(metadata)

        self._save_index(index)

    def _load_index(self) -> Dict[str, Any]:

        with open(self.index_file, "r") as f:
            return json.load(f)

    def _save_index(self, index: Dict[str, Any]):

        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=4)