"""MongoDB repository for persisting prediction metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from bson.objectid import ObjectId

from pymongo import MongoClient


class MongoPredictionRepository:
    """Persistence adapter that writes prediction records to MongoDB."""

    def __init__(
        self,
        mongo_uri: str,
        database_name: str,
        collection_name: str,
        client: MongoClient | None = None,
    ):
        self._client = client or MongoClient(mongo_uri)
        self._collection = self._client[database_name][collection_name]

    def save_prediction(
        self,
        source: str,
        model_version: str,
        predictions: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist one prediction event and return the inserted ID."""

        document = {
            "source": source,
            "model_version": model_version,
            "predictions": predictions,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        }
        insert_result = self._collection.insert_one(document)
        return str(insert_result.inserted_id)

    def fetch_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return most recent prediction events in descending time order."""

        cursor = self._collection.find().sort("created_at", -1).limit(limit)
        records: list[dict[str, Any]] = []
        for item in cursor:
            item["id"] = str(item.pop("_id"))
            records.append(item)
        return records

    def delete_prediction(self, record_id: str) -> bool:
        """Delete a prediction record by its ID."""

        try:
            result = self._collection.delete_one({"_id": ObjectId(record_id)})
            return result.deleted_count > 0
        except Exception:
            return False
