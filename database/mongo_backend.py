"""
Waste Classifier - MongoDB Storage Backend
Primary database backend using MongoDB for rich document storage.
Requires: pymongo, a running MongoDB instance.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from database.base import StorageBackend 
from database.models import WasteRecord 
from utils.logger import get_logger

logger = get_logger(__name__)


class MongoDBBackend(StorageBackend):
    """
    MongoDB storage backend for waste classification records.
    
    Each record is stored as a document in the waste_records collection. 
    MongoDB's flexible schema naturally handles nested YOLO detections 
    and CNN probability distributions.
    """

    def __init__(
        self,
        connection_uri: str = "mongodb://localhost:27017",
        database_name: str = "waste_classifier",
        collection_name: str = "waste_records",
    )-> None:
        """
        Initialize MongoDB backend.

        Args:
            connection_uri: MongoDB connection string.
            database_name: Name of the database.
            collection_name: Name of the collection.
        """
        self.connection_uri = connection_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self._client = None
        self._db = None
        self._collection = None

    def connect(self) -> None:
        """Connect to MongoDB and ensure indexes exist."""
        try:
            from pymongo import MongoClient, ASCENDING, DESCENDING 
        except ImportError as e: 
            raise ImportError( 
                "pymongo is required for MongoDB backend. "
                "Install via: pip install pymongo" 
            ) from e
        
        self._client = MongoClient( 
            self.connection_uri, 
            serverSelectionTimeoutMS=5000,
        ) 

        # Verify connection
        try:
            self._client.admin.command("ping")
        except Exception as e:
            raise ConnectionError( 
                f"Cannot connect to MongoDB at {self.connection_uri}." 
                f"Ensure MongoDB is running. Error: {e}" 
            ) from e
        
        self._db = self._client[self.database_name]
        self._collection = self._db[self.collection_name]

        # Create indexes for efficient queries
        self._collection.create_index([("record_id", ASCENDING)], unique=True)
        self._collection.create_index([("final_category", ASCENDING)])
        self._collection.create_index([("timestamp", DESCENDING)])
        self._collection.create_index([("source", ASCENDING)])

        logger.info(
            "MongoDB connected: %s/%s.%s", 
            self.connection_uri, self.database_name, self.collection_name,
        )

    def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
            logger.info("MongoDB disconnected.")

    def _ensure_connected(self) -> None:
        if self._collection is None: 
            self.connect()
    
    #------------------------------------------------------------#
    # CRUD Operations
    #------------------------------------------------------------#

    def insert_record(self, record: WasteRecord) -> str:
        """Insert a single record as a MongoDB document."""
        self._ensure_connected()

        doc = record.to_dict()
        doc["_id"]  = record.record_id # Use record_id as MongoDB _id

        self._collection.insert_one(doc) 
        logger.debug("Inserted record %s", record.record_id) 
        return record.record_id
    
    def insert_many(self, records: List[WasteRecord]) -> List[str]: 
        """Insert multiple records."""
        self._ensure_connected()

        docs = []
        ids = []
        for record in records: 
            doc = record.to_dict() 
            doc["_id"] = record.record_id 
            docs.append(doc) 
            ids.append(record.record_id)

        if docs:
            self._collection.insert_many(docs)

        logger.info("Inserted %d records into MongoDB.", len(ids))
        return ids
        
    def get_record(self, record_id: str) -> Optional[WasteRecord]: 
        """Retrieve a single record by ID."""
        self._ensure_connected()

        doc = self._collection.find_one({"record_id": record_id})
        if doc is None:
            return None
        
        doc.pop("_id", None) 
        return WasteRecord.from_dict(doc)
    
    def get_all_records(
        self, limit: Optional[int] = None, offset: int = 0
    )-> List[WasteRecord]:
        """Retrieve all records with optional pagination."""
        self._ensure_connected()

        cursor = self._collection.find().sort("timestamp", -1).skip(offset)
        if limit is not None:
            cursor = cursor.limit(limit)

        records = []
        for doc in cursor:
            doc.pop("_id", None)
            records.append(WasteRecord.from_dict(doc))
        
        return records
    
    def query_by_category(self, category: str) -> List[WasteRecord]:
        """Retrieve all records for a given waste category."""
        self._ensure_connected()

        cursor = self._collection.find(
            {"final_category": category}
        ).sort("timestamp", -1)

        records = []
        for doc in cursor: 
            doc.pop("_id", None) 
            records.append(WasteRecord.from_dict(doc)) 
        return records
    
    def query_by_date_range(
        self, start_date: str, end_date: str 
    )-> List[WasteRecord]:
        """Retrieve records within a date range (ISO format)."""
        self._ensure_connected()

        cursor = self._collection.find( 
            {"timestamp": {"$gte": start_date, "$lte": end_date}} 
        ).sort("timestamp", -1)

        records = []
        for doc in cursor:
            doc.pop("_id", None) 
            records.append(WasteRecord.from_dict(doc))
        return records
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics using MongoDB aggregation pipeline."""
        self._ensure_connected()

        total = self._collection.count_documents({})

        # Average confidence
        pipeline_avg = [
            {"$group": {"_id": None, "avg_conf": {"$avg": "$final_confidence"}}}
        ]
        avg_result = list(self._collection.aggregate(pipeline_avg))
        avg_conf = avg_result[0]["avg_conf"] if avg_result else 0.0
        
        # Category counts
        pipeline_cat = [
            {"$group": {"_id": "$final_category", "count": {"$sum": 1}}}, 
            {"$sort": {"count": -1}},
        ]
        category_counts = {
            doc["_id"]: doc["count"]
            for doc in self._collection.aggregate(pipeline_cat)
        }
        
        # Source counts
        pipeline_src = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
        ] 
        source_counts = {
            doc["_id"]: doc["count"]
            for doc in self._collection.aggregate(pipeline_src)
        } 

        # Strategy counts
        pipeline_strat = [
            {"$group": {"_id": "$strategy", "count": {"$sum": 1}}},
        ]
        strategy_counts = {
            doc["_id"]: doc["count"]
            for doc in self._collection.aggregate(pipeline_strat) 
        }
        
        # Date range
        pipeline_dates = [
            {
                "$group": {
                    "_id": None,
                    "earliest": {"$min": "$timestamp"},
                    "latest": {"$max": "$timestamp"},
                }
            }
        ]
        date_result = list(self._collection.aggregate(pipeline_dates))

        return {
            "total_records": total,
            "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "category_counts": category_counts,
            "source_counts": source_counts,
            "strategy counts": strategy_counts,
            "date_range": {
                "earliest": date_result[0]["earliest"] if date_result else None, 
                "latest": date_result[0]["latest"] if date_result else None,
            }, 
        }
    
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        self._ensure_connected()
        
        result = self._collection.delete_one({"record_id": record_id})
        return result.deleted_count > 0
        
    def count(self) -> int:
        """Return total record count."""
        self._ensure_connected()
        return self._collection.count_documents({})