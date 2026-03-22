"""
Waste Classifier - SQLite Storage Backend
Zero-setup fallback database using Python's built-in sqlite3 module.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.base import StorageBackend
from database.models import WasteRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteBackend(StorageBackend):
    """
    SQLite storage backend for waste classification records.

    Uses a single table 'waste_records' with JSON columns for 
    nested data (detections, predictions, probabilities).
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS waste_records (
        record_id            TEXT PRIMARY KEY,
        timestamp            TEXT NOT NULL,
        image_path           TEXT NOT NULL,
        final_category       TEXT NOT NULL,
        final_confidence     REAL NOT NULL,
        strategy             TEXT,
        yolo_detections      TEXT, -- JSON
        cnn_predictions      TEXT, -- JSON
        all_probabilities    TEXT, -- JSON
        source               TEXT,
        notes                TEXT
    );
    """

    CREATE_INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_category ON waste_records(final_category);", 
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON waste_records (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx source ON waste_records (source);",
    ]
        
    def __init__(self, db_path: str = "data/waste_classifier.db") -> None:
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path= db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Create/open the SQLite database and ensure tables exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        # Create table and indices
        self._conn.execute(self.CREATE_TABLE_SQL)
        for idx_sql in self.CREATE_INDEX_SQL:
            self._conn.execute(idx_sql)
        self._conn.commit()

        logger.info("SQLite connected: %s", self.db_path)

    def disconnect(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite disconnected.")
        
    def _ensure_connected(self) -> None:
        if self._conn is None:
            self.connect()

    #------------------------------------------------------------#
    # CRUD Operations
    #------------------------------------------------------------#

    def insert_record(self, record: WasteRecord) -> str:
        """Insert a single record."""
        self._ensure_connected()

        self._conn.execute(
            """
            INSERT INTO waste_records
                (record_id, timestamp, image_path, final_category, 
                final_confidence, strategy, yolo_detections, 
                cnn_predictions, all_probabilities, source, notes) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (    
                record.record_id,
                record.timestamp,
                record.image_path,
                record.final_category,
                record.final_confidence,
                record.strategy, 
                json.dumps(record.yolo_detections),
                json.dumps(record.cnn_predictions),
                json.dumps(record.all_probabilities),
                record.source, 
                record.notes,
            ),
        )
        self._conn.commit()
        logger.debug("Inserted record %s", record.record_id) 
        return record.record_id
    
    def insert_many(self, records: List[WasteRecord]) -> List[str]: 
        """Insert multiple records in a single transaction."""
        self._ensure_connected()

        ids = []
        with self._conn:
            for record in records:
                self._conn.execute(
                    """
                    INSERT INTO waste_records
                        (record_id, timestamp, image_path, final_category, 
                        final_confidence, strategy, yolo_detections, 
                        cnn_predictions, all_probabilities, source, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.timestamp,
                        record.image_path,
                        record.final_category, 
                        record.final_confidence,
                        record.strategy,
                        json.dumps(record.yolo_detections),
                        json.dumps(record.cnn_predictions),
                        json.dumps(record.all_probabilities),
                        record.source,
                        record.notes,
                    ),
                )
                ids.append(record.record_id)

        logger.info("Inserted %d records.", len(ids)) 
        return ids
        
    def get_record(self, record_id: str) -> Optional[WasteRecord]:
        """Retrieve a single record by ID."""
        self._ensure_connected()

        row = self._conn.execute(
            "SELECT * FROM waste_records WHERE record_id= ?", (record_id,) 
        ).fetchone()
        
        return self._row_to_record(row) if row else None
        
    def get_all_records( 
        self, limit: Optional[int] = None, offset: int = 0 
    )-> List[WasteRecord]: 
        """Retrieve all records with optional pagination."""
        self._ensure_connected()

        query = "SELECT FROM waste_records ORDER BY timestamp DESC"
        params: list = []

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = [limit, offset]

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]
    
    def query_by_category(self, category: str) -> List[WasteRecord]:
        """Retrieve all records for a given waste category."""
        self._ensure_connected()

        rows = self._conn.execute(
            "SELECT FROM waste_records WHERE final_category = ? ORDER BY timestamp DESC", 
            (category,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]
    
    def query_by_date_range(
        self, start_date: str, end_date: str
    )-> List[WasteRecord]:
        """Retrieve records within a date range (ISO format)."""
        self._ensure_connected()

        rows = self._conn.execute(
            """
            SELECT FROM waste_records 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            """
            (start_date, end_date),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for reporting/PowerBI."""
        self._ensure_connected()
    
        total = self._conn.execute(
            "SELECT COUNT(*) FROM waste_records"
        ).fetchone()[0]
        
        avg_conf = self._conn.execute(
            "SELECT AVG(final_confidence) FROM waste_records"
        ).fetchone()[0] or 0.0
        
        category_counts = {}
        rows = self._conn.execute(
            "SELECT final_category, COUNT(*) as cnt FROM waste_records GROUP BY final_category ORDER BY cnt DESC"
        ).fetchall()
        for row in rows:
            category_counts[row["final_category"]] = row["cnt"]
        
        source_counts = {}
        rows = self._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM waste_records GROUP BY source"
        ).fetchall()
        for row in rows:
            source_counts[row["source"]] = row["cnt"]
        
        date_range = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM waste_records"
        ).fetchone()
        
        # Strategy distribution
        strategy_counts = {}
        rows = self._conn.execute(
            "SELECT strategy, COUNT(*) as cnt FROM waste_records GROUP BY strategy" 
        ).fetchall()
        for row in rows:
            strategy_counts[row["strategy"]] = row["cnt"]
        
        return {
            "total_records": total,
            "average_confidence": round(avg_conf, 4),
            "category_counts": category_counts,
            "source_counts": source_counts,
            "strategy_counts": strategy_counts,
            "date_range": {
                "earliest": date_range[0] if date_range else None,
                "latest": date_range[1] if date_range else None,
            },
        } 
        
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        self._ensure_connected()
        
        cursor = self._conn.execute(
            "DELETE FROM waste_records WHERE record_id = ?", (record_id,)
        ) 
        self._conn.commit()
        return cursor.rowcount > 0
    
    def count(self) -> int:
        """Return total record count."""
        self._ensure_connected()
        return self._conn.execute("SELECT COUNT(*) FROM waste_records").fetchone()[0]
    
    #------------------------------------------------------------#    
    # Helpers
    #------------------------------------------------------------#

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> WasteRecord: 
        """Convert a sqlite3.Row to a WasteRecord."""
        return WasteRecord(
            record_id=row["record_id"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            final_category=row["final_category"],
            final_confidence=row["final_confidence"],
            strategy=row["strategy"],
            yolo_detections=json.loads(row["yolo_detections"] or "[]"),
            cnn_predictions=json.loads(row["cnn_predictions"] or "[]"),
            all_probabilities=json.loads(row["all_probabilities"] or "{}"),
            source=row["source"],
            notes=row["notes"] or "",
        )
