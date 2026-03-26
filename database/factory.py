"""


"""

from typing import Optional

from database.base import StorageBackend
from utils.logger import get_logger

logger = get_logger(__name__)


def get_storage_backend(
    backend: str = "mongodb",
    **kwargs,
) -> StorageBackend:
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    backend = backend.lower().strip()

    if backend == "mongodb":
        try:
            from database.mongo_backend import MongoDBBackend

            return MongoDBBackend(
                connection_uri=kwargs.get("connection_uri", "mongodb://localhost:27017"),
                database_name=kwargs.get("database_name", "waste_classifier"),
                collection_name=kwargs.get("collection_name", "waste_records"),
            )
        except ImportError:
            logger.warning(
                "pymongo not installed. Falling back to SQLite backend."
                "Install via: pip install pymongo"
            )
            backend = "sqlite" # Fall through to SQLite
    
    if backend == "sqlite":
        from database.sqlite_backend import SQLiteBackend

        return SQLiteBackend(
            db_path=kwargs.get("db_path", "data/waste_classifier.db"),
        )
    
    raise ValueError(
        f"Unknown backend: '{backend}'. Supported: 'mongodb', 'sqlite'."
    )