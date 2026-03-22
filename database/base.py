"""


"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from database.models import WasteRecord


class StorageBackend(ABC):
    """"""

    @abstractmethod
    def connect(self) -> None:
        """"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """"""
        ...

    @abstractmethod
    def insert_record(self, record: WasteRecord) -> str:
        """
        
        
        
        
        
        
        
        """
        ...

    @abstractmethod
    def insert_many(self, records: List[WasteRecord]) -> List[str]:
        """
        
        
        
        
        
        
        
        """
        ...

    @abstractmethod
    def get_record(self, record_id: str) -> Optional[WasteRecord]:
        """"""
        ...

    @abstractmethod
    def get_all_records(
        self, 
        limit: Optional[int] = None, 
        offset: int = 0,
    ) -> List[WasteRecord]:
        """"""
        ...

    @abstractmethod
    def  query_by_category(self, category: str) -> List[WasteRecord]:
        """"""
        ...

    @abstractmethod
    def query_by_date_range(
        self, start_date: str, end_date:str,
    ) -> List[WasteRecord]:
        """"""
        ...

    @abstractmethod
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        
        
        
        
           
        """
        ...

    @abstractmethod
    def delete_record(self, record_id: str) -> bool:
        """"""
        ...

    @abstractmethod
    def count(self) -> int:
        """"""
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()