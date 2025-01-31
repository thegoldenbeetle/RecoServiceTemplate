from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    @abstractmethod
    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        pass
