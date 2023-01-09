from typing import List, Optional

from .base_model import BaseModel
from .common import load_dill, register_model


@register_model("offline_itemknn_model")
class OfflineItemKNNModel(BaseModel):
    def __init__(
        self,
        model_dill: str,
    ):
        pickle_data = load_dill(model_dill)
        self._data: List[Optional[List[int]]] = pickle_data["recs"]
        self._popular_predicts: List[int] = pickle_data["popular_recs"]

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id < len(self._data) and self._data[user_id] is not None:
            return self._data[user_id][:k_recs]
        return self._popular_predicts[:k_recs]
