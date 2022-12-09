from typing import List

import dill

from .base_model import BaseModel
from .common import register_model


@register_model("offline_itemknn_model")
class OfflineItemKNNModel(BaseModel):
    def __init__(
        self,
        model_dill: str,
    ):
        with open(model_dill, "rb") as f:
            pickle_data = dill.load(f)
        self._data = pickle_data["recs"]
        self._popular_predicts = pickle_data["popular_recs"]

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id < len(self._data) and self._data[user_id] is not None:
            return self._data[user_id][:k_recs]
        return self._popular_predicts[:k_recs]
