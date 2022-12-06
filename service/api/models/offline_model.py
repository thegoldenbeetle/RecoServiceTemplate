from typing import List

import numpy as np
import pandas as pd

from .base_model import BaseModel
from .common import register_model


@register_model("offline_itemknn_model")
class OfflineItemKNNModel(BaseModel):
    def __init__(
        self,
        data_csv: str,
        popular_predicts: str,
    ):
        self._data = pd.read_csv(
            data_csv, converters={"item_id": self._items_to_list}
        )
        self._popular_predicts = np.loadtxt(popular_predicts)

    @staticmethod
    def _items_to_list(items_str):
        return list(map(int, items_str[1:-1].split(", ")))

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id not in list(self._data["user_id"]):
            return self._popular_predicts[:k_recs].tolist()
        return self._data[self._data["user_id"] == user_id]["item_id"].iloc[0][
            :k_recs
        ]
