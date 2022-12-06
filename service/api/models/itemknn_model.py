from typing import List

import dill
import pandas as pd
from rectools.dataset import Dataset

from .base_model import BaseModel
from .common import register_model


@register_model("itemknn_model")
class ItemKNNModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        popular_model_path: str,
        interactions: pd.DataFrame,
    ):
        with open(model_path, "rb") as f:
            self._model = dill.load(f)
        with open(popular_model_path, "rb") as f:
            self._popular_model = dill.load(f)
        self._interactions = interactions
        self._dataset = Dataset.construct(
            interactions_df=interactions,
            user_features_df=None,
            item_features_df=None,
        )

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id not in self._dataset.user_id_map.external_ids:
            return list(
                self._popular_model.recommend(
                    [self._dataset.user_id_map.external_ids[0]],
                    dataset=self._dataset,
                    k=k_recs,
                    filter_viewed=False,
                )["item_id"]
            )
        return list(
            self._model.recommend(
                [user_id],
                dataset=self._dataset,
                k=k_recs,
                filter_viewed=False,
            )["item_id"]
        )
