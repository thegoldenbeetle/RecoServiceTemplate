import random
from typing import List

import numpy as np
from pandas import Series
from rectools.dataset import Dataset, IdMap
from rectools.models import ImplicitItemKNNWrapperModel, PopularModel
from scipy import sparse

from .base_model import BaseModel
from .common import load_dill, register_model


@register_model("itemknn_model")
class ItemKNNModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        popular_model_path: str,
        dataset: Dataset,
    ):
        self._model: ImplicitItemKNNWrapperModel = load_dill(model_path)
        self._popular_model: PopularModel = load_dill(popular_model_path)
        self._dataset: Dataset = dataset
        self._user_items: sparse.csr_matrix = dataset.get_user_item_matrix(
            include_weights=True
        )
        self._item_id_map: IdMap = dataset.item_id_map
        self._to_intornal_user_id: Series = dataset.user_id_map.to_internal
        self._popular_predicts: List[int] = self._predict_popular_model(
            k_recs=100
        )

    def _predict_model(
        self,
        user_id: int,
        k_recs: int = 100,
    ) -> List[int]:
        user_id = self._to_intornal_user_id[user_id]
        # pylint: disable=protected-access
        rec_ids, _ = self._model._recommend_for_user(
            user_id,
            self._user_items,
            k_recs,
            filter_viewed=True,
            sorted_item_ids=None,
        )
        while len(rec_ids) < k_recs:
            item = random.choice(self._item_id_map.internal_ids)
            if (
                item not in self._user_items[user_id].indices
                and item not in rec_ids
            ):
                rec_ids = np.append(rec_ids, item)
        return self._item_id_map.convert_to_external(rec_ids).tolist()

    def _predict_popular_model(self, k_recs: int = 100) -> List[int]:
        return list(
            self._popular_model.recommend(
                [self._dataset.user_id_map.external_ids[0]],
                dataset=self._dataset,
                k=k_recs,
                filter_viewed=False,
            )["item_id"]
        )

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id not in self._dataset.user_id_map.external_ids:
            if k_recs <= 100:
                return self._popular_predicts[:k_recs]
            return self._predict_popular_model(k_recs)
        return self._predict_model(user_id, k_recs)
