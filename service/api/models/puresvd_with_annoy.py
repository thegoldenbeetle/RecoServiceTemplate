import random
from typing import List

from annoy import AnnoyIndex
from pandas import Series
from rectools.dataset import Dataset, IdMap
from rectools.models import PureSVDModel
from rectools.models.utils import get_viewed_item_ids
from scipy import sparse

from .base_model import BaseModel
from .common import load_dill, register_model


@register_model("puresvd")
class PureSVDWithAnnoy(BaseModel):
    def __init__(
        self,
        model_path: str,
        popular_path: str,
        dataset_path: str,
    ):
        self._model: PureSVDModel = load_dill(model_path)
        self._popular_predicts: List[int] = load_dill(popular_path)
        dataset: Dataset = load_dill(dataset_path)
        (
            self._user_embeddings,
            item_embeddings,
        ) = self._model.get_vectors()
        self._item_id_map: IdMap = dataset.item_id_map
        self._to_intornal_user_id: Series = dataset.user_id_map.to_internal
        self._user_items: sparse.csr_matrix = dataset.get_user_item_matrix(
            include_weights=True
        )

        self._annoy_index = AnnoyIndex(
            self._user_embeddings.shape[1], "euclidean"
        )
        for i, emb in enumerate(item_embeddings):
            self._annoy_index.add_item(i, emb)
        self._annoy_index.build(10)

    def _predict_model(
        self,
        user_id: int,
        k_recs: int = 100,
    ) -> List[int]:
        user_id = self._to_intornal_user_id[user_id]
        viewed_ids = get_viewed_item_ids(self._user_items, user_id)

        rec_ids = self._annoy_index.get_nns_by_vector(
            self._user_embeddings[user_id], 100
        )
        rec_ids = [item for item in rec_ids if item not in viewed_ids][:k_recs]

        while len(rec_ids) < k_recs:
            item = random.choice(self._item_id_map.internal_ids)
            if (
                item not in self._user_items[user_id].indices
                and item not in rec_ids
            ):
                rec_ids.append(item)
        return self._item_id_map.convert_to_external(rec_ids).tolist()

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        if user_id not in self._to_intornal_user_id:
            return self._popular_predicts[:k_recs]
        return self._predict_model(user_id, k_recs)
