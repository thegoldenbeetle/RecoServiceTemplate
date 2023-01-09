# pylint: disable=too-many-instance-attributes
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from pandas import Series
from rectools.dataset import Dataset, IdMap
from rectools.models import ImplicitItemKNNWrapperModel, PopularModel
from scipy import sparse

from .base_model import BaseModel
from .common import load_dill, register_model


@register_model("ranking")
class Ranking(BaseModel):
    cols = [
        "itemknn_score",
        "itemknn_rank",
        "age",
        "income",
        "sex",
        "kids_flg",
        "user_hist",
        "user_avg_pop",
        "user_last_pop",
        "content_type",
        "release_year",
        "for_kids",
        "age_rating",
        "studios",
        "item_pop",
        "item_avg_hist",
    ]
    cat_cols = [
        "age",
        "income",
        "sex",
        "kids_flg",
        "content_type",
        "for_kids",
        "studios",
    ]

    def __init__(
        self,
        model_path: str,
        popular_model_path: str,
        ranker_model_path: str,
        data_path: str,
    ):
        dataset = load_dill(Path(data_path) / "base_model_dataset.dill")

        self._users = load_dill(Path(data_path) / "users.dill")
        self._users.index = self._users["user_id"].tolist()

        self._items = load_dill(Path(data_path) / "items.dill")
        self._items.index = self._items["item_id"].tolist()

        self._default_values_users = load_dill(
            Path(data_path) / "default_values_users.dill"
        )
        self._default_values_items = load_dill(
            Path(data_path) / "default_values_items.dill"
        )
        self._interactions_default_values = load_dill(
            Path(data_path) / "interactions_default_values.dill"
        )

        self._model: ImplicitItemKNNWrapperModel = load_dill(model_path)
        self._popular_model: PopularModel = load_dill(popular_model_path)
        self._ranker_model: LGBMRanker = load_dill(ranker_model_path)
        self._user_external_ids: pd.Series = dataset.user_id_map.external_ids
        self._user_items: sparse.csr_matrix = dataset.get_user_item_matrix(
            include_weights=True
        )
        self._item_id_map: IdMap = dataset.item_id_map
        self._to_intornal_user_id: Series = dataset.user_id_map.to_internal
        self._to_intornal_item_id: Series = self._item_id_map.to_internal
        self._popular_predicts: Tuple[
            Sequence[int], Sequence[float], Sequence[int]
        ] = self._predict_popular_model(dataset, k_recs=100)

    @staticmethod
    def _encode_cat_cols(
        df: pd.DataFrame, cat_cols
    ) -> Tuple[pd.DataFrame, Dict]:
        cat_col_encoding = {}  # словарь с категориями

        # Тут мы могли бы заполнять пропуски как еще одну категорию,
        # но они и так заполняются таким образом автоматически ниже
        # default_values = {col: 'None' for col in cat_cols}
        # df.fillna(default_values, inplace=True)

        for col in cat_cols:
            cat_col = df[col].astype("category").cat
            cat_col_encoding[col] = cat_col.categories
            df[col] = cat_col.codes.astype("category")
        return df, cat_col_encoding

    def _first_stage(
        self, user_id: int, k_recs: int = 100
    ) -> Tuple[Sequence[int], Sequence[float], Sequence[int]]:
        if user_id not in self._user_external_ids:
            return (
                self._popular_predicts[0][:k_recs],
                self._popular_predicts[1][:k_recs],
                self._popular_predicts[2][:k_recs],
            )
        return self._predict_model(user_id, k_recs)

    def _predict_model(
        self,
        user_id: int,
        k_recs: int = 100,
    ) -> Tuple[Sequence[int], Sequence[float], Sequence[int]]:
        user_id = self._to_intornal_user_id[user_id]
        # pylint: disable=protected-access
        rec_ids, scores = self._model._recommend_for_user(
            user_id,
            self._user_items,
            k_recs,
            filter_viewed=True,
            sorted_item_ids=None,
        )
        ranks = np.arange(len(rec_ids)) + 1.0
        return (
            self._item_id_map.convert_to_external(rec_ids),
            scores,
            ranks.tolist(),
        )

    def _predict_popular_model(
        self,
        dataset: Dataset,
        k_recs: int = 100,
    ) -> Tuple[Sequence[int], Sequence[float], Sequence[int]]:
        predict = self._popular_model.recommend(
            [dataset.user_id_map.external_ids[0]],
            dataset=dataset,
            k=k_recs,
            filter_viewed=False,
        )
        return (
            predict["item_id"],
            predict["score"],
            predict["rank"],
        )

    def _prepare_data_to_second_stage(
        self, user_id: int, items: Sequence[int], data: pd.DataFrame
    ) -> pd.DataFrame:
        default_values = {
            "itemknn_score": -0.01,
            "itemknn_rank": 101,
            **self._interactions_default_values,
        }
        data.fillna(default_values, inplace=True)

        if user_id in self._users.index:
            data = data.merge(
                self._users.loc[[user_id], :],
                how="left",
                on=["user_id"],
                copy=False,
            )
        else:
            data[["user_hist", "user_avg_pop", "user_last_pop"]] = 0.0
            data[["age", "income", "sex", "kids_flg"]] = None
            data[["age", "income", "sex", "kids_flg"]] = data[
                ["age", "income", "sex", "kids_flg"]
            ].astype("category")
        data = data.merge(
            self._items.loc[items, :], how="left", on=["item_id"], copy=False
        )

        data.fillna(self._default_values_items, inplace=True)
        data.fillna(self._default_values_users, inplace=True)

        for col in Ranking.cat_cols:
            if -1 not in data[col].cat.categories:
                data[col] = data[col].cat.add_categories(-1)
            data[col].fillna(-1, inplace=True)

        data.sort_values(
            by=["item_id"],
            inplace=True,
        )
        return data

    def predict(self, user_id: int, k_recs: int = 100) -> List[int]:
        items, scores, rank = self._first_stage(user_id, k_recs=100)
        data = pd.DataFrame(
            {
                "user_id": [user_id] * len(items),
                "item_id": items,
                "itemknn_score": scores,
                "itemknn_rank": rank,
            }
        )
        data = self._prepare_data_to_second_stage(user_id, items, data)
        data[Ranking.cat_cols] = data[Ranking.cat_cols].astype(int)

        score = self._ranker_model.predict(data[Ranking.cols])
        data["score"] = score

        # Hybrid score
        mask = (data["itemknn_rank"] < 101).to_numpy()
        eps: float = 0.001
        min_score: float = min(score) - eps
        data["hybrid_score"] = data["score"] * mask
        data["hybrid_score"].replace(
            0,
            min_score,
            inplace=True,
        )
        data.sort_values(
            by=["hybrid_score"],
            ascending=[False],
            inplace=True,
        )
        recs = data["item_id"].iloc[:k_recs].values.tolist()
        while len(recs) < k_recs:
            item = random.choice(self._item_id_map.external_ids)
            if (
                self._to_intornal_item_id[item]
                not in self._user_items[user_id].indices
                and item not in recs
            ):
                recs.append(item)
        return recs
