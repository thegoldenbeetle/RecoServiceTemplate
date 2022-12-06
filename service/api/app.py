import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import uvloop
from fastapi import FastAPI
from rectools import Columns

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .models import BaseModel, ItemKNNModel, OfflineItemKNNModel
from .views import add_views

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def load_dataset(
    config: ServiceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = Path(config.kion_dataset_path)
    interactions = pd.read_csv(dataset_path / "interactions.csv")
    users = pd.read_csv(dataset_path / "users.csv")
    items = pd.read_csv(dataset_path / "items.csv")

    # rename columns, convert timestamp
    interactions.rename(
        columns={
            "last_watch_dt": Columns.Datetime,
            "total_dur": Columns.Weight,
        },
        inplace=True,
    )

    interactions["datetime"] = pd.to_datetime(interactions["datetime"])

    return interactions, users, items


def init_models(config: ServiceConfig) -> Dict[str, BaseModel]:
    interactions, _, _ = load_dataset(config)
    return {
        "itemknn_model": ItemKNNModel(
            config.itemknn_model_path,
            config.popular_model_path,
            interactions,
        ),
        "offline_itemknn_model": OfflineItemKNNModel(
            config.offline_itemknn_path,
            config.offline_popular_path,
        ),
    }


def create_app(config: ServiceConfig, preload_model: bool = True) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)

    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs
    app.state.admin_token = config.admin_token
    app.state.models = init_models(config) if preload_model else {}

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
