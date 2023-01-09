from typing import Callable, Dict, Type

import dill

from service.api.exceptions import RegisterModelError

from .base_model import BaseModel

MODELS: Dict[str, Type[BaseModel]] = {}


def load_dill(path: str):
    with open(path, "rb") as f:
        return dill.load(f)


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    def register(cls: Type[BaseModel]) -> Type[BaseModel]:
        if not issubclass(cls, BaseModel):
            raise RegisterModelError(cls, name)
        MODELS[name] = cls
        return cls

    return register
