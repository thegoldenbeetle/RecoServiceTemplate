from pydantic import BaseSettings


class Config(BaseSettings):
    class Config:
        case_sensitive = False


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {"env": ["log_level"]},
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    admin_token: str = "correct_token"

    itemknn_model_path: str = "data/bm25_itemknn.dill"
    popular_model_path: str = "data/popular_model.dill"
    kion_dataset_path: str = "data/kion_train"

    offline_model_path: str = "data/offline_bm25_itemknn.dill"

    log_config: LogConfig


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
