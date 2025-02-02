import os
from typing import Annotated, Any

from pydantic import AnyUrl, BeforeValidator, computed_field
from pydantic_settings import BaseSettings


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    BACKEND_CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []
    PORT: int = 9090
    local_path: str
    model_filename: str
    preprocessor_filename: str

    @computed_field
    @property
    def MODEL(self) -> str:
        file_path = os.path.join(self.local_path, self.model_filename)
        if os.path.exists(file_path):
            return file_path
        raise ValueError("model file path does not exist.")

    @computed_field
    @property
    def PREPROCESSOR(self) -> str:
        file_path = os.path.join(self.local_path, self.preprocessor_filename)
        if os.path.exists(file_path):
            return file_path
        raise ValueError("Preprocessor file path does not exist.")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
