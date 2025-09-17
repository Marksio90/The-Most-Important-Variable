
from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    sample_data_path: str = str(Path("data") / "avocado.csv")

def get_settings() -> Settings:
    return Settings()
