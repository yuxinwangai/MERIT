import json
from pathlib import Path


DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "configs" / "model_settings.json"


def load_settings(path=None):
    return json.loads(Path(path or DEFAULT_SETTINGS).read_text(encoding="utf-8"))


def resolve_model(model, settings=None):
    settings = settings or load_settings()
    if model in settings["models"]:
        return model, settings["models"][model]
    for provider, values in settings["models"].items():
        if values["model"] == model:
            return provider, values
    raise ValueError("Select a model key or identifier present in the model settings")
