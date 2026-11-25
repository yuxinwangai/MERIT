import argparse
from pathlib import Path

from .generation import QwenImageGenerator
from .settings import load_settings, resolve_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    template = args.prompt.read_text(encoding="utf-8")
    clinical_text = args.text.read_text(encoding="utf-8")
    if "{clinical_situation}" not in template:
        parser.error("The prompt must contain {clinical_situation}")
    prompt = template.replace("{clinical_situation}", clinical_text)
    provider, values = resolve_model(args.model, load_settings(args.settings))
    if provider != "qwen":
        parser.error("Select the local Qwen model")
    generator = QwenImageGenerator(values, args.device)
    result = generator.edit_image(args.composite.read_bytes(), prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)


if __name__ == "__main__":
    main()
