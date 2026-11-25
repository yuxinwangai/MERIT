from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .settings import load_settings, resolve_model


class TechnicalGenerationError(ValueError):
    pass

def _looks_like_b64_image(value: str) -> bool:
    return isinstance(value, str) and len(value) > 100 and value.startswith(("iVBOR", "/9j/", "R0lGOD"))


def _find_base64_image(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if value.startswith("data:image/") and "base64," in value:
                    return value.split("base64,", 1)[1]
                if key in {"b64_json", "data"} and _looks_like_b64_image(value):
                    return value
                match = re.search(r"data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=\n\r]+)", value)
                if match:
                    return match.group(1)
            found = _find_base64_image(value)
            if found:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _find_base64_image(item)
            if found:
                return found
    return None


def extract_image_bytes(response: Any) -> bytes | None:
    if hasattr(response, "data") and response.data:
        item = response.data[0]
        b64_json = getattr(item, "b64_json", None)
        if b64_json:
            return base64.b64decode(b64_json)
        url = getattr(item, "url", None)
        if url:
            with urllib.request.urlopen(url, timeout=300) as handle:
                return handle.read()
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if isinstance(data, dict):
        for item in data.get("data", []):
            if item.get("url"):
                with urllib.request.urlopen(item["url"], timeout=300) as handle:
                    return handle.read()
    b64 = _find_base64_image(data)
    return base64.b64decode(b64) if b64 else None



class OpenAIImageGenerator:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        from openai import APIConnectionError, APIStatusError, OpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MERIT_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in the execution environment")
        self.model = model
        self._transport_errors = (APIConnectionError,)
        self._status_errors = (APIStatusError,)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=0,
        )

    def edit_image(self, image_png: bytes, prompt: str) -> bytes:
        image_file = io.BytesIO(image_png)
        image_file.name = "condition.png"
        try:
            response = self.client.images.edit(
                model=self.model,
                image=image_file,
                prompt=prompt,
                size="1024x1024",
            )
        except getattr(self, "_transport_errors", (TimeoutError, ConnectionError)) as error:
            raise TechnicalGenerationError(str(error)) from error
        except self._status_errors as error:
            if error.status_code in (408, 429) or error.status_code >= 500:
                raise TechnicalGenerationError("Transient provider HTTP status " + str(error.status_code)) from error
            raise
        self.last_response_id = getattr(response, "_request_id", None) or getattr(response, "id", None)
        self.last_response_model = getattr(response, "model", None)
        try:
            image = extract_image_bytes(response)
        except (binascii.Error, ValueError) as error:
            raise TechnicalGenerationError("Provider image encoding was invalid") from error
        if image is None:
            raise TechnicalGenerationError("Model response did not contain an image")
        return image


class JsonImageGenerator:
    def __init__(self, model, provider, transport=None):
        self.model = model
        self.provider = provider
        self.transport = transport or urllib.request.urlopen
        self.last_response_id = None
        self.last_response_model = None

    def edit_image(self, image_png, prompt):
        encoded = base64.b64encode(image_png).decode("ascii")
        if self.provider == "grok":
            key = os.environ.get("XAI_API_KEY")
            if not key:
                raise ValueError("Set XAI_API_KEY in the execution environment")
            url = "https://api.x.ai/v1/images/edits"
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            payload = {"model": self.model, "prompt": prompt,
                       "image": {"url": "data:image/png;base64," + encoded, "type": "image_url"},
                       "n": 1, "response_format": "b64_json", "aspect_ratio": "1:1", "resolution": "1k"}
        elif self.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY in the execution environment")
            url = "https://generativelanguage.googleapis.com/v1beta/models/" + self.model + ":generateContent"
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
            payload = {"contents": [{"role": "user", "parts": [
                {"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": encoded}}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                     "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"}}}
        else:
            raise ValueError("Unsupported JSON image provider")
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with self.transport(request, timeout=300) as handle:
                data = json.loads(handle.read())
                self.last_response_id = data.get("responseId") or handle.headers.get("x-request-id")
        except urllib.error.HTTPError as error:
            if error.code in (408, 429) or error.code >= 500:
                raise TechnicalGenerationError("Transient provider HTTP status " + str(error.code)) from error
            raise ValueError("Provider rejected the request with HTTP status " + str(error.code)) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TechnicalGenerationError("Provider transport failed") from error
        except (ValueError, UnicodeError) as error:
            raise TechnicalGenerationError("Provider response was not valid JSON") from error
        self.last_response_model = data.get("model") or data.get("modelVersion")
        try:
            if self.provider == "gemini":
                parts = [part for candidate in data.get("candidates", [])
                         for part in candidate.get("content", {}).get("parts", []) if not part.get("thought")]
                inline = [part.get("inlineData", part.get("inline_data", {})) for part in parts]
                image_data = next((part.get("data") for part in inline
                                   if part.get("mimeType", part.get("mime_type", "")).startswith("image/")), None)
                image = base64.b64decode(image_data) if image_data else None
            else:
                image = extract_image_bytes(data)
        except (binascii.Error, ValueError) as error:
            raise TechnicalGenerationError("Provider image encoding was invalid") from error
        if image is None:
            raise TechnicalGenerationError("Model response did not contain an image")
        return image


class QwenImageGenerator:
    def __init__(self, settings, device="cuda", pipeline=None):
        self.settings = settings
        self.model = settings["model"]
        self.device = device
        self.pipeline = pipeline
        self.last_response_id = None
        self.last_response_model = self.model

    def edit_image(self, image_png, prompt):
        import torch
        from PIL import Image

        if self.pipeline is None:
            from diffusers import QwenImageEditPlusPipeline
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(self.model, torch_dtype=torch.bfloat16)
            self.pipeline.to(self.device)
        image = Image.open(io.BytesIO(image_png)).convert("RGB")
        values = self.settings
        generator = torch.Generator(device="cpu").manual_seed(values["seed"])
        result = self.pipeline(image=[image], prompt=prompt, negative_prompt=values["negative_prompt"],
                               generator=generator, true_cfg_scale=values["true_cfg_scale"],
                               guidance_scale=values["guidance_scale"],
                               num_inference_steps=values["num_inference_steps"],
                               num_images_per_prompt=values["num_images_per_prompt"], width=1024, height=1024)
        stream = io.BytesIO()
        result.images[0].save(stream, format="PNG")
        return stream.getvalue()


def create_generator(model, settings_path=None, device="cuda"):
    provider, values = resolve_model(model, load_settings(settings_path))
    if provider == "gpt":
        return OpenAIImageGenerator(values["model"])
    if provider in ("gemini", "grok"):
        return JsonImageGenerator(values["model"], provider)
    if provider == "qwen":
        return QwenImageGenerator(values, device=device)
    raise ValueError("Unsupported provider")


def ImageEditGenerator(model, base_url=None):
    if base_url:
        raise ValueError("Provider routes are selected from the locked model configuration")
    return create_generator(model)

