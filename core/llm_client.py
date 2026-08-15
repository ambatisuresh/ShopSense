# core/llm_client.py
import os
import json
from typing import Type, TypeVar
from pydantic import BaseModel
from litellm import completion

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.environ.get("DEFAULT_LLM_MODEL", "gemini/gemini-2.0-flash")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    def complete(self, messages: list[dict], **kwargs) -> str:
        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            **kwargs,
        )
        return response.choices[0].message.content

    def complete_structured(self, messages: list[dict], schema: Type[T], **kwargs) -> T:
        """Force structured output validated against a Pydantic schema."""
        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            response_format=schema,   # LiteLLM passes this through to providers that support it
            **kwargs,
        )
        raw = response.choices[0].message.content
        return schema.model_validate_json(raw)