# core/llm_client.py
import os
import json
from typing import Type, TypeVar
from pydantic import BaseModel
from litellm import completion

T = TypeVar("T", bound=BaseModel)

class LLMClient:

    def __init__(self, model: str = None, api_key: str = None):
        #M1.1 Model Agnostic completions
        provider = os.environ.get("LLM_PROVIDER", "GEMINI").upper()
        if provider == "OPENAI":
            default_model = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
            default_key = os.environ.get("OPENAI_API_KEY")
        elif provider in ("GOOGLE", "GEMINI"):
            default_model = os.environ.get("GEMINI_LLM_MODEL", "gemini/gemini-2.0-flash")
            default_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

        self.model = model or default_model
        self.api_key = api_key or default_key

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
        print(raw)
        return schema.model_validate_json(raw)
    
    
    def complete_json(self, messages: list[dict], **kwargs) -> str:
        """Returns raw JSON string, no schema validation. Caller validates."""
        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return response.choices[0].message.content