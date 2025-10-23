"""LLM client for vLLM OpenAI-compatible server."""

import json
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from core.config import RoleParams, llm_config


class LLMClient:
    """Client for interacting with vLLM server via OpenAI-compatible API."""

    def __init__(self):
        """Initialize the client."""
        self.base_url = llm_config.base_url
        self.api_key = llm_config.api_key or "EMPTY"
        self.model_name = None  # Will be fetched
        self.max_context_tokens = llm_config.max_context_tokens
        self._model_fetched = False

    async def _fetch_model(self):
        """Fetch available model from the server."""
        if self._model_fetched:
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                }
                response = await client.get(f"{self.base_url}/models", headers=headers)
                response.raise_for_status()

                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    self.model_name = data["data"][0]["id"]
                    print(f"Auto-detected model: {self.model_name}")
                else:
                    # Fallback to config if available
                    self.model_name = llm_config.model_name or "unknown"
                    print(f"No models found, using: {self.model_name}")

                self._model_fetched = True
        except Exception as e:
            print(f"Warning: Could not fetch models from server: {e}")
            # Fallback to config
            self.model_name = llm_config.model_name or "unknown"
            self._model_fetched = True

    async def _ensure_model(self):
        """Ensure model name is available."""
        if not self._model_fetched:
            await self._fetch_model()

    async def complete(
        self,
        messages: list[dict[str, str]],
        params: RoleParams,
        response_format: Optional[type[BaseModel]] = None,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            messages: Chat messages in OpenAI format
            params: Role-specific parameters (temperature, etc.)
            response_format: Optional Pydantic model for structured output

        Returns:
            The generated text response
        """
        import re

        await self._ensure_model()

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
            }

            # Add response format for JSON mode
            if params.json_mode and response_format:
                payload["response_format"] = {"type": "json_object"}
                # Add JSON schema hint to system message
                if messages and messages[0]["role"] == "system":
                    schema_hint = f"\n\nYou must respond with valid JSON matching this schema:\n{response_format.model_json_schema()}"
                    messages[0]["content"] += schema_hint

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Strip <think> tags (common in some models for chain-of-thought)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()

            return content

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        params: RoleParams,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """
        Generate a structured JSON response.

        Args:
            messages: Chat messages in OpenAI format
            params: Role-specific parameters
            response_model: Pydantic model to parse response into

        Returns:
            Parsed instance of response_model
        """
        import re

        content = await self.complete(messages, params, response_format=response_model)

        # Strip any <think> tags and their content (common in some models)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content or "```" in content:
            # Extract content between code fences
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        elif content.startswith("```"):
            # Simple case: starts with code fence
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        # Try to extract JSON object if embedded in text
        if not content.startswith("{") and not content.startswith("["):
            # Look for JSON object in the text
            json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        # Parse JSON
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content[:500]}...")

    async def complete_batch(
        self,
        requests: list[tuple[list[dict[str, str]], RoleParams, Optional[type[BaseModel]]]],
    ) -> list[str]:
        """
        Process multiple completion requests concurrently.

        Args:
            requests: List of (messages, params, response_format) tuples

        Returns:
            List of generated responses
        """
        import asyncio

        tasks = [self.complete(messages, params, response_format) for messages, params, response_format in requests]

        return await asyncio.gather(*tasks)


# Global client instance
llm_client = LLMClient()
