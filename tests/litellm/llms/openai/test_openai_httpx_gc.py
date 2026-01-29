import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.openai import OpenAIChatCompletion


def _has_closed_client_error(exc: Exception) -> bool:
    if "client has been closed" in str(exc).lower():
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and "client has been closed" in str(cause).lower():
        return True
    return False


def test_should_reproduce_closed_httpx_client_under_concurrency(monkeypatch):
    monkeypatch.setenv("LITELLM_DEBUG_FORCE_HTTPX_CLIENT_GC", "true")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = {}
        if request.content:
            try:
                body = json.loads(request.content.decode())
            except Exception:
                body = {}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def run_test():
        transport = httpx.MockTransport(handler)
        openai_chat = OpenAIChatCompletion()
        timeout = httpx.Timeout(5.0)
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
        }
        logging_obj = MagicMock()
        openai_client = None

        async def run_one():
            return await openai_chat.make_openai_chat_completion_request(
                openai_aclient=openai_client,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )

        try:
            with patch.object(
                AsyncHTTPHandler, "_create_async_transport", return_value=transport
            ):
                openai_client = openai_chat._get_openai_client(
                    is_async=True,
                    api_key="test-key",
                    api_base="https://mock.local/v1",
                    timeout=timeout,
                )
                assert openai_client is not None

                tasks = [run_one() for _ in range(30)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if hasattr(litellm, "in_memory_llm_clients_cache"):
                litellm.in_memory_llm_clients_cache.flush_cache()

        errors = [result for result in results if isinstance(result, Exception)]
        assert errors, "Expected at least one failure due to a closed HTTPX client."
        assert any(_has_closed_client_error(error) for error in errors)

    asyncio.run(run_test())
