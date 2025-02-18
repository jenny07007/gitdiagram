from openai import OpenAI
from dotenv import load_dotenv
from app.utils.format_message import format_user_message
from app.services.base_llm_service import BaseLLMService
import tiktoken
import os
import aiohttp
import json
from typing import Literal, AsyncGenerator

load_dotenv()


class OpenAIO3Service(BaseLLMService):
    def __init__(self):
        self.default_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # https://github.com/openai/tiktoken/blob/e35ab0915e37b919946b70947f1d0854196cb72c/tiktoken/model.py#L8
        self.encoding = tiktoken.get_encoding("o200k_base")
        # This is correct for OpenAI API
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def call_api(
        self,
        system_prompt: str,
        data: dict,
        api_key: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
    ) -> str:
        user_message = format_user_message(data)
        self._log_request(system_prompt, user_message,
                          api_key, reasoning_effort)

        client = OpenAI(api_key=api_key) if api_key else self.default_client

        try:
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=12000,
                reasoning_effort=reasoning_effort,
            )

            if completion.choices[0].message.content is None:
                self._log_error("No content returned from o3-mini")
                raise ValueError("No content returned from o3-mini")

            response_content = completion.choices[0].message.content
            self._log_response(response_content)
            return response_content

        except Exception as e:
            self._log_error(e)
            raise

    async def call_api_stream(
        self,
        system_prompt: str,
        data: dict,
        api_key: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
    ) -> AsyncGenerator[str, None]:
        user_message = format_user_message(data)
        self._log_stream_request(
            system_prompt, user_message, api_key, reasoning_effort)

        headers = {
            "Authorization": f"Bearer {api_key or self.default_client.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "o3-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": 12000,
            "stream": True,
            "reasoning_effort": reasoning_effort,
        }

        try:
            async with aiohttp.ClientSession() as session:
                print(f"Making request to: {self.base_url}")
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        self._log_stream_error(response.status, error_text)
                        raise Exception(
                            f"API Error: {response.status} - {error_text}")

                    print("\n=== O3-Mini Stream Started ===")
                    buffer = ""
                    chunk_count = 0
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                self._log_stream_complete(chunk_count, buffer)
                                break
                            try:
                                data = json.loads(line[6:])
                                if (content := data.get("choices", [{}])[0]
                                    .get("delta", {})
                                        .get("content")):
                                    buffer += content
                                    chunk_count += 1
                                    self._log_chunk_progress(chunk_count)
                                    yield content
                            except json.JSONDecodeError as e:
                                self._log_chunk_error(e, line)
                                continue

        except Exception as e:
            self._log_stream_error(None, str(e))
            raise

    def count_tokens(self, prompt: str) -> int:
        return len(self.encoding.encode(prompt))

    # Helper methods for logging
    def _log_request(self, system_prompt: str, user_message: str, api_key: str | None, reasoning_effort: str):
        print("\n=== O3-Mini API Request ===")
        print(f"System prompt length: {len(system_prompt)} chars")
        print(f"User message length: {len(user_message)} chars")
        print(
            f"Token count: {self.count_tokens(system_prompt + user_message)}")
        print(
            f"Using API key: {api_key[:8]}... (custom)" if api_key else "Using default API key")
        print(f"Reasoning effort: {reasoning_effort}")
        print("=========================\n")

    def _log_response(self, response_content: str):
        print("\n=== O3-Mini API Response ===")
        print(f"Response length: {len(response_content)} chars")
        print(f"First 200 chars: {response_content[:200]}...")
        print("==========================\n")

    def _log_error(self, error):
        print(f"\n=== O3-Mini API Error ===")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {str(error)}")
        print("=======================\n")

    def _log_stream_request(self, system_prompt: str, user_message: str, api_key: str | None, reasoning_effort: str):
        print("\n=== O3-Mini Streaming Request ===")
        print(f"System prompt length: {len(system_prompt)} chars")
        print(f"User message length: {len(user_message)} chars")
        print(
            f"Token count: {self.count_tokens(system_prompt + user_message)}")
        print(
            f"Using API key: {api_key[:8]}... (custom)" if api_key else "Using default API key")
        print(f"Reasoning effort: {reasoning_effort}")
        print("==============================\n")

    def _log_stream_error(self, status: int | None, error_text: str):
        print(f"\n=== O3-Mini Stream Error ===")
        if status:
            print(f"Status code: {status}")
        print(f"Error response: {error_text}")
        print("=========================\n")

    def _log_stream_complete(self, chunk_count: int, buffer: str):
        print("\n=== O3-Mini Stream Complete ===")
        print(f"Total chunks received: {chunk_count}")
        print(f"Total response length: {len(buffer)} chars")
        if not buffer:
            print("Warning: No content received!")
        print("============================\n")

    def _log_chunk_progress(self, chunk_count: int):
        if chunk_count % 100 == 0:
            print(f"Received {chunk_count} chunks...")

    def _log_chunk_error(self, error: Exception, line: str):
        print(f"Error processing chunk: {error}")
        print(f"Problematic line: {line}")
