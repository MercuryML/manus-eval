import requests
from openai import OpenAI
from typing import Optional, List
import os
import json

class OpenToolCall:
    # url = "https://api.siliconflow.cn/v1/rerank"
    url = "https://api.siliconflow.cn/v1"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("SILICON_TOKEN", token)
        self.model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
        self.client = OpenAI(base_url=self.url, api_key=self.token)

    def ensure_tool_call(self, messages: List[dict], func: dict, **options) -> int:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,  # 启用流式输出
            tools=[func],
            **options,
        )
        score = 0
        if tool_calls := response.choices[0].message.tool_calls:
            # print("tool call: ", tool_calls)
            params = json.loads(tool_calls[0].function.arguments)
            score = sum(params.values())
        if function_call := response.choices[0].message.function_call:
            # print("function call: ", function_call)
            params = json.loads(tool_calls[0].function.arguments)
            score = sum(params.values())
        return score


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    func = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        },
    }
    messages = [
        {"role": "system", "content": "多使用工具回答."},
        {"role": "user", "content": "杭州天气怎么样?"},
    ]
    toolcall = OpenToolCall()
    out = toolcall.ensure_tool_call(messages, func)
    print(out)
