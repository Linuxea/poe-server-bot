"""

Sample bot that wraps OpenRouter API.

"""

from __future__ import annotations

import os
from typing import AsyncIterable

import fastapi_poe as fp
from modal import App, Image, asgi_app
from openai import AsyncOpenAI

# TODO: set your bot access key, openrouter api key, and bot name for this bot to work
# see https://creator.poe.com/docs/quick-start#configuring-the-access-credentials
bot_access_key = os.getenv("POE_ACCESS_KEY")
kimi_api_key = os.getenv("KIMI_API_KEY")
bot_name = "linuxea-poe-server"

# Configure OpenAI client to use OpenRouter
client = AsyncOpenAI(base_url="https://api.moonshot.cn/v1", api_key=kimi_api_key)


async def stream_chat_completion(request: fp.QueryRequest):
    messages = []
    # using all conversation history instead of just the last 5 messages
    for query in request.query:
        if query.role == "system":
            messages.append({"role": "system", "content": query.content})
        elif query.role == "bot":
            messages.append({"role": "assistant", "content": query.content})
        elif query.role == "user":
            messages.append({"role": "user", "content": query.content})
        else:
            raise

    # print all messages
    print("==============================")
    print("Messages for OpenRouter:\n", messages)
    print("==============================")

    response = await client.chat.completions.create(
        model="kimi-k2-0711-preview",  # You can change this to any OpenRouter supported model
        messages=messages,
        temperature=1.0,
        stream=True,
        max_tokens=16000,
        extra_headers={
            # "HTTP-Referer": "https://poe.com",  # Optional: for including your app on openrouter.ai rankings
            "X-Title": "kimi poe wrapper"  # Optional: show in rankings on openrouter.ai
        },
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield fp.PartialResponse(text=chunk.choices[0].delta.content)


class OpenRouterWrapperBot(fp.PoeBot):
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        async for msg in stream_chat_completion(request):
            yield msg


REQUIREMENTS = ["fastapi-poe", "openai"]
image = (
    Image.debian_slim()
    .pip_install(*REQUIREMENTS)
    .env({"POE_ACCESS_KEY": bot_access_key, "KIMI_API_KEY": kimi_api_key})
)
app = App("openrouter-wrapper-bot-poe")


@app.function(image=image)
@asgi_app()
def fastapi_app():
    bot = OpenRouterWrapperBot()
    app = fp.make_app(
        bot,
        access_key=bot_access_key,
        bot_name=bot_name,
        # allow_without_key=not (bot_access_key and bot_name),
        allow_without_key=False,
    )
    return app
