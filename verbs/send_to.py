from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff

import openai
from secret import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def chat_completion_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def text_embed_with_backoff(**kwargs):
    return openai.embeddings.create(
        **kwargs,
    )


def send_to_open_ai(messages, **kwargs):
    response = chat_completion_with_backoff(
        **{
            "model": "gpt-3.5-turbo",
            "messages": messages,
        },
        **kwargs,
    )

    output = response.choices[0].message.content
    return output


def send_to_open_ai_text_emb(input, **kwargs):
    embedding = text_embed_with_backoff(
        input=input,
        model="text-embedding-3-small",
        **kwargs,
    )
    return [d.embedding for d in embedding.data]
