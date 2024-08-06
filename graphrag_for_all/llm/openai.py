from typing import List
from .send import ChatLLM, Messages, ModelArgs, LLMResponse
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

def set_openai_api_key(api_key):
    openai.api_key = api_key


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


def get_openai_send_fn(
    model_name: str = "gpt-3.5-turbo",
) -> ChatLLM:

    def send_to(messages: Messages, model_args: ModelArgs) -> LLMResponse:
        response = chat_completion_with_backoff(
            **{
                "model": model_name,
                "messages": messages,
            },
            **model_args,
        )

        output = response.choices[0].message.content

        return LLMResponse(
            output=output,
            history=[*messages, {"role": "assistant", "content": output}],
        )

    return send_to


def get_openai_text_emb_send_fn(
    model_name: str = "text-embedding-3-small",
) -> ChatLLM:

    def send_to(input: List[str], model_args: ModelArgs) -> LLMResponse:
        embedding = text_embed_with_backoff(
            input=input,
            model=model_name,
            **model_args,
        )
        return [d.embedding for d in embedding.data]

    return send_to


def send_to_openai(messages: Messages, model_args: ModelArgs) -> LLMResponse:
    response = chat_completion_with_backoff(
        **{
            "model": "gpt-3.5-turbo",
            "messages": messages,
        },
        **model_args,
    )

    output = response.choices[0].message.content

    return LLMResponse(
        output=output,
        history=[*messages, {"role": "assistant", "content": output}],
    )


def send_to_openai_text_emb(
    input: List[str], model_args: ModelArgs = {}
) -> List[List[float]]:
    embedding = text_embed_with_backoff(
        input=input,
        model="text-embedding-3-small",
        **model_args,
    )
    return [d.embedding for d in embedding.data]
