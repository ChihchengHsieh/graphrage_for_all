from .openai import get_openai_send_fn, send_to_openai, send_to_openai_text_emb
from .huggingface import get_huggingface_send_fn, get_huggingface_text_emb_send_fn


def get_send_fn(source: str, model_name: str):
    match source:
        case "openai":
            return get_openai_send_fn(model_name)
        case "huggingface":
            return get_huggingface_send_fn(model_name)
        case _:
            raise NotImplementedError(f"LLM: [{source}] is not implemented.")


def get_text_emb_send_fn(source: str, model_name: str):
    match source:
        case "openai":
            return send_to_openai_text_emb(model_name)
        case "huggingface":
            return get_huggingface_text_emb_send_fn(model_name)
        case _:
            raise NotImplementedError(f"LLM: [{source}] is not implemented.")
