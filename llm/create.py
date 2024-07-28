from .openai import send_to_openai, send_to_openai_text_emb
from .huggingface import get_huggingface_send_fn, get_huggingface_text_emb_send_fn


def get_send_fn(llm: str, checkpoint: str):
    match llm:
        case "openai":
            return send_to_openai
        case "huggingface":
            return get_huggingface_send_fn(checkpoint)
        case _:
            raise NotImplementedError(f"LLM: [{llm}] is not implemented.")


def get_text_emb_send_fn(llm: str, checkpoint: str):
    match llm:
        case "openai":
            return send_to_openai_text_emb
        case "huggingface":
            return get_huggingface_text_emb_send_fn(checkpoint)
        case _:
            raise NotImplementedError(f"LLM: [{llm}] is not implemented.")


def get_default_llm_args(llm: str):
    match llm:
        case "openai":
            return 
        case "huggingface":
            return 
        case _:
            raise NotImplementedError(f"LLM: [{llm}] is not implemented.")
