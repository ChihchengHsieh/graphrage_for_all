from .send import Messages, ModelArgs, LLMResponse, ChatLLM, EmbLLM
from transformers import pipeline
import torch

HUGGINGFACE_TOKEN = None

pipe = None


def set_hugging_face_token(token):
    global HUGGINGFACE_TOKEN
    HUGGINGFACE_TOKEN = token


def init_pipe(checkpoint):
    global HUGGINGFACE_TOKEN
    if HUGGINGFACE_TOKEN is None:
        raise ValueError("Set up HUGGINGFACE_TOKEN before initialisation.")

    global pipe
    if pipe is None:
        pipe = pipeline(
            "text-generation",
            checkpoint,
            token=HUGGINGFACE_TOKEN,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=12_000,
        )
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id


def parse_to_huggingface_args(args: ModelArgs):
    output = {}
    if "max_tokens" in args:
        output["max_new_tokens"] = args["max_tokens"]

    if "temperature" in args:
        if args["temperature"] == 0:
            output["do_sample"] = False
            output["temperature"] = None
            output["top_p"] = 0
        else:
            output["temperature"] = args["temperature"]

    return output


def get_huggingface_send_fn(
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> ChatLLM:
    init_pipe(checkpoint)
    def send_to(messages: Messages, model_args: ModelArgs) -> LLMResponse:
        global pipe
        model_args = parse_to_huggingface_args(model_args)

        res = pipe(
            messages,
            **model_args,
        )

        output = res[0]["generated_text"][-1]["content"]
        return LLMResponse(
            output=output,
            history=[*messages, {"role": "assistant", "content": output}],
        )

    return send_to


def get_huggingface_text_emb_send_fn(
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> EmbLLM:
    init_pipe(checkpoint)

    global pipe
    model = pipe.model
    tokenizer = pipe.tokenizer

    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=HUGGINGFACE_TOKEN)
    # model = AutoModel.from_pretrained(checkpoint, token=HUGGINGFACE_TOKEN)

    def text_emnb_send_to(input: list[str], model_args: ModelArgs) -> list[list[float]]:
        inputs = tokenizer(input, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                # max_new_tokens=0,
            )
            embeddings = (
                outputs.hidden_states[-1].mean(dim=1).tolist()
            )  # (B, L, D).mean(dim=1) => (B, D)
        return embeddings

    return text_emnb_send_to
