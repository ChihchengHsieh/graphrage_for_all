from .send import Messages, ModelArgs, LLMResponse, ChatLLM, EmbLLM
from transformers import pipeline
import torch

HUGGINGFACE_TOKEN = None

pipe = None

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "ruslanmv/Medical-Llama3-8B"
device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
stop_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token matches the stop token
        return input_ids[0, -1] == self.stop_token_id


def medi_llama_askme(messages):
    # sys_message = """
    # You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
    # provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    # """
    # Create messages structured for the chat template
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": query},
    # ]

    # Applying chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=5000,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([StopOnToken(stop_token_id)]),
    )

    # Extract and return the generated text, removing the prompt
    response_text = tokenizer.batch_decode(outputs)[0].strip()
    answer = response_text.split("<|im_start|>assistant")[-1].strip()
    return answer.split("<|im_end|>")[0].strip()


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
            max_new_tokens=8000,
        )
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
        if not hasattr(pipe.tokenizer, "chat_template"):
            pipe.tokenizer.chat_template = "default"
        if hasattr(pipe.tokenizer, "apply_chat_template"):
            sys_message = """ 
            You are an AI Assistant trained on a vast dataset of information. Please be thorough and
            provide an informative answer. If you don't know the answer to a specific inquiry, advise seeking professional help.
            """
            messages = [{"role": "system", "content": sys_message}]
            pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )


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
        ### Just conduct medi-llama here.
        # output = medi_llama_askme(messages)
        # outttt = medi_llama_askme(
        #     [
        #         {
        #             "role": "system",
        #             "content": "You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.",
        #         },
        #         {
        #             "role": "user",
        #             "content": "How are you?",
        #         },
        #     ]
        # # )
        # return LLMResponse(
        #     output=output,
        #     history=[*messages, {"role": "assistant", "content": output}],
        # )

        if pipe.model.config.name_or_path == "google/gemma-2-2b-it":
            input_messages = [
                {
                    "role": "user",
                    "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
                }
            ] + messages[
                2:
            ]  # Combine the first two messages into user message for gemma, since it doesn't support system message.

        elif pipe.model.config.name_or_path == "ruslanmv/Medical-Llama3-8B":
            input_messages = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_messages = messages

        res = pipe(
            input_messages,
            **model_args,
        )

        if pipe.model.config.name_or_path in [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ]:
            output = res[0]["generated_text"][-1]["content"]

        elif pipe.model.config.name_or_path == "ruslanmv/Medical-Llama3-8B":
            output = res[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()
        else:
            output = res[0]["generated_text"]
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
