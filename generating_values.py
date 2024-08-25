import argparse
import logging
import secret
import pickle
import os

from query import get_questions_by_lesion
from graphrag_for_all.llm.openai import set_openai_api_key
from graphrag_for_all.llm.huggingface import set_hugging_face_token
from graphrag_for_all.llm.create import get_send_fn
from graphrag_for_all.index.generate import IndexGenerator
from keywords import responses_to_keywords
from graphrag_for_all.search.searcher import Searcher
from dataclasses import dataclass

TOP_5_LESSIONS = [
    # "pulmonary edema",
    "enlarged cardiac silhouette",
    "pulmonary consolidation",
    "atelectasis",
    "pleural abnormality",
]


@dataclass
class ModelSelection:
    source: str
    model: str
    emb_model: str


def get_llm_argument_set(source, model) -> ModelSelection:
    match source:
        case "openai":
            return ModelSelection(
                source=source,
                model=model,
                emb_model="text-embedding-3-small",
            )
        case "huggingface":
            # "huggingface"
            # "meta-llama/Meta-Llama-3.1-8B-Instruct"
            return ModelSelection(
                source=source,
                model=model,
                emb_model="text-embedding-3-small",
            )
        case _:
            raise NotImplementedError(f"Source [{source}] is not supported.")


# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Index documents", add_help=False)
    parser.add_argument("--query", default=None, type=str, help="Query used for retrieving documents.")
    parser.add_argument("--doc_dir", default="./documents", type=str, help="Directory for saving retrieved documents.")
    parser.add_argument("--doc_top_k", default=10, type=int, help="Directory for saving retrieved documents.")
    parser.add_argument("--force", action="store_true", help="Run the index without local results (.parquets).")
    parser.add_argument("--source", default="openai", type=str, help="Source of the LLM: [openai, huggingface]",)
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to load for generation.")
    parser.add_argument("--store_type", default="graphrag", help="Model to load for text embedding. Options: [graphrag,  ]")
    parser.add_argument("--output_dir", default="./index_results", type=str, help="Output directory for generated index files.",)
    return parser
# fmt: on


def main(args):
    logging.basicConfig(
        # filename=f"{args.name}.log",
        level=logging.INFO,
    )
    set_openai_api_key(secret.OPENAI_API_KEY)
    set_hugging_face_token(secret.HUGGINGFACE_TOKEN)

    model_config = get_llm_argument_set(args.source, args.model)

    # load one sending function globally since the memory concern.
    send_fn = get_send_fn(source=model_config.source, model_name=model_config.model)

    response_dict = {}
    keyword_dict = {}
    for lesion in TOP_5_LESSIONS:
        ## adding prior knowledge from 8 questions.
        lesion_questions = get_questions_by_lesion(lesion)

        ## generate the index for specific lesion.
        # TODO: move to argument.
        indexer = IndexGenerator(
            source=model_config.source,
            model_name=model_config.model,
            text_emb_source=model_config.source,
            text_emb_model_name=model_config.emb_model,
            doc_top_k=args.doc_top_k,
            output_dir=args.output_dir,
        )

        indexer.generate(
            store_type=args.store_type,
            query=lesion,
        )

        # Use researcher to ask about the answer:
        # TODO: move all these to arguments
        knowledge_graph_dir = os.path.join(
            f"./{args.output_dir}", args.store_type, f"/{lesion}_top_{args.doc_top_k}"
        )
        searcher = Searcher(
            input_dir=knowledge_graph_dir,
            send_to=send_fn,
            community_level=1,
        )

        responses = []
        for q in lesion_questions:
            result = searcher.search(query=q)
            responses.append(result.response)

        keywords = responses_to_keywords(lesion, responses, send_fn)

        response_dict[lesion] = responses
        keyword_dict[lesion] = keywords

    # dump keywords.
    with open(
        os.path.join(args.output_dir, args.store_type, "extracted_keywords.pkl", "wb")
    ) as f:
        pickle.dump({"responses": response_dict, "keywords": keyword_dict}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generating values for extended keywords", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
