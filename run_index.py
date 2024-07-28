from index.indexer import Indexer
from llm.send import LLMSendToConfig
import argparse
import logging
from llm.create import get_send_fn, get_text_emb_send_fn

# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Index documents", add_help=False)
    parser.add_argument("--output_dir", default="./index_results", type=str, help="Output directory for generated index files.",)
    parser.add_argument("--query", default=None, type=str, help="Query used for retrieving documents.")
    parser.add_argument("--doc_dir", default="./documents", type=str, help="Directory for saving retrieved documents.")
    parser.add_argument("--doc_top_k", default=10, type=int, help="Directory for saving retrieved documents.")
    parser.add_argument("--force", action="store_true", help="Run the index without local results (.parquets)." )
    parser.add_argument("--llm", default="openai", type=str, help="LLM to use: [openai, huggingface]",)
    parser.add_argument("--hf_checkpoint", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Checkpoint to load from huggingface.")
    return parser
# fmt: on


def main(args):

    DEFAULT_LLM_ARGS = {
        "temperature": 0.0,
        "top_p": 1.0,
    }  # This setup for reproducibility.

    logging.basicConfig(
        # filename=f"{args.name}.log",
        level=logging.INFO,
    )

    send_to = get_send_fn(args.llm, args.hf_checkpoint)
    text_emb_send_to = get_text_emb_send_fn(args.llm, args.hf_checkpoint)

    indexer = Indexer(
        doc_top_k=args.doc_top_k,
        final_entities_text_emb_llm_config=LLMSendToConfig(
            llm_send_to=text_emb_send_to,
            llm_model_args={},
        ),
        graph_extractor_llm_config=LLMSendToConfig(
            llm_send_to=send_to,
            llm_model_args=DEFAULT_LLM_ARGS,
        ),
        summarize_extractor_llm_config=LLMSendToConfig(
            llm_send_to=send_to,
            llm_model_args=DEFAULT_LLM_ARGS,
        ),
        community_report_llm_config=LLMSendToConfig(
            llm_send_to=send_to,
            llm_model_args=DEFAULT_LLM_ARGS,
        ),
        output_dir=args.output_dir,
        doc_saving_dir=args.doc_dir,
    )

    """
    Since we retrieved a specific documents for 
    """

    # Check if the query already has the index exist in local.
    indexer.generate(
        query=args.query,
        save=True,
        try_load=not args.force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Indexing script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
