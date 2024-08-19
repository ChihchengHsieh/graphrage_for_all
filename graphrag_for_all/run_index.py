from index.graphrag import GraphRAGIndexer
from index.vectorstore import VectorStoreIndexer
from graphrag_for_all.llm.send import LLMSendToConfig
import argparse
import logging
import os
from graphrag_for_all.llm.create import get_send_fn, get_text_emb_send_fn

# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Index documents", add_help=False)
    parser.add_argument("--output_dir", default="./index_results", type=str, help="Output directory for generated index files.",)
    parser.add_argument("--query", default=None, type=str, help="Query used for retrieving documents.")
    parser.add_argument("--doc_dir", default="./documents", type=str, help="Directory for saving retrieved documents.")
    parser.add_argument("--doc_top_k", default=10, type=int, help="Directory for saving retrieved documents.")
    parser.add_argument("--force", action="store_true", help="Run the index without local results (.parquets)." )
    parser.add_argument("--source", default="openai", type=str, help="Source of the LLM: [openai, huggingface]",)
    parser.add_argument("--text_emb_source", default="openai", type=str, help="Source of the text embedding LLM: [openai, huggingface]",)
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model to load for generation.")
    parser.add_argument("--text_emb_model_name", default="text-embedding-3-small", help="Model to load for text embedding.")
    parser.add_argument("--store_type", default="graphrag", help="Model to load for text embedding. Options: [graphrag,  ]")
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

    send_to = get_send_fn(args.source, args.model_name)
    text_emb_send_to = get_text_emb_send_fn(
        args.text_emb_source, args.text_emb_model_name
    )

    if args.store_type == "graphrag":
        indexer = GraphRAGIndexer(
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
            output_dir=os.path.join(args.output_dir, "graphrag"),
            doc_saving_dir=args.doc_dir,
        )

        indexer.generate(
            query=args.query,
            save=True,
            try_load=not args.force,
        )
    elif args.store_type == "vectorstore":
        indexer = VectorStoreIndexer(
            emb_llm_config=LLMSendToConfig(
                llm_send_to=text_emb_send_to,
                llm_model_args={},
            ),
            output_dir=os.path.join(args.output_dir, "vectorstore"),
            doc_saving_dir=args.doc_dir,
            doc_top_k=args.doc_top_k,
        )
        query_output_dir = indexer.generate(
            query=args.query,
            save=True,
        )

        indexer.save_emb_llm_info(
            query_output_dir=query_output_dir,
            source=args.text_emb_source,
            model_name=args.text_emb_model_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Indexing script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
