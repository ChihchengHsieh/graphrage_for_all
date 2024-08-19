from .global_search import GlobalSearch
import pandas as pd
import tiktoken
from .community_context import GlobalCommunityContext
from .read import read_indexer_entities, read_indexer_reports
from graphrag_for_all.llm.send import ChatLLM

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
COMMUNITY_LEVEL = 2
DEFAULT_CONTEXT_BUILDER_PARAMS = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}
DEFAULT_MAP_LLM_PARAMS = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}
DEFAULT_REDUCE_LLM_PARAMS = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}


class Searcher:
    def __init__(
        self,
        input_dir: str,
        send_to: ChatLLM,
        community_report_table=COMMUNITY_REPORT_TABLE,
        entity_table=ENTITY_TABLE,
        entity_embedding_table=ENTITY_EMBEDDING_TABLE,
        community_level=COMMUNITY_LEVEL,
        token_encoder=tiktoken.get_encoding("cl100k_base"),
        context_builder_params=DEFAULT_CONTEXT_BUILDER_PARAMS,
        map_llm_params=DEFAULT_MAP_LLM_PARAMS,
        reduce_llm_params=DEFAULT_REDUCE_LLM_PARAMS,
    ) -> None:
        # self.community_report_table = community_report_table
        # self.entity_table = entity_table
        # self.entity_embedding_table = entity_embedding_table
        # self.community_level = community_level
        # self.token_encoder = token_encoder
        # self.context_builder_params = context_builder_params
        # self.map_llm_params = map_llm_params
        # self.reduce_llm_params = reduce_llm_params
        # self.send_to = send_to

        entity_df = pd.read_parquet(f"{input_dir}/{entity_table}.parquet")
        report_df = pd.read_parquet(f"{input_dir}/{community_report_table}.parquet")
        entity_embedding_df = pd.read_parquet(
            f"{input_dir}/{entity_embedding_table}.parquet"
        )
        reports = read_indexer_reports(report_df, entity_df, community_level)
        entities = read_indexer_entities(
            entity_df, entity_embedding_df, community_level
        )
        print(f"Report records: {len(report_df)}")
        print(report_df.head())

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,  # default to None if you don't want to use community weights for ranking
            token_encoder=token_encoder,
        )

        self.search_engine = GlobalSearch(
            send_to=send_to,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            # concurrent_coroutines=32,
            response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

    def search(self, query: str):
        return self.search_engine.search(query)
