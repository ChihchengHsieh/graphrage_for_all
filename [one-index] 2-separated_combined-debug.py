import pandas as pd
import secret
import json
import os
import pickle
import re

from graphrag_for_all.llm.openai import set_openai_api_key
from graphrag_for_all.llm.huggingface import set_hugging_face_token
from graphrag_for_all.llm.create import get_send_fn
from utils.query import get_questions_by_lesion
from graphrag_for_all.search.searcher import Searcher
from collections import OrderedDict

MIMIC_EYE_PATH = "F:\\mimic-eye"
REFLACX_LESION_LABEL_COLS = [
    "Pulmonary edema",
    "Enlarged cardiac silhouette",
    "Consolidation",
    "Atelectasis",
    "Pleural abnormality",
]
DEFAULT_LLM_ARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
}

# This combined keywords does not work well, so we use separated keywords instead.
# with open(
#     "./combined_index_results/graphrag/index_graphrag_llama3v1_combined_top_1/combined_extracted_keywords.pkl",
#     "rb",
# ) as f:
#     combined_extracted_keywords = pickle.load(f)

# searcher = Searcher(
#     input_dir="./combined_index_results/graphrag/index_graphrag_llama3v1_combined_top_1/",
#     send_to=send_fn,
#     community_level=1,
# )

def main():
    set_openai_api_key(secret.OPENAI_API_KEY)
    set_hugging_face_token(secret.HUGGINGFACE_TOKEN)
    send_fn = get_send_fn(
        source="huggingface", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    with open(
        "./combined_index_results/graphrag/index_graphrag_llama3v1_combined_top_1/separated_extracted_keywords.pkl",
        "rb",
    ) as f:
        separated_extracted_keywords = pickle.load(f)

    extracted_keywords_results = separated_extracted_keywords


    def build_contextual_inputs(extracted_keywords_results):
        lesion_qa_pairs = extracted_keywords_results["responses"]

        prior_knowledge = OrderedDict({})
        for lesion, q_a in lesion_qa_pairs.items():
            q_a_section = ""
            for q, a in q_a.items():
                q_a_section += f"\n#############################################\n**Question**: {q}\n**Answer**:\n{a}\n"
            lesion_content = f"## Lesion: {lesion}\n" + q_a_section
            prior_knowledge[lesion] = lesion_content
        return prior_knowledge

    prior_knowledge = build_contextual_inputs(extracted_keywords_results)
    all_prior_knowledge = "\n\n\n".join(prior_knowledge.values())

    requesting_prompt = f""" The following is the information from {len(prior_knowledge)} lesions, including {", ".join(list(prior_knowledge.keys()))}. Please combine and summarize them.

    {all_prior_knowledge}

    (Please return the summarized version directly, without additional text.)

    """

    pk_res = send_fn(
        [
            {"role": "system", "content": "You are a helpful clinical assistant."},
            {"role": "user", "content": requesting_prompt},
        ],
        DEFAULT_LLM_ARGS,
    )

    result_json = json.dumps(
        {
            "Respiratory Rate (breaths per minute)": {
                "type": "numerical",
                "healthy range": "12~25",
            },
            "Heart rate (beats per minute)": {
                "type": "numerical",
                "healthy range": "60~100",
            },
            "Infection": {
                "type": "boolean",
            },
        }
    )

    lesion_keywords = "\n\n".join(
        [
            f"## Lesion: {k}\n**Features:**{json.dumps(v)}\n"
            for k, v in extracted_keywords_results["keywords"].items()
        ]
    )

    keyword_combining_prompt = f"""The following json objects are features from {len(extracted_keywords_results['keywords'])} different lesions. The key represents the feature, while the value indicates the data type.

    Please refine and combine the following features from {len(extracted_keywords_results['keywords'])} lesions, including {", ".join(list(extracted_keywords_results['keywords'].keys()))}. 

    These features will be used to predict diseases and lesions. However, some may have incorrect data types, so please correct them.

    Additionally, repetitive or similar features from different lesions should be combined or removed. For numerical values, ensure the unit is placed at the end of each feature, if applicable.

    When a feature is a diagnostic finding, change its data type to boolean.

    For example: 

    ```json
    {{
      "Lung Volume": "numerical",
      "Heart rate": "numerical",  
      "A showing B": "numerical",
    }}
    ```
    Should be:
    ```
    {{
      "Lung Volume (L)": "numerical",
      "Heart rate" (beats per minute): "numerical",  
      "A showing B": "boolean",
    }}
    ```

    If it's something like risk factors, please put the range at the end (e.g., 1-5, 1-10)

    (Please only return the json object without additional text)

    # Features

    {lesion_keywords}
    """

    refined_keywords_res = send_fn(
        [
            {
                "role": "system",
                "content": f"You are a helpful clinical assistant and has following information in mind:\n{pk_res.output}",
            },
            {"role": "user", "content": keyword_combining_prompt},
        ],
        DEFAULT_LLM_ARGS,
    )

    dataset_features = [
        "Gender",
        "Age",
        "Blood Pressure",
        "Body Temperature",
        "Heart rate",
        "Respiratory Rate",
        "Oxygen Saturation",
        "Age",
        "Gender",
    ]

    dataset_features_str = ", ".join(dataset_features)

    res_existing_features = send_fn(
        refined_keywords_res.history
        + [
            {
                "role": "user",
                "content": f"From above refined features, please indicate me the features that are exactly included in: {dataset_features_str}. (Only return a list of related features without additional text)",
            }
        ],
        DEFAULT_LLM_ARGS,
    )

    def extract_json_string(text):
        """
        This function extract the last json object in the text, because the responses return repetitive json objects sometimes.
        """
        json_start = text.find("{")
        json_end = text.find("}") + 1

        if json_end > json_start:
            return text[json_start:json_end]
        else:
            raise ValueError(
                f"The end of bracket is earlier than the start of bracket in the response:\n{text}."
            )

    keywords = json.loads(extract_json_string(refined_keywords_res.output))
    numerical_keywords = [k for k, v in keywords.items() if v == "numerical"]

    keyword_combining_prompt = f"""

    Please specify range of values of features for an health adult.
    Please return the json object without additional text.

    # Example

    ## Input
    ```json
    ["Respiratory rate (breaths per minute)", "Heart rate (beats per minute)"]
    ```

    ## Output
    ```json
    {{
      "Respiratory rate (breaths per minute)": "12-20",
      "Heart rate (beats per minute)": "60-100",  
    }}
    ```

    # Real

    ## Input
    ```json
    {json.dumps(numerical_keywords)}
    ```

    ## Output
    """
    healthy_numerical_range_res = send_fn(
        [
            {
                "role": "system",
                "content": f"You are a helpful clinical assistant and has following information in mind:\n{pk_res.output}",
            },
            {"role": "user", "content": keyword_combining_prompt},
        ],
        DEFAULT_LLM_ARGS,
    )

    keyword_combining_prompt = f"""

    Please specify range of possible values.
    Please return the json object without additional text.

    # Example

    ## Input
    ```json
    ["Respiratory rate (breaths per minute)", "Heart rate (beats per minute)"]
    ```

    ## Output
    ```json
    {{
      "Respiratory rate (breaths per minute)": "0-28",
      "Heart rate (beats per minute)": "0-300",  
    }}
    ```

    # Real

    ## Input
    ```json
    {json.dumps(numerical_keywords)}
    ```

    ## Output
    """
    possible_numerical_range_res = send_fn(
        [
            {
                "role": "system",
                "content": f"You are a helpful clinical assistant and has following information in mind:\n{pk_res.output}",
            },
            {"role": "user", "content": keyword_combining_prompt},
        ],
        DEFAULT_LLM_ARGS,
    )

    with open("separated_combined_results", "wb") as f:
        pickle.dump(
            {
                "prior_knowledge": pk_res,
                "refined_keyword": refined_keywords_res,
                "existing_features": res_existing_features,
                "healthy_numerical_range_res": healthy_numerical_range_res,
                "possible_numerical_range_res": possible_numerical_range_res,
            },
            f,
        )


if __name__ == "__main__":
    main()
