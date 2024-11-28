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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

MIMIC_EYE_PATH = "F:\\mimic-eye"
REFLACX_LESION_LABEL_COLS = [
    # "Fibrosis",
    # "Quality issue",
    # "Wide mediastinum",
    # "Fracture",
    # "Airway wall thickening",
    ######################
    # "Hiatal hernia",
    # "Acute fracture",
    # "Interstitial lung disease",
    # "Enlarged hilum",
    # "Abnormal mediastinal contour",
    # "High lung volume / emphysema",
    # "Pneumothorax",
    # "Lung nodule or mass",
    # "Groundglass opacity",
    ######################
    "Pulmonary edema",
    "Enlarged cardiac silhouette",
    "Consolidation",
    "Atelectasis",
    "Pleural abnormality",
    # "Support devices",
]
DEFAULT_LLM_ARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
}  #


def build_prior_knowledge(extracted_keywords_results):
    lesion_qa_pairs = extracted_keywords_results["responses"]

    prior_knowledge = OrderedDict({})
    for lesion, q_a in lesion_qa_pairs.items():
        q_a_section = ""
        for q, a in q_a.items():
            q_a_section += f"\n#############################################\n**Question**: {q}\n**Answer**:\n{a}\n"
        lesion_content = f"## Lesion: {lesion}\n" + q_a_section
        prior_knowledge[lesion] = lesion_content
    return prior_knowledge


def clean_and_parse_json(json_res):
    # Step 1: Remove code block markers and initial text
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Step 2: Extract JSON content between braces
    json_string = re.search(r"{.*}", cleaned_string, re.DOTALL)
    if json_string:
        json_string = json_string.group(0)
    else:
        raise ValueError("No JSON object found in the response.")

    # Step 3: Remove trailing commas before closing brackets/braces
    json_string = re.sub(r",\s*([\]}])", r"\1", json_string)

    # Clean up any escape sequences or extra whitespace
    # json_string = json_string.replace('\\"', '"').replace("\\'", "'")

    # Regex to find JSON objects that do not close properly within an array
    # Add a missing closing brace before the closing bracket of an array element
    pattern = r"(\{[^}]*?)\n\s*]\s*(?=[,}])"
    json_string = re.sub(pattern, r"\1\n    }]", json_string)

    # Step 6: Maintain line breaks for readability
    json_string = re.sub(r"[\r\n]+", " ", json_string).strip()

    # Return cleaned JSON string
    try:
        result = json.loads(json_string)
    except json.JSONDecodeError:
        print("error loading json, json=%s", json_string)
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
)
def combine_and_refine_keywords(
    extracted_keywords_results: dict,
    send_fn,
    DEFAULT_LLM_ARGS,
    pk_res,
):
    lesion_keywords = "\n\n".join(
        [
            f"## Lesion: {k}\n**Keywords:**{json.dumps(v)}\n"
            for k, v in extracted_keywords_results["keywords"].items()
        ]
    )

    keyword_combining_prompt = f"""The following json objects are keywords from {len(extracted_keywords_results['keywords'])} different lesions. The key represents the feature, while the value indicates the data type.

# Keywords
{lesion_keywords}

Please refine and combine the keywords above from {len(extracted_keywords_results['keywords'])} lesions into a single set of keywords.

Please only refine the existing keywords and combine them into a single set. Do not add new keywords.

Try to make keywords consistent and remove any duplicates or similar keywords.

The keyword is not a long phrase or sentence, should be a single word or a few words that represent a feature. 

Refine the keywords the keywords if you find it to be a long phrase or sentence.

The keywords should be those that can be represented as numerical or boolean values and stored as tabular data.

These keywords will be used to predict diseases and lesions. However, some may have incorrect data types, so please correct them.

Additionally, repetitive or similar keywords from different lesions should be combined or removed. 

For numerical values, ensure the unit is placed at the end of each feature, if applicable.

For example: 

```json
{{
"Lung Volume": "numerical",
"Heart rate": "numerical",  
"A showing B": "numerical",
}}
```
Should be:
```json
{{
"Lung Volume (L)": "numerical",
"Heart rate" (beats per minute): "numerical",  
"A showing B": "boolean",
}}
```

Ensure that you return a flat JSON object, where each feature is a key and its data type is the value. Avoid using nested objects.

When a feature is a diagnostic finding, change its data type to boolean.

If it's something like risk factors, please put the range at the end (e.g., 1-5, 1-10)

(Please only return the json object without additional text)


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
    return clean_and_parse_json(refined_keywords_res.output), refined_keywords_res


def main():

    set_openai_api_key(secret.OPENAI_API_KEY)
    set_hugging_face_token(secret.HUGGINGFACE_TOKEN)
    send_fn = get_send_fn(
        source="huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )

    with open(
        "./combined_index_results/graphrag/index_graphrag_mistral_combined_top_1/separated_extracted_keywords.pkl",
        "rb",
    ) as f:
        extracted_keywords_results = pickle.load(f)

    # with open(
    #     "./combined_index_results/graphrag/index_graphrag_mistral_combined_top_1/combined_extracted_keywords.pkl",
    #     "rb",
    # ) as f:
    #     combined_extracted_keywords = pickle.load(f)

    # searcher = Searcher(
    #     input_dir="./combined_index_results/graphrag/index_graphrag_mistral_combined_top_1/",
    #     send_to=send_fn,
    #     community_level=1,
    # )

    prior_knowledge = build_prior_knowledge(extracted_keywords_results)
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

    refined_keywords, refined_keywords_res = combine_and_refine_keywords(
        extracted_keywords_results,
        send_fn,
        DEFAULT_LLM_ARGS,
        pk_res,
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
                "content": f"From above refined keywords, please indicate me the keywords that are exactly included in: {dataset_features_str}. (Only return a list of related keywords without additional text)",
            }
        ],
        DEFAULT_LLM_ARGS,
    )

    numerical_keywords = [k for k, v in refined_keywords.items() if v == "numerical"]

    keyword_combining_prompt = f"""

    Please specify range of values of keywords for a health adult.
    The json object should have the keyword as key and the range as value. No nested objects.

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

    (Please return the json object without additional text.)
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

    with open("separated_combined_results_mistral", "wb") as f:
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
