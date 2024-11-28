from typing import Callable
import pandas as pd
import secret
import json
import os
import pickle
import re
import regex

from tqdm import tqdm
from graphrag_for_all.llm.openai import set_openai_api_key
from graphrag_for_all.llm.huggingface import set_hugging_face_token
from graphrag_for_all.llm.create import get_send_fn
from utils.query import get_questions_by_lesion
import warnings
from graphrag_for_all.search.community_context import (
    ConversationHistory,
    ConversationRole,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


warnings.filterwarnings("ignore")

set_openai_api_key(secret.OPENAI_API_KEY)
set_hugging_face_token(secret.HUGGINGFACE_TOKEN)
send_fn = get_send_fn(
    source="huggingface",
    model_name="ruslanmv/Medical-Llama3-8B",  # ["meta-llama/Meta-Llama-3.1-8B-Instruct", "microsoft/Phi-3.5-mini-instruct", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
)
DEFAULT_LLM_ARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
}  #
MAX_TOKENS = 12000
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


def create_map_from_string(data):
    # Initialize an empty dictionary
    key_value_map = {}

    # Use a regex pattern to extract the key-value pairs
    pattern = re.compile(r'"([^"]+)":\s*([^\n]+)')

    # Find all matches and add them to the dictionary
    for match in pattern.findall(data):
        key = match[0].strip()  # Clean the key
        value = (
            match[1].strip().strip(",")
        )  # Clean the value and remove the trailing comma if present

        # Convert value to appropriate type (boolean, int, float, etc.)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif re.match(r"^\d+(\.\d+)?$", value):
            value = float(value)
        else:
            value = None

        key_value_map[key] = value

    return key_value_map


def extract_values_from_map(key_value_map, keywords):
    extracted_values = {}

    # Search for each keyword in the map
    for keyword in keywords:
        if keyword in key_value_map:
            extracted_values[keyword] = key_value_map[keyword]

    return extracted_values


def remove_data_reports(text):
    # Remove patterns like "[Data: Reports (0, 5, 12, 18, 26)]"
    cleaned_text = re.sub(r"\[Data: Reports \(.*?\)\]", "", text)

    # Remove extra spaces that might result from removal
    # cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text


def get_diagnosis(data, label_cols):
    diagnosis = [k for k, v in dict(data[label_cols] > 0).items() if v > 0]
    if len(diagnosis) == 0:
        return " No lesion found"

    diagnosis_str = ""
    for l in diagnosis:
        diagnosis_str += f" {l},"

    return diagnosis_str[:-1]


def dict_raise_on_duplicates(ordered_pairs):
    """Overwrite duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        # if k in d:
        #    raise ValueError("duplicate key: %r" % (k,))
        # else:
        d[k] = v
    return d


def get_report(
    data,
    mimic_eye_path: str = MIMIC_EYE_PATH,
    label_cols: str = REFLACX_LESION_LABEL_COLS,
):
    # reflacx_id = data['id']
    patient_id = data["subject_id"]
    study_id = data["study_id"]
    # dicom_id = data['dicom_id']
    report_path = os.path.join(
        mimic_eye_path,
        f"patient_{patient_id}",
        "CXR-DICOM",
        f"s{study_id}.txt",
    )
    with open(report_path) as f:
        report = f.read()

    report = (
        report.strip()
        .replace("FINAL REPORT\n", "")
        .replace("\n \n ", "\n")
        .replace("\n ", "")
        .strip()
    )

    clinical_features_str = f"DIAGNOSED LESIONS:{get_diagnosis(data, label_cols)}.\n"
    if data["age"] is not None:
        clinical_features_str += f"AGE: {data['age']}.\n"
    if data["gender"] is not None:
        clinical_features_str += f"GENDER: {data['gender']}.\n"
    if data["temperature"] is not None:
        clinical_features_str += (
            f"BODY TEMPERATURE (FAHRENHEIT): {data['temperature']}.\n"
        )
    if data["heartrate"] is not None:
        clinical_features_str += f"HEART RATE (PER MINUTE): {data['heartrate']}.\n"
    if data["resprate"] is not None:
        clinical_features_str += f"RESPIRATORY RATE (PER MINUTE): {data['resprate']}.\n"
    if data["o2sat"] is not None:
        clinical_features_str += f"OXYGEN SATURATION (PERCENTAGE): {data['o2sat']}.\n"
    if data["sbp"] is not None:
        clinical_features_str += f"SYSTOLIC BLOOD PRESSURE (mmHg): {data['sbp']}.\n"
    if data["dbp"] is not None:
        clinical_features_str += f"SYSTOLIC BLOOD PRESSURE (mmHg): {data['dbp']}.\n"
    if data["pain"] is not None:
        clinical_features_str += f"Pain (0-10): {data['dbp']}.\n"
    if data["acuity"] is not None:
        clinical_features_str += (
            f"Acuity (1(highest priority) - 5(lowest priority)): {data['acuity']}.\n"
        )

    return re.sub(
        "[^0-9A-Za-z.\s\:']",
        "",
        f"{report}\n{clinical_features_str}",
    )


EXAMPLE_JSON_STR = json.dumps(
    {
        "Chest pain": "boolean",
        "Weight loss": "boolean",
        "History of COPD": "boolean",
        "Heartrate (beats per minute)": "numerical",
        "Oxygen levels (mmHg)": "numerical",
    }
)

EXAMPLE_HEALTHY_RANGE = json.dumps(
    {
        "Heartrate (beats per minute)": "60-100",
        "Oxygen levels (mmHg)": "75-100",
    }
)

EXAMPLE_OUTPUT_STR = json.dumps(
    {
        "Chest pain": True,
        "Weight loss": False,
        "History of COPD": True,
        "Heartrate (beats per minute)": 90,
        "Oxygen levels (mmHg)": 99.0,
    }
)


def get_system_message(prior_knowledge):
    return f"""You are a clinical expert. With following extra knowledge in mind:\n
{prior_knowledge}

# Task you're going to perform

You will be given a report regarding a patient, and a json object with for you to sepculate the values of each attribute. 
You will return the speculated values according to the data type specified in the give JSON Object. You need to speculate the value, and null value is not acceptable.
Please keep your responses concise, with a maximum length of {MAX_TOKENS} tokens.
Don't be unit (e.g., cm, L) in the value. Only numerical characters for numerical data types, and only true or false for boolean data types.

Following is an example for you:

## Report
INDICATION:  Central venous line placement.
TECHNIQUE:  Frontal chest radiograph.
COMPARISON:  Chest radiograph 12:42 today.
FINDINGS: 
A right subclavian catheter has been placed in the interim. The catheterterminates at the confluence of the brachiocephalic vein and superior venacava and if indicated could be advanced 3.7 cm for termination within thelow SVC.
There is no pleural effusion or pneumothorax. The cardiac silhouette remainsmildly enlarged. There is no focal airspace consolidation worrisome forpneumonia.
High density material is again seen in the paritally imaged colon in the leftabdomen. Cholecystectomy clips are noted. There are carotid calcificationsleft greater than right.
DIAGNOSED LESIONS: Enlarged cardiac silhouette.
AGE: 69.
GENDER: Female.

## Data Type 
{EXAMPLE_JSON_STR}

## Healthy Patient Feature Range
{EXAMPLE_HEALTHY_RANGE}

## Speculated Values
{EXAMPLE_OUTPUT_STR}
"""


# - In the final result, you just need to return the **Expected Output** section without additional text.
def get_prompt(report, keyword_json, range_json):
    keyword_json_str = json.dumps(keyword_json)
    range_json_str = json.dumps(range_json)

    return f"""Based on the following "Report", determine the potential values for the keywords listed in the 'Data Type' JSON object, which specifies the required keywords and their data types.

## IMPORTANT
- Return the results in JSON format, adhering to the data types defined in the 'Data Type' JSON object.
- Make sure to include all keywords in the "Data Type" when returning speculated values, especially when both boolean and numerical keywords are required. For example, if the keywords include both "Increased Cardiothoracic ratio" and "Cardiothoracic ratio," ensure that neither is missed.
- Return actual values instead of data types.
- Please keep your responses concise, with a maximum length of {MAX_TOKENS} tokens.
- If uncertain, speculate the value rather than stating that you cannot answer rather than stating 'unknown' or 'not mentioned,' Don't say "I am sorry but I am unable to answer this question given the provided data.".
- Avoid generating repetitive results.
- Avoid returning repetitive keywords.
- For numerical keywords, refer to the 'Healthy Patient Feature Range' JSON object for guidance. However, the values do not need to fall strictly within these ranges, as not all patients are healthy—just ensure the values are realistic.
- Your response must a JSON object without additional text.
- You only return the speculated values in JSON format without additional content. 

## Report
{report}

## Data Type 
{keyword_json_str}

## Healthy Patient Feature Range
{range_json_str}

## Speculated Values 
"""


def extract_json_string(text):
    """
    Extracts the first valid JSON object from the text, ensuring no unmatched curly braces are included.
    """
    # This regular expression matches the first JSON object (between curly braces).
    # It ensures that the content inside curly braces is balanced.
    pattern = r"(\{(?:[^{}]*|(?R))*\})"

    match = regex.search(pattern, text)

    if match:
        json_str = match.group(1)
        return json_str
        # Remove the outer curly braces to return a valid JSON string without them
        # return json_str[1:-1]
    else:
        raise ValueError("No valid JSON object found in the text.")


def reverse_extract_json_string(text):
    """
    Extracts the last valid JSON object from the text, ensuring no unmatched curly braces are included.
    """
    # This regular expression matches the last JSON object (between curly braces).
    # It ensures that the content inside curly braces is balanced.
    pattern = r"(\{(?:[^{}]*|(?R))*\})$"

    match = regex.search(pattern, text)

    if match:
        json_str = match.group(1)
        return json_str
        # Remove the outer curly braces to return a valid JSON string without them
        # return json_str[1:-1]
    else:
        raise ValueError("No valid JSON object found in the text.")


def load_json_with_latest_key(data):
    # Use regex to find all key-value pairs
    pattern = re.compile(r'"([^"]+)":\s*([^\n,]+)')

    # Dictionary to store the latest occurrence of each key
    key_value_map = {}

    # Iterate over all matches
    for match in pattern.findall(data):
        key = match[0].strip()  # Clean key
        value = match[1].strip().strip(",")  # Clean value

        # Convert value to appropriate type (boolean, int, float, etc.)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif re.match(r"^\d+(\.\d+)?$", value):
            value = float(value)

        # Store the latest occurrence of each key
        key_value_map[key] = value

    # Convert the map back to a valid JSON string
    json_string = json.dumps(key_value_map)

    # Load it into a Python dictionary
    return json.loads(json_string)


def generate_values_without_graphrag(messages: str, send_fn: Callable):
    return send_fn(
        messages,
        DEFAULT_LLM_ARGS,
    )


def extracting_json_forward_and_backward(
    response: str, keywords: dict, send_fn: Callable
):

    cleaned = extract_json_string(response)

    if cleaned is None or cleaned == "":
        keywords_str = json.dumps(keywords)
        input_messages = [
            {"role": "system", "content": "You are a helpful clinical assistant."},
            {
                "role": "user",
                "content": f"The following information is not in the json format expected. Please transform them into json format.\n{response}\n\n\n\nThis is a json object indicates the data type of each feature.\n{keywords_str}\n\n\n\nPlease return the json format containing the values instead of data type without additional text. ",
            },
        ]
        refined_res = send_fn(
            input_messages,
            DEFAULT_LLM_ARGS,
        )
        cleaned = extract_json_string(refined_res.output)
        if cleaned is None:
            raise ValueError(f"JSON Object can't be extracted from output: {response}")

    json_obj = load_json_with_latest_key(cleaned)

    # try to get the last
    # Value check
    if ("boolean" in json_obj.values()) or ("numerical" in json_obj.values()):
        cleaned = reverse_extract_json_string(response)
        if cleaned is None or cleaned == "":
            keywords_str = json.dumps(keywords)
            input_messages = [
                {"role": "system", "content": "You are a helpful clinical assistant."},
                {
                    "role": "user",
                    "content": f"The following information is not in the json format expected. Please transform them into json format.\n{response}\n\n\n\nThis is a json object indicates the data type of each feature.\n{keywords_str}\n\n\n\nPlease return the json format containing the values instead of data type without additional text.",
                },
            ]
            refined_res = send_fn(
                input_messages,
                DEFAULT_LLM_ARGS,
            )
            cleaned = extract_json_string(refined_res.output)
            if cleaned is None:
                raise ValueError(
                    f"JSON Object can't be extracted from output: {response}"
                )
        json_obj = load_json_with_latest_key(cleaned)

    if ("boolean" in json_obj.values()) or ("numerical" in json_obj.values()):
        raise ValueError(f"data types in {json_obj}")

    if json_obj is None:
        raise ValueError("Retrieved JSON object is None")

    return json_obj


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
)
def get_and_parse_json(
    system_message: str,
    query: str,
    keywords_json: dict,
    send_fn: Callable,
    range_json: dict,
):
    res = generate_values_without_graphrag(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        send_fn=send_fn,
    )
    print("Response: ", res.output)
    json_obj = extracting_json_forward_and_backward(res.output, keywords_json, send_fn)
    # Keyword check
    required_keywords = set(list(keywords_json.keys()))
    res_keywords = set(list(json_obj.keys()))
    if res_keywords != required_keywords:
        lose_keywords = required_keywords.difference(res_keywords)
        extra_keywords = res_keywords.difference(required_keywords)

        if len(lose_keywords) == 0:
            # remove extra keywords
            for k in extra_keywords:
                del json_obj[k]
        else:
            print(f"Missing keywords {lose_keywords}")

            missing_keywords_map = {k: keywords_json[k] for k in lose_keywords}
            missing_range_map = {k: range_json[k] for k in lose_keywords}

            missing_query = get_prompt(
                report,
                missing_keywords_map,
                missing_range_map,
            )

            # instead of sending with history, just send a new request so the LLM does not get confused.
            missing_res = generate_values_without_graphrag(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": missing_query},
                ],
                send_fn=send_fn,
            )
            # TODO: Always missing for some instances(How?):
            # Q: Are them in the data type json object?
            # - Cardiothoracic ratio
            # - Lung expansion

            print("Missing response: ", missing_res.output)
            missing_json_obj = extracting_json_forward_and_backward(
                missing_res.output, keywords_json, send_fn
            )

            json_obj.update(missing_json_obj)

            res_keywords = set(list(json_obj.keys()))
            lose_keywords = required_keywords.difference(res_keywords)
            if json_obj is None or len(lose_keywords) != 0:
                raise ValueError(f"Still Missing keywords {lose_keywords}")

    return json_obj


def clean_json_response(json_res):
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)
    json_string = re.search(r"{.*}", cleaned_string, re.DOTALL).group(0)
    json_string = json_string.replace("]\n  ]\n}", "}\n  ]\n}")
    json_string = json_string.replace('"\n  ]\n}', '"\n    }\n  ]\n}')
    # json_string = json_string.replace('\n', '').replace(',}', '}').replace(',]', ']') # form to one line.
    return json_string


if __name__ == "__main__":
    with open("separated_combined_results", "rb") as f:
        separated_combined_results = pickle.load(f)
    top_5_lesions = [
        # "pulmonary edema",
        "enlarged cardiac silhouette",
        "pulmonary consolidation",
        "atelectasis",
        "pleural abnormality",
    ]
    sample_df = pd.read_csv("./spreadsheets/reflacx_clinical.csv")
    augmented = []
    keywords_json = json.loads(
        clean_json_response(separated_combined_results["refined_keyword"].output)
    )

    range_json = json.loads(
        clean_json_response(
            separated_combined_results["healthy_numerical_range_res"].output
        )
    )

    keywords_json["Body Temperature (degrees Celsius)"] = keywords_json["Body Temperature (°C)"]
    del keywords_json["Body Temperature (°C)"]
    keywords_json["White Blood Cell Count (cells/mcL)"] = keywords_json["White Blood Cell Count (cells/μL)"]
    del keywords_json["White Blood Cell Count (cells/μL)"]

    range_json["Body Temperature (degrees Celsius)"] = range_json["Body Temperature (°C)"]
    del range_json["Body Temperature (°C)"]
    range_json["White Blood Cell Count (cells/mcL)"] = range_json["White Blood Cell Count (cells/μL)"]
    del range_json["White Blood Cell Count (cells/μL)"]

    
    system_message = get_system_message(
        prior_knowledge=separated_combined_results["prior_knowledge"].output
    )

    for idx, row in tqdm(sample_df.iterrows()):
        if idx < 221:
            continue

        print(f"Processing Index [{idx}]")
        report = get_report(row)
        # query = get_prompt(report, keywords_json)
        query= get_prompt(
                report,
                keywords_json,
                range_json,
            )
        
        # input_messages = [
        #     {
        #         "role": "system",
        #         "content": system_message,
        #     },
        #     {
        #         "role": "user",
        #         "content": get_prompt(
        #             report,
        #             keywords_json,
        #         ),
        #     },
        # ]
        aug_values = get_and_parse_json(
            system_message=system_message,
            query=query,
            keywords_json=keywords_json,
            send_fn=send_fn,
            range_json=range_json,
        )
        aug_instance = dict(row)
        aug_instance.update({f"Augmented_{k}": v for k, v in aug_values.items()})
        aug_instance.update(
            {
                "augmenting_query": query,
                "augmenting_output": json.dumps(aug_values),
            }
        )
        augmented.append(aug_instance)
        augmented_df = pd.DataFrame(augmented)
        augmented_df.to_csv(f"medi-llama-augmented-nodb_{idx}.csv")
