def get_prompt_backup2(report, keyword_json, range_json):
    json_object_str = json.dumps(keyword_json)
    return f"""According to the following report, what can be the value of keywords provided in the json object? The json object provides the requested feature and their data types. Try to speculate the value instead of saying you don't know or not mentioned. Please return the values in json format as the data type json object used for indicating data types. And, don't forget the request of specific JSON format in the system prompt, so the returned json object should be a json string in the **description** in the **points**.
And, don't generated repetitive results. 

## Report
{report}

## IMPORTANT
- Please ensure that all keywords in the JSON object are not missed.
- Return actual values instead of data types.
- Please keep your responses concise, with a maximum length of {MAX_TOKENS} tokens.
- If uncertain, speculate the value rather than stating that you cannot answer.
- One keyword can't show up more than once.

## Data Type Json Object 
{json_object_str}


## 
"""


def get_prompt_backup(report, json_object):
    json_object_str = json.dumps(json_object)
    return f"""

IMPORTANT: You don't need to return Report and Json Object section again. You only need to return the speculated value in the required json format without additional text. You need to speculate the value, and null value is not acceptable. Try to speculate the value, you can't say that you're unable to answer this question.
The json object in the system message is just an example, please focus on the json project below.

## Report
{report}

## Json Object 
{json_object_str}

## Expected Output"""


# Don't miss any keyword in the **Data Type** when returning speculated values.
def get_system_message_backup(prior_knowledge):
    system_message = f"""You are a clinical expert. With following extra knowledge in mind:\n
{prior_knowledge}

# Task you're going to perform

You will be given a report regarding a patient, and a json object with for you to sepculate the values of each attribute. 
You will return the speculated values according to the data type specified in the give JSON Object.
Please, only return the json object without additional text.
You need to speculate the value, and null value is not acceptable.


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

## Expected Output
{EXAMPLE_OUTPUT_STR}
"""
    return system_message


def extract_json_string_backup(text):
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


def reverse_extract_json_string_backup(text):
    """
    This function extract the last json object in the text, because the responses return repetitive json objects sometimes.
    """
    json_start = text.rfind("{")
    json_end = text.rfind("}") + 1

    if json_end > json_start:
        return text[json_start:json_end]
    else:
        raise ValueError(
            f"The end of bracket is earlier than the start of bracket in the response:\n{text}."
        )


def extract_json_string_backup(text):
    # Check if "{" and "}" each occur exactly once
    if text.count("{") == 1 and text.count("}") == 1:
        # Extract the content between "{" and "}" (including braces)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json_str
    return
