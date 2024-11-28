from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import json
import re
import html

from typing import Callable, Any


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back

    input = re.search(r"{.*}", input, re.DOTALL).group(0)

    if not isinstance(input, str):
        return input

    re.sub(r"^\(|\)$", "", input.strip())

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def responses_to_keywords(
    lesion,
    responses: list[str],
    send_fn: Callable,
):
    combined_response = " ".join(responses)

    prompt = f"""The subsequent paragraph are the information related to lesion, Please extract the clinical keywords from the following paragraph. The keywords should be those that can be represented as numerical or boolean values and stored as tabular data. And, don't extract the lesion itself as a keyword. (note: Please return the keyword with format of dictionary without additional text.)


Please return only the keywords along with their corresponding data types. There is no need to include the values of the keywords.

Below, I will provide an example, after which you will extract the keywords from the context in "Real Data".

#################################
Example:
#################################
- Lesion: Atelectasis
- Context: 
--- Start of context ---
Atelectasis is more common in certain populations, including surgical patients, particularly those undergoing chest or abdominal procedures, due to factors like anesthesia, pain, and restricted breathing. Aging and chronic respiratory conditions like COPD and asthma increase the risk, while prolonged immobility, especially in bedridden individuals, further reduces lung expansion. Premature infants are also vulnerable due to underdeveloped lungs. Symptoms of atelectasis include shortness of breath, rapid shallow breathing, coughing, sharp chest pain, and low oxygen levels, which can lead to cyanosis. Fever may indicate an infection, and decreased breath sounds are often observed during physical examination.
--- End of context ---
########
Output: 
{{"Surgical patients": "boolean","Chest or abdominal procedures": "boolean","Anesthesia": "boolean","Pain": "boolean","Restricted breathing": "boolean","Breathing Rate": "numerical","Age": "numerical","Chronic respiratory conditions": "boolean","COPD": "boolean","Asthma": "boolean","Prolonged immobility": "boolean","Bedridden": "boolean","Premature infants": "boolean","Shortness of breath": "boolean","Rapid shallow breathing": "boolean","Coughing": "boolean","Sharp chest pain": "boolean","Oxygen levels": "numerical","Cyanosis": "boolean","Fever": "boolean","Body Temperature": "numerical","Decreased breath sounds": "boolean"}}

#################################
Real Data:
#################################
- Lesion: {lesion}
- Context: 
--- Start of context ---
{combined_response}
--- End of context ---
########
Output:"""
    res = send_fn(
        [
            {"role": "system", "content": "You are a helpful clinical expert."},
            {"role": "user", "content": prompt},
        ],
        {},
    )

    keywords = json.loads(clean_str(res.output))
    return keywords


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def doc_responses_to_keywords(
    responses: dict[str, dict[str, str]],
    send_fn: Callable,
):

    all_lesion_combined_response = "\n\n\n\n".join(
        ["\n".join(list(r_v.values())) for r_v in responses.values()]
    )

    prompt = f"""The subsequent paragraph are the information related to lesions, Please extract the clinical keywords from the following paragraph. The keywords should be those that can be represented as numerical or boolean values and stored as tabular data. (note: Please return the keyword with json format without additional text.)

    #################################
    Example:
    #################################
    Context:
    Atelectasis is more common in certain populations, including surgical patients, particularly those undergoing chest or abdominal procedures, due to factors like anesthesia, pain, and restricted breathing. Aging and chronic respiratory conditions like COPD and asthma increase the risk, while prolonged immobility, especially in bedridden individuals, further reduces lung expansion. Premature infants are also vulnerable due to underdeveloped lungs. Symptoms of atelectasis include shortness of breath, rapid shallow breathing, coughing, sharp chest pain, and low oxygen levels, which can lead to cyanosis. Fever may indicate an infection, and decreased breath sounds are often observed during physical examination.
    ########
    Output: 
    {{"Surgical patients": "boolean","Chest or abdominal procedures": "boolean","Anesthesia": "boolean","Pain": "boolean","Restricted breathing": "boolean","Breathing Rate": "numerical","Age": "numerical","Chronic respiratory conditions": "boolean","COPD": "boolean","Asthma": "boolean","Prolonged immobility": "boolean","Bedridden": "boolean","Premature infants": "boolean","Shortness of breath": "boolean","Rapid shallow breathing": "boolean","Coughing": "boolean","Sharp chest pain": "boolean","Oxygen levels": "numerical","Cyanosis": "boolean","Fever": "boolean","Body Temperature": "numerical","Decreased breath sounds": "boolean"}}

    #################################
    Real Data:
    #################################
    Context:
    {all_lesion_combined_response}
    ########
    Output:"""
    res = send_fn(
        [
            {"role": "system", "content": "You are a helpful clinical expert."},
            {"role": "user", "content": prompt},
        ],
        {},
    )

    keywords = json.loads(clean_str(res.output))
    return keywords
