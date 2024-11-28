from ..llm.send import ChatLLM, ModelArgs, replace_and_send
from ..template.community_report import COMMUNITY_REPORT_PROMPT
from ..utils.json import try_parse_json_object

from typing import Any
from dataclasses import dataclass
import re


import json

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
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)

    # Clean up any escape sequences or extra whitespace
    # json_string = json_string.replace('\\"', '"').replace("\\'", "'")

    # Regex to find JSON objects that do not close properly within an array
    # Add a missing closing brace before the closing bracket of an array element
    pattern = r'(\{[^}]*?)\n\s*]\s*(?=[,}])'
    json_string = re.sub(pattern, r'\1\n    }]', json_string)

    # Step 6: Maintain line breaks for readability
    json_string = re.sub(r'[\r\n]+', ' ', json_string).strip()

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


def clean_and_parse_json_backup_2(json_res):
    # Step 1: Remove code blocks and initial markers
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)
    
    # Step 2: Extract the JSON object between the braces
    json_string = re.search(r"{.*}", cleaned_string, re.DOTALL)
    if json_string:
        json_string = json_string.group(0)
    else:
        raise ValueError("No JSON object found in the response.")
    
    # Step 3: Remove trailing commas before closing braces/brackets
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)

    # Step 4: Ensure proper escaping of quotes
    json_string = json_string.replace('\\"', '"')

    # Step 5: Check if the JSON string is missing a closing brace
    if json_string.count('{') > json_string.count('}'):
        json_string += '}'

    # Step 6: Final cleaning to format newlines and escape characters
    json_string = re.sub(r'[\r\n]+', ' ', json_string).strip()

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


def clean_and_parse_json_backup(input_json: str):

    # Remove leading/trailing whitespace and extra characters if present
    cleaned_json = input_json.strip()

    # Unescape double backslashes to single backslashes
    cleaned_json = cleaned_json.replace("\\\\", "\\")

    # Replace escaped newline characters with actual newline
    cleaned_json = cleaned_json.replace("\\n", "\n")

    # Remove trailing commas before closing braces/brackets
    cleaned_json = re.sub(r",\s*([\]}])", r"\1", cleaned_json)

    # Ensure all colons are properly formatted
    cleaned_json = re.sub(r"\s*:\s*", ": ", cleaned_json)
    cleaned_json = re.sub(r"\s*,\s*", ", ", cleaned_json)

    # Remove invalid control characters if present
    cleaned_json = re.sub(r"[\x00-\x1f\x7f]", "", cleaned_json)

    # Try extracting a valid JSON string
    try:
        json_string = re.search(r"{.*}", cleaned_json, re.DOTALL).group(0)
    except AttributeError:
        raise ValueError("No valid JSON object found in the input string.")

    # Parse the JSON string
    try:
        result = json.loads(json_string)
    except json.JSONDecodeError:
        print("error loading json, json=%s", json_string)
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result


def clean_json_response(json_res):
    # Remove leading/trailing whitespace and extra characters if present
    json_res = json_res.strip()

    # Unescape double backslashes to single backslashes
    json_res = json_res.replace("\\\\", "\\")

    # Replace escaped newline characters with actual newline
    json_res = json_res.replace("\\n", "\n")

    # Remove trailing commas before closing braces/brackets
    json_res = re.sub(r",\s*([\]}])", r"\1", json_res)

    # Ensure all colons are properly formatted
    json_res = re.sub(r"\s*:\s*", ": ", json_res)
    json_res = re.sub(r"\s*,\s*", ", ", json_res)

    # Remove invalid control characters if present
    json_res = re.sub(r"[\x00-\x1f\x7f]", "", json_res)

    # Try extracting a valid JSON string
    try:
        json_string = re.search(r"{.*}", json_res, re.DOTALL).group(0)
    except AttributeError:
        raise ValueError("No valid JSON object found in the input string.")

    # Return the cleaned JSON string
    return json_string


def clean_json_response_backup5(json_res):
    # Remove code block markers if they exist
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Replace escaped double quotes with standard double quotes
    cleaned_string = cleaned_string.replace('\\"', '"')

    # Replace non-standard Unicode quotes with standard quotes
    cleaned_string = cleaned_string.replace("\u201c", '"').replace("\u201d", '"')
    cleaned_string = cleaned_string.replace("\u2018", "'").replace("\u2019", "'")

    # Ensure all newlines inside strings are properly escaped
    cleaned_string = re.sub(r'(?<!\\)(\\n|\n)(?=[^"]*")', r"\\n", cleaned_string)

    # Remove trailing commas before closing brackets/braces
    cleaned_string = re.sub(r",\s*([\]}])", r"\1", cleaned_string)

    # Remove extra spaces around colons or commas to ensure valid JSON format
    cleaned_string = re.sub(r"\s*:\s*", ": ", cleaned_string)
    cleaned_string = re.sub(r"\s*,\s*", ", ", cleaned_string)

    # Remove any double commas created during cleanup
    cleaned_string = re.sub(r",\s*,", ",", cleaned_string)

    # Ensure valid character encoding and remove any invalid control characters
    cleaned_string = re.sub(r"[\x00-\x1f\x7f]", "", cleaned_string)

    # Attempt to find the JSON object and clean up extraneous characters
    try:
        json_string = re.search(r"{.*}", cleaned_string, re.DOTALL).group(0)
    except AttributeError:
        raise ValueError("No valid JSON object found in the input string.")

    # Additional fix: check for invalid control characters and escape them
    json_string = re.sub(
        r"(?<!\\)([\b\f\r\t])", lambda x: "\\" + x.group(1), json_string
    )

    return json_string


def clean_json_response_backup_4(json_res):
    # Remove code block markers if they exist
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Remove double escaping (\\")
    cleaned_string = cleaned_string.replace('\\\\"', '"')

    # Replace single-escaped quotes (e.g., \") with standard double quotes
    cleaned_string = cleaned_string.replace('\\"', '"')

    # Replace non-standard quotes with standard quotes
    cleaned_string = (
        cleaned_string.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # Remove trailing commas before closing brackets/braces
    cleaned_string = re.sub(r",\s*([\]}])", r"\1", cleaned_string)

    # Try to find the JSON object and clean up any extraneous newlines or characters
    return cleaned_string


def clean_json_response_backup_3(json_res):
    # Remove code block markers
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Replace non-standard quotes with standard quotes
    cleaned_string = (
        cleaned_string.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # Ensure all double quotes within a string are properly escaped
    cleaned_string = re.sub(r'(?<!\\)"', r"\"", cleaned_string)

    # Remove trailing commas before closing brackets/braces
    cleaned_string = re.sub(r",\s*([\]}])", r"\1", cleaned_string)

    # Try to find the JSON object and clean up any extraneous newlines or characters
    try:
        json_string = re.search(r"{.*}", cleaned_string, re.DOTALL).group(0)
    except AttributeError:
        raise ValueError("No valid JSON object found in the input string.")

    return json_string


def clean_json_response_backup_2(json_res):
    # Remove any triple backticks or similar markers
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Replace single quotes with double quotes, but only for JSON properties/values
    cleaned_string = re.sub(r"(?<!\\)'", '"', cleaned_string)

    # Replace escape characters (like \n) with their literal counterparts
    cleaned_string = cleaned_string.replace("\\n", "\n").replace("\\t", "\t")

    # Clean up any trailing commas before closing braces or brackets
    cleaned_string = re.sub(r",\s*([\]}])", r"\1", cleaned_string)

    # Ensure it's a valid JSON format by loading it
    try:
        json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format after cleaning: {e}")

    return cleaned_string


def clean_json_response_backup(json_res):
    # Remove code block markers (e.g., ```json or ```python)
    cleaned_string = re.sub(r"```[a-zA-Z]*\n", "", json_res)

    # Extract the JSON object (ensure it captures everything between braces)
    json_string = re.search(r"{.*}", cleaned_string, re.DOTALL).group(0)

    # Remove any trailing commas before a closing brace or bracket
    json_string = re.sub(r",\s*([\]}])", r"\1", json_string)

    # Replace unescaped newline characters within strings with \n
    json_string = re.sub(r"(?<!\\)\n", r"\\n", json_string)

    # Ensure proper handling of quotes within strings
    json_string = json_string.replace("“", '"').replace("”", '"').replace("’", "'")

    try:
        # Parse to check if it is a valid JSON format
        json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format after cleaning: {e}")

    return json_string


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: dict


def dict_has_keys_with_types(
    data: dict, expected_fields: list[tuple[str, type]]
) -> bool:
    """Return True if the given dictionary has the given keys with the given types."""
    for field, field_type in expected_fields:
        if field not in data:
            return False

        value = data[field]
        if not isinstance(value, field_type):
            return False
    return True


def is_response_valid(x):
    return dict_has_keys_with_types(
        x,
        [
            ("title", str),
            ("summary", str),
            ("findings", list),
            ("rating", float),
            ("rating_explanation", str),
        ],
    )


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _send_to: ChatLLM
    _input_text_key: str
    _extraction_prompt: str
    _output_formatter_prompt: str
    _max_report_length: int
    _llm_args: ModelArgs
    _max_retries: int

    def __init__(
        self,
        send_to: ChatLLM,
        input_text_key: str | None = None,
        extraction_prompt: str | None = None,
        max_report_length: int | None = None,
        llm_args: ModelArgs = {},
        max_retries: int = 10,
    ):
        """Init method definition."""
        self._send_to = send_to
        self._input_text_key = input_text_key or "input_text"
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._max_report_length = max_report_length or 1500
        self._llm_args = llm_args
        self._max_retries = max_retries

    def __call__(self, inputs: dict[str, Any]):
        """Call method definition."""
        output = None

        # initialise retries.
        count = 0
        response_is_valid = False
        try:
            while count < self._max_retries and (not response_is_valid):
                # response = (
                #     self._send_to(
                #         self._extraction_prompt,
                #         json=True,
                #         name="create_community_report",
                #         variables={self._input_text_key: inputs[self._input_text_key]},
                #         is_response_valid=lambda x: dict_has_keys_with_types(
                #             x,
                #             [
                #                 ("title", str),
                #                 ("summary", str),
                #                 ("findings", list),
                #                 ("rating", float),
                #                 ("rating_explanation", str),
                #             ],
                #         ),
                #         model_parameters={"max_tokens": self._max_report_length},
                #     )
                #     or {}
                # )

                response = (
                    replace_and_send(
                        send_to=self._send_to,
                        template=self._extraction_prompt,
                        history=[],
                        replacing_variable={
                            self._input_text_key: inputs[self._input_text_key]
                        },
                        llm_args={
                            **self._llm_args,
                            "max_tokens": self._max_report_length,
                        },
                    )
                    or {}
                )

                json_response = clean_and_parse_json(response.output)

                response_is_valid = is_response_valid(json_response)

            if not response_is_valid:
                raise TimeoutError(
                    f"Have attempted {count} time on community report extraction."
                )

            output = json_response or {}

        except Exception as e:
            raise ValueError("error generating community report")
            print("error generating community report")
            output = {}

        text_output = self._get_text_output(output)
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _get_text_output(self, parsed_output: dict) -> str:
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"
