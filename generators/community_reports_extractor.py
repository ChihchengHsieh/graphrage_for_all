from dataclasses import dataclass
from llm.send import ChatLLM, ModelArgs, replace_and_send
from template.community_report import COMMUNITY_REPORT_PROMPT
from typing import Any
from utils.json import try_parse_json_object


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
                    or {},
                )

                json_response = try_parse_json_object(response.output)
                response_is_valid = is_response_valid(json_response)

            if not response_is_valid:
                raise TimeoutError(
                    f"Have attempted {count} time on community report extraction."
                )

            output = json_response or {}
        except Exception as e:
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
