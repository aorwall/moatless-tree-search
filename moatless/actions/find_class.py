import logging
from typing import List, Type, ClassVar

from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse

logger = logging.getLogger(__name__)


class FindClassArgs(SearchBaseArgs):
    """Find a specific class in the codebase."""

    class_name: str = Field(
        ..., description="Specific class name to include in the search."
    )

    @model_validator(mode="after")
    def validate_names(self) -> "FindClassArgs":
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        return self

    class Config:
        title = "FindClass"


class FindClass(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = FindClassArgs

    def to_prompt(self):
        prompt = f"Searching for class: {self.args.class_name}"
        if self.args.file_pattern:
            prompt += f" in files matching the pattern: {self.args.file_pattern}"
        return prompt

    def _search(self, args: FindClassArgs) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {args.class_name} (file_pattern: {args.file_pattern})"
        )
        return self._code_index.find_class(
            args.class_name, file_pattern=args.file_pattern
        )

    def _select_span_instructions(self, search_result: SearchCodeResponse) -> str:
        return (
            f"The class is too large. You must add the relevant functions to context to be able to use them. "
            f"Use the function RequestMoreContext and specify the SpanIDs of the relevant functions to add them to context.\n"
        )

    def _search_for_alternative_suggestion(
        self, args: FindClassArgs
    ) -> SearchCodeResponse:
        if args.file_pattern:
            return self._code_index.find_class(args.class_name, file_pattern=None)
        return SearchCodeResponse()

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Identifier Correctness: Verify that the class name is accurate.",
            ]
        )
        return criteria
