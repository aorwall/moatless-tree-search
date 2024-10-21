from typing import Optional, List, Type

from pydantic import Field

from moatless.actions.search_base import SearchBaseAction, logger
from moatless.codeblocks import CodeBlockType
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
from moatless.actions.action import Action, ActionArguments
from moatless.actions.action import FileContext, ActionOutput


class FindFunctionArgs(ActionArguments):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")
    function_name: str = Field(..., description="Specific function names to include in the search.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )
    class_name: Optional[str] = Field(
        default=None, description="Specific class name to include in the search."
    )


class FindFunction(Action):
    name: str = "FindFunction"
    description: str = "Search for a specific function or class in the codebase."
    args_schema: Type[ActionArguments] = FindFunctionArgs

    def execute(self, args: FindFunctionArgs, file_context: FileContext | None = None) -> ActionOutput:
        # Implementation of the execute method
        pass

    @property
    def log_name(self):
        if self.class_name:
            return f"FindFunction({self.class_name}.{self.function_name})"

        return f"FindFunction({self.function_name})"

    def to_prompt(self):
        prompt = f"Searching for function: {self.function_name}"
        if self.class_name:
            prompt += f" in class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def _search(self, code_index: CodeIndex) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {self.function_name} (class_name: {self.class_name}, file_pattern: {self.file_pattern})"
        )
        return code_index.find_function(
            self.function_name,
            class_name=self.class_name,
            file_pattern=self.file_pattern,
        )

    def _search_for_alternative_suggestion(
        self, code_index: CodeIndex
    ) -> SearchCodeResponse:
        """Return methods in the same class or other methods in same file with the method name the method in class is not found."""

        if self.class_name and self.file_pattern:
            file = self._workspace.file_repo.get_file(self.file_pattern)

            span_ids = []
            if file and file.module:
                class_block = file.module.find_by_identifier(self.class_name)
                if class_block and class_block.type == CodeBlockType.CLASS:
                    function_blocks = class_block.find_blocks_with_type(
                        CodeBlockType.FUNCTION
                    )
                    for function_block in function_blocks:
                        span_ids.append(function_block.belongs_to_span.span_id)

                function_blocks = file.module.find_blocks_with_identifier(
                    self.function_name
                )
                for function_block in function_blocks:
                    span_ids.append(function_block.belongs_to_span.span_id)

            if span_ids:
                return SearchCodeResponse(
                    hits=[
                        SearchCodeHit(
                            file_path=self.file_pattern,
                            spans=[SpanHit(span_id=span_id) for span_id in span_ids],
                        )
                    ]
                )

            return code_index.find_class(
                self.class_name, file_pattern=self.file_pattern
            )

        return SearchCodeResponse()

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Function Identifier Accuracy: Ensure that the function name is correctly specified.",
                "Class Name Appropriateness: Verify that the class names, if any, are appropriate.",
            ]
        )
        return criteria
