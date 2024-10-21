from typing import Optional, List, Type

from pydantic import Field

from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs, logger
from moatless.codeblocks import CodeBlockType
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, ActionOutput
from moatless.actions.action import FileContext


class FindFunctionArgs(SearchBaseArgs):
    """Search for a specific function or class in the codebase."""

    function_name: str = Field(..., description="Specific function names to include in the search.")
    class_name: Optional[str] = Field(
        default=None, description="Specific class name to include in the search."
    )

    class Config:
        title = 'FindFunction'

    def to_prompt(self):
        prompt = f"Searching for function: {self.function_name}"
        if self.class_name:
            prompt += f" in class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt


class FindFunction(SearchBaseAction):
    args_schema: Type[ActionArguments] = FindFunctionArgs

    
    def _search(self, args: FindFunctionArgs) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {args.function_name} (class_name: {args.class_name}, file_pattern: {args.file_pattern})"
        )
        return self._code_index.find_function(
            args.function_name,
            class_name=args.class_name,
            file_pattern=args.file_pattern,
        )

    def _search_for_alternative_suggestion(
        self, args: FindFunctionArgs
    ) -> SearchCodeResponse:
        """Return methods in the same class or other methods in same file with the method name the method in class is not found."""

        if args.class_name and args.file_pattern:
            file = self._repository.get_file(args.file_pattern)

            span_ids = []
            if file and file.module:
                class_block = file.module.find_by_identifier(args.class_name)
                if class_block and class_block.type == CodeBlockType.CLASS:
                    function_blocks = class_block.find_blocks_with_type(
                        CodeBlockType.FUNCTION
                    )
                    for function_block in function_blocks:
                        span_ids.append(function_block.belongs_to_span.span_id)

                function_blocks = file.module.find_blocks_with_identifier(
                    args.function_name
                )
                for function_block in function_blocks:
                    span_ids.append(function_block.belongs_to_span.span_id)

            if span_ids:
                return SearchCodeResponse(
                    hits=[
                        SearchCodeHit(
                            file_path=args.file_pattern,
                            spans=[SpanHit(span_id=span_id) for span_id in span_ids],
                        )
                    ]
                )

            return self._code_index.find_class(
                args.class_name, file_pattern=args.file_pattern
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
