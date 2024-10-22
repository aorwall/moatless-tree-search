import logging
from typing import Optional, Type

from pydantic import Field

from moatless.actions.model import ActionArguments
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.index.types import SearchCodeResponse

logger = logging.getLogger(__name__)


class FindCodeSnippetArgs(SearchBaseArgs):
    """Request to search for an exact code snippet."""

    code_snippet: str = Field(..., description="The exact code snippet to find.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    class Config:
        title = "FindCodeSnippet"

    def to_prompt(self):
        prompt = f"Searching for code snippet: {self.code_snippet}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt


class FindCodeSnippet(SearchBaseAction):
    args_schema: Type[ActionArguments] = FindCodeSnippetArgs

    def _search(self, args: FindCodeSnippetArgs) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {args.code_snippet} (file_pattern: {args.file_pattern})"
        )

        return self._code_index.semantic_search(
            code_snippet=args.code_snippet,
            file_pattern=args.file_pattern,
            max_results=5,
        )
