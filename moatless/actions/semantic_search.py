from typing import Optional, List, Type

from pydantic import Field

from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.actions.action import Action
from moatless.actions.model import ActionArguments
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse


class SemanticSearchArgs(SearchBaseArgs):
    """Search for code snippets based on semantic similarity."""

    query: str = Field(..., description="Natural language description of what you're looking for.")
    category: Optional[str] = Field(
        None,
        description="The category of files to search for. This can be 'implementation' for core implementation files or 'test' for test files.",
    )

    class Config:
        title = 'SemanticSearch'

    def to_prompt(self):
        prompt = f"Searching for code using the query: {self.query}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt


class SemanticSearch(SearchBaseAction):
    args_schema: Type[ActionArguments] = SemanticSearchArgs


    def _search(self, args: SemanticSearchArgs) -> SearchCodeResponse:
        return self._code_index.semantic_search(
            args.query,
            file_pattern=args.file_pattern,
            max_results=25,
            category=args.category,
        )

    def _search_for_alternative_suggestion(self, args: SemanticSearchArgs) -> SearchCodeResponse:
        return SearchCodeResponse()

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Query Relevance: Evaluate if the search query is well-defined and likely to find relevant code.",
                "Category Appropriateness: Assess if the category (implementation or test) aligns with the search intent.",
            ]
        )
        return criteria
