import logging
from abc import ABC
from typing import List, Optional, Type, Any, Dict, ClassVar

from pydantic import Field, PrivateAttr, BaseModel

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, RewardScaleEntry
from moatless.completion import CompletionModel
from moatless.completion.model import Message, UserMessage
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse
from moatless.repository.repository import Repository
from moatless.schema import FileWithSpans

logger = logging.getLogger(__name__)

IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with identifying relevant code in a codebase. Your goal is to select key code spans from the search results provided by another AI agent to address a specified issue.

# Input Structure:

<thoughts>: Contains the analysis and reflections from the initial AI agent on the task requirements.
<search_parameters>: Contains the search parameters used to retrieve code segments.
<search_results>: Contains the new search results, including various code spans.

# Your Task:

 1. Understand the Previous Agent's Intentions:
  * Carefully analyze the <thoughts> and <search_parameters> to comprehend the previous AI agent's search intentions and what it expects to find.
  * Identify key elements such as functions, variables, classes, or patterns that the previous agent deemed relevant.

 2. Evaluate Search Results:
  * Examine each code span in the <search_results> for alignment with the intentions and expectations identified from the previous agent's thoughts.
  * Assess the relevance of each code span based on how well it matches the expected findings.
  * Consider entire sections of code to ensure a complete understanding of the context and logic.
  * Note any references to other parts of the codebase that might be necessary to fulfill the previous agent's intentions.

 3. Respond Using the Identify Function:
  * Select and respond with the code spans that best align with the previous AI agent's search intentions and expectations.
  * Ensure that your response provides adequate context and completeness, reflecting the previous agent's thought process.
"""


class SearchBaseArgs(ActionArguments, ABC):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific files or directories.",
    )


class IdentifiedSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )


class Identify(ActionArguments):
    """Identify if the provided search result is relevant to the reported issue."""

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code spans in the search results identified as relevant to the reported issue.",
    )


class SearchBaseAction(Action):
    args_schema: ClassVar[Type[ActionArguments]] = SearchBaseArgs

    max_search_tokens: int = Field(
        1000,
        description="The maximum number of tokens allowed in the search results.",
    )
    max_hits: int = Field(
        5,
        description="The maximum number of search hits to display.",
    )
    completion_model: CompletionModel = Field(
        ...,
        description="The completion model used to identify relevant code spans in search results.",
    )

    _repository: Repository = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()

    def __init__(
        self,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        super().__init__(completion_model=completion_model, **data)
        self._repository = repository
        self._code_index = code_index

    def execute(
        self, args: SearchBaseArgs, file_context: FileContext | None = None
    ) -> Observation:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the search action."
            )

        properties = {"search_hits": [], "search_tokens": 0}

        identify_spans = False
        search_result = self._search(args)
        if not search_result.hits:
            search_result = self._search_for_alternative_suggestion(args)
            if not search_result.hits:
                return Observation(message=search_result.message or "No search results found", properties=properties)
            else:
                identify_spans = True
                logger.info(
                    f"{self.name}: No relevant search results found. Will use alternative suggestion with {search_result.hits} hits."
                )

        search_tokens = 0
        span_count = 0
        for hit in search_result.hits:
            search_tokens += sum(span.tokens for span in hit.spans)
            span_count += len(hit.spans)

            properties["search_hits"].append(
                {
                    "file_path": hit.file_path,
                    "spans": [span.span_id for span in hit.spans],
                }
            )

        properties["search_tokens"] = search_tokens

        message = search_result.message or ""

        if search_tokens > self.max_search_tokens or span_count > self.max_hits:
            logger.info(
                f"{self.name}: Search too large. {search_tokens=} {span_count=}, will ask for clarification."
            )
            identify_spans = True

        completion = None
        search_hit_str = ""
        found_files = set()
        if identify_spans:
            identify_message = self._generate_identify_prompt(args, search_result)
            identified_code, completion = self.completion_model.create_completion_with_response_model(
                messages=[identify_message],
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                response_model=Identify,
            )

            if identified_code.identified_spans:
                message += f"\nIdentified the following relevant code spans:\n"
                for identified_spans in identified_code.identified_spans:
                    search_hit_str += f"\n- File: {identified_spans.file_path}\n  Span IDs:"
                    for span_id in identified_spans.span_ids:
                        if not file_context.has_span(identified_spans.file_path, span_id):
                            found_files.add(identified_spans.file_path)
                            file_context.add_span_to_context(identified_spans.file_path, span_id)
                        search_hit_str += f"\n  - {span_id}"
        else:
            for hit in search_result.hits:
                search_hit_str += f"\n- File: {hit.file_path}\n  Span IDs:"
                for span in hit.spans:
                    if not file_context.has_span(hit.file_path, span.span_id):
                        found_files.add(hit.file_path)
                        file_context.add_span_to_context(hit.file_path, span.span_id)
                    search_hit_str += f"\n  - {span.span_id}"

        message += f"\n{search_hit_str}"

        if found_files:
            extra = "<updated_file_context>\n"
            extra += file_context.create_prompt(
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
                files=found_files,
            )
            extra += "\n</updated_file_context>"
        else:
            extra = None

        return Observation(message=message, extra=extra, properties=properties, completion=completion)

    def _select_span_instructions(self, search_result: SearchCodeResponse) -> str:
        return (
            f"The search result is too large. You must select the relevant code spans in the search results to the file context. "
            f"Use the function RequestMoreContext and specify the SpanIDs of the relevant code spans to add them to context.\n"
        )

    def _select_span_response_prompt(self, search_result: SearchCodeResponse) -> str:
        search_result_context = FileContext(repo=self._repository)
        for hit in search_result.hits:
            for span in hit.spans:
                search_result_context.add_span_to_context(
                    hit.file_path, span.span_id, tokens=100
                )

        search_result_str = search_result_context.create_prompt(
            show_span_ids=True,
            show_line_numbers=False,
            exclude_comments=True,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        prompt = self._select_span_instructions(search_result)
        prompt += f"\n<search_results>\n{search_result_str}\n</search_result>\n"
        return prompt

    def _search(self, args: SearchBaseArgs) -> SearchCodeResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def _search_for_alternative_suggestion(
        self, args: SearchBaseArgs
    ) -> SearchCodeResponse:
        return SearchCodeResponse()

    def _generate_identify_prompt(self, args: SearchBaseArgs, search_result: SearchCodeResponse) -> UserMessage:
        search_result_str = self._select_span_response_prompt(search_result)

        content = f"""<thoughts>
{args.scratch_pad}
</thoughts>

<search_parameters>
{args.model_dump_json(exclude={"scratch_pad"})}
</search_parameters>

<search_results>
{search_result_str}
</search_results>
"""

        return UserMessage(content=content)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Query Relevance: Evaluate if the search query or parameters are well-defined and likely to find relevant code.",
                "Search Scope Appropriateness: Check if the file patterns and class/function names narrow down the search effectively.",
                "Relevance of Search Results: Assess whether the search results are directly related to the problem and useful for making progress.",
                "Size of Search Results: Ensure that the code context provided is appropriately sized—not too large to overwhelm nor too small to be unhelpful.",
            ]
        )

        return evaluation_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        if trajectory_length <= 3:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action is excellent, with well-defined parameters yielding only highly relevant results.",
                    ),
                    (
                        75,
                        89,
                        "The search action is good, with reasonable parameters yielding relevant results.",
                    ),
                    (
                        25,
                        74,
                        "The search action have issues with parameters or yields few or no relevant results.",
                    ),
                    (
                        0,
                        24,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )
        else:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action significantly advances the solution, providing highly relevant and appropriately sized search results.",
                    ),
                    (
                        75,
                        89,
                        "The search action contributes positively towards solving the problem, with relevant results and minor issues.",
                    ),
                    (
                        50,
                        74,
                        "The search action is acceptable but may have issues with relevance or provides search results that are too large or too small.",
                    ),
                    (
                        25,
                        49,
                        "The search action provides results that are not helpful due to relevance or size issues.",
                    ),
                    (
                        0,
                        24,
                        "The search action has minimal impact, providing few relevant results.",
                    ),
                    (
                        -50,
                        -1,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )

    @classmethod
    def model_validate(cls, obj: Any) -> "SearchBaseAction":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository = obj.pop("repository")
            code_index = obj.pop("code_index")
            return cls(code_index=code_index, repository=repository, **obj)
        return super().model_validate(obj)
