import logging
from typing import Optional, Tuple

from pydantic import Field, BaseModel

from moatless.actions.model import ActionArguments
from moatless.completion import CompletionModel
from moatless.completion.model import Completion, UserMessage
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class IdentifiedSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    start_line: int = Field(
        description="Starting line number of the relevant code section."
    )
    end_line: int = Field(
        description="Ending line number of the relevant code section."
    )


class Identify(ActionArguments):
    """Identify if the provided search result is relevant to the reported issue."""

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code sections in the search results identified as relevant to the reported issue.",
    )


class IdentifyCodeSpans(BaseModel):
    max_identify_tokens: int = Field(
        8000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )

    completion_model: Optional[CompletionModel] = Field(
        None,
        description="The completion model used to identify relevant code sections in search results.",
    )

    def _identify_code(
        self, instructions: str, file_context: FileContext
    ) -> Tuple[IdentifiedSpans, Completion]:
        file_context_str = file_context.create_prompt(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
        )

        content = instructions

        content += f"\n\nIdentify the relevant code sections from:\n{file_context_str}"
        identify_message = UserMessage(content=content)

        messages = [identify_message]
        completion = None

        MAX_RETRIES = 3
        for retry_attempt in range(MAX_RETRIES):
            identified_code, completion = self.completion_model.create_completion(
                messages=messages,
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                response_model=Identify,
            )
            logger.info(
                f"Identifying relevant code sections. Attempt {retry_attempt + 1} of {MAX_RETRIES}.\n{identified_code.identified_spans}"
            )

            view_context = FileContext(repo=self._repository)
            if identified_code.identified_spans:
                for identified_spans in identified_code.identified_spans:
                    view_context.add_line_span_to_context(
                        identified_spans.file_path,
                        identified_spans.start_line,
                        identified_spans.end_line,
                        add_extra=True,
                    )
            else:
                return view_context, completion

            tokens = view_context.context_size()

            if tokens > self.max_identify_tokens:
                logger.info(
                    f"Identified code sections are too large ({tokens} tokens)."
                )

                messages.append(
                    AssistantMessage(content=identified_code.model_dump_json())
                )

                messages.append(
                    UserMessage(
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_search_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections."
                    )
                )
            else:
                logger.info(
                    f"Identified code sections are within the token limit ({tokens} tokens)."
                )
                return view_context, completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {self.max_search_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
        )
