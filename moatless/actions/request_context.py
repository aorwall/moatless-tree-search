import logging
from typing import List, Optional

from pydantic import Field, BaseModel, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, FewShotExample, Observation, RewardScaleEntry
from moatless.codeblocks import CodeBlockType
from moatless.file_context import FileContext, ContextFile
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class CodeSpan(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    start_line: Optional[int] = Field(
        None, description="The start line of the code to add to context."
    )
    end_line: Optional[int] = Field(
        None, description="The end line of the code to add to context."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    @property
    def log_name(self):
        log = self.file_path

        if self.start_line and self.end_line:
            log += f" {self.start_line}-{self.end_line}"

        if self.span_ids:
            log += f" {', '.join(self.span_ids)}"

        return log


class RequestMoreContextArgs(ActionArguments):
    """Request additional code spans to be added to your current context."""

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")
    files: List[CodeSpan] = Field(
        ..., description="The code that should be provided in the file context."
    )

    class Config:
        title = "RequestMoreContext"

    @property
    def log_name(self):
        if len(self.files) == 1:
            return f"RequestMoreContext({self.files[0].log_name})"
        else:
            logs = []
            for i, file in enumerate(self.files):
                logs.append(f"{i}=[{file.log_name}]")
            return f"RequestMoreContext(" + ", ".join(logs) + ")"

    def to_prompt(self):
        prompt = "Requesting more context for the following files:\n"
        for file in self.files:
            prompt += f"* {file.file_path}\n"
            if file.start_line and file.end_line:
                prompt += f"  Lines: {file.start_line}-{file.end_line}\n"
            if file.span_ids:
                prompt += f"  Spans: {', '.join(file.span_ids)}\n"
        return prompt


class RequestMoreContext(Action):
    args_schema = RequestMoreContextArgs

    _repository: Repository = PrivateAttr()

    def __init__(self, repository: Repository | None = None, **data):
        super().__init__(**data)
        self._repository = repository

    # TODO?
    _max_tokens_in_edit_prompt = 750

    def execute(
        self, args: RequestMoreContextArgs, file_context: FileContext
    ) -> Observation:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the search action."
            )

        properties = {"files": {}}
        message = ""
        found_files = set()

        for file_with_spans in args.files:
            file = file_context.get_file(file_with_spans.file_path)

            if not file:
                logger.info(
                    f"{file_with_spans.file_path} is not found in the file repository."
                )
                message += f"The requested file {file_with_spans.file_path} is not found in the file repository. Use the search functions to search for the code if you are unsure of the file path."
                continue

            if self._repository.is_directory(file.file_path):
                logger.info(
                    f"{file_with_spans.file_path} is a directory and not a file."
                )
                message += f"The requested file {file_with_spans.file_path} is a directory and not a file. Use the search functions to search for the code if you are unsure of the file path."
                continue

            if file_with_spans.start_line and file_with_spans.end_line:
                file_context.add_line_span_to_context(
                    file.file_path, file_with_spans.start_line, file_with_spans.end_line
                )
            elif not file_with_spans.span_ids and not file.module:
                message += f"Please provide the line numbers to add to context."

            elif not file_with_spans.span_ids:
                message += self.create_retry_message(
                    file,
                    f"Please provide the line numbers or span ids for the code to add to context.",
                )

            missing_span_ids = set()
            suggested_span_ids = set()
            found_span_ids = set()
            if file_with_spans.span_ids and not file.module:
                logger.warning(
                    f"Tried to add span ids {file_with_spans.span_ids} to not parsed file {file.file_path}."
                )
                message += self.create_retry_message(
                    file, f"No span ids found. Is it empty?"
                )
                return Observation(
                    message=message, properties=properties, expect_correction=False
                )

            for span_id in file_with_spans.span_ids:
                block_span = file.module.find_span_by_id(span_id)
                if not block_span:
                    # Try to find the relevant code block by code block identifier
                    block_identifier = span_id.split(".")[-1]
                    blocks = file.module.find_blocks_with_identifier(block_identifier)

                    if not blocks:
                        missing_span_ids.add(span_id)
                    elif len(blocks) > 1:
                        for block in blocks:
                            if block.belongs_to_span.span_id not in suggested_span_ids:
                                suggested_span_ids.add(block.belongs_to_span.span_id)
                    else:
                        block_span = blocks[0].belongs_to_span

                if block_span:
                    if block_span.initiating_block.type == CodeBlockType.CLASS:
                        class_block = block_span.initiating_block
                        found_span_ids.add(block_span.span_id)
                        if class_block.sum_tokens() < self._max_tokens_in_edit_prompt:
                            for child_span_id in class_block.span_ids:
                                found_span_ids.add(child_span_id)
                    else:
                        found_span_ids.add(block_span.span_id)

            if not found_span_ids and suggested_span_ids:
                logger.info(f"Suggested spans: {', '.join(suggested_span_ids)}")
                message = self.create_retry_message(
                    file,
                    f"Spans not found. Did you mean one of these spans: {', '.join(suggested_span_ids)}\n",
                )

            if found_span_ids:
                message += f"\nAdded the following spans from {file.file_path} to context:\n{', '.join(found_span_ids)}"

            if file_with_spans.start_line and file_with_spans.end_line:
                file_context.add_line_span_to_context(
                    file_with_spans.file_path,
                    file_with_spans.start_line,
                    file_with_spans.end_line,
                )
                found_files.add(file_with_spans.file_path)

            for span_id in file_with_spans.span_ids:
                if not file_context.has_span(file_with_spans.file_path, span_id):
                    file_context.add_span_to_context(file_with_spans.file_path, span_id)
                    found_files.add(file_with_spans.file_path)

            if missing_span_ids:
                logger.info(
                    f"Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}"
                )
                message += self.create_retry_message(
                    file, f"Spans not found: {', '.join(missing_span_ids)}"
                )

            properties["files"][file_with_spans.file_path] = {
                "missing_span_ids": list(missing_span_ids),
                "found_span_ids": list(found_span_ids),
            }

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

        # TODO: Determine which scenarios where we should expect a correction
        return Observation(
            message=message, extra=extra, properties=properties, expect_correction=False
        )

    def create_retry_message(self, file: ContextFile, message: str):
        retry_message = f"\n\nProblems when trying to find spans in {file.file_path}. "
        retry_message += message

        hint = self.create_hint(file)
        if hint:
            retry_message += f"\n\n{hint}"

        if file.module and file.span_ids:
            retry_message += (
                f"\n\nAvailable span ids:\n{self.span_id_list(file.module.span_ids)}"
            )

        return retry_message

    def create_hint(self, file: ContextFile):
        if "test" in file.file_path:
            return "If you want to write a new test method, start by adding one of the existing ones that might relevant for reference."

        return None

    def span_id_list(self, span_ids: set[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = [
            "Relevance of Requested Context: Ensure that the requested context is directly related to the problem and necessary for making progress.",
            "Avoiding Hallucinations: Verify that the agent is requesting context for code that actually exists in the codebase.",
            "Efficiency: Assess whether the agent is requesting an appropriate amount of context without overloading unnecessary information.",
            "Appropriateness of Action: Evaluate if requesting more context is logical at this point in the problem-solving process.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The requested context is highly relevant, precise, and necessary for solving the problem; the agent avoids hallucinations.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The requested context is relevant and helpful, with minor issues in specificity or relevance.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The requested context is somewhat relevant but may include unnecessary information or lacks specificity.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The requested context has minimal relevance or includes excessive unnecessary information.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The requested context is irrelevant, demonstrates misunderstanding, or the agent is hallucinating code that doesn't exist.",
            ),
        ]

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="I need to see the implementation of the authenticate method in the AuthenticationService class",
                action=RequestMoreContextArgs(
                    scratch_pad="To understand the authentication implementation, we need to examine the authenticate method within the AuthenticationService class.",
                    files=[
                        CodeSpan(
                            file_path="auth/service.py",
                            span_ids=["AuthenticationService.authenticate"]
                        )
                    ]
                )
            ),
            FewShotExample.create(
                user_input="Show me lines 50-75 of the database configuration file",
                action=RequestMoreContextArgs(
                    scratch_pad="To examine the database configuration settings, we'll look at the specified line range in the config file.",
                    files=[
                        CodeSpan(
                            file_path="config/database.py",
                            start_line=50,
                            end_line=75
                        )
                    ]
                )
            )
        ]
