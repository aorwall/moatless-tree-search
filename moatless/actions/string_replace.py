import logging
from pathlib import Path
from typing import Optional, List

from pydantic import Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.completion.model import ToolCall
from moatless.file_context import FileContext
from moatless.repository.file import do_diff
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.index.code_index import CodeIndex

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4

class StringReplaceArgs(ActionArguments):
    """
    Replace text in a file with exact string matching.
    
    Notes:
    * The old_str parameter must match EXACTLY one or more consecutive lines from the original file
    * Whitespace and indentation must match exactly
    * The old_str must be unique within the file - include enough surrounding context to ensure uniqueness
    * The new_str parameter contains the replacement text that will replace old_str
    * No changes will be made if old_str appears multiple times or cannot be found
    """
    path: str = Field(..., description="Path to the file to edit")
    old_str: str = Field(..., description="Exact string from the file to replace - must match exactly and be unique")
    new_str: str = Field(..., description="New string to replace the old_str with")

    class Config:
        title = "StringReplace"


class StringReplace(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to replace strings in a file.
    """
    args_schema = StringReplaceArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        # Initialize mixin attributes directly
        object.__setattr__(self, '_runtime', runtime)
        object.__setattr__(self, '_code_index', code_index)
        object.__setattr__(self, '_repository', repository)

    def execute(self, args: StringReplaceArgs, file_context: FileContext) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        old_str = args.old_str.expandtabs()
        new_str = args.new_str.expandtabs()

        if old_str == new_str:
            return Observation(
                message="The replacement string is the same as the original string. No changes were made.",
                properties={"fail_reason": "no_changes"},
                expect_correction=True,
            )

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            new_str_occurrences = file_content.count(new_str)
            if new_str_occurrences > 0:
                return Observation(
                    message=f"New string '{new_str}' already exists in {path}. No changes were made.",
                    properties={"fail_reason": "string_already_exists"}
                )

            return Observation(
                message=f"String '{old_str}' not found in {path}",
                properties={"fail_reason": "string_not_found"},
                expect_correction=True,
            )
        elif occurrences > 1:
            # Find line numbers for each occurrence
            lines = []
            pos = 0
            while True:
                pos = file_content.find(old_str, pos)
                if pos == -1:
                    break
                # Count newlines before this occurrence to get line number
                line_number = file_content.count('\n', 0, pos) + 1
                lines.append(line_number)
                pos += len(old_str)

            return Observation(
                message=f"Multiple occurrences of string found at lines {lines}",
                properties={"fail_reason": "multiple_occurrences"},
                expect_correction=True,
            )

        # Find the line numbers of the change
        change_pos = file_content.find(old_str)
        start_line = file_content.count('\n', 0, change_pos) + 1
        end_line = start_line + old_str.count('\n')

        # Check if the lines to be modified are in context
        if not context_file.lines_is_in_context(start_line, end_line):
            return Observation(
                message=f"The lines {start_line}-{end_line} are not in context. Please add them using RequestMoreContext.",
                properties={"fail_reason": "lines_not_in_context"},
                expect_correction=True,
            )

        new_file_content = file_content.replace(old_str, new_str)
        diff = do_diff(str(path), file_content, new_file_content)
        
        context_file.apply_changes(new_file_content)

        # Create a snippet of the edited section
        start_line = max(0, start_line - SNIPPET_LINES - 1)
        end_line = start_line + SNIPPET_LINES + new_str.count('\n')
        snippet = "\n".join(new_file_content.split("\n")[start_line:end_line])

        # Format the snippet with line numbers
        snippet_with_lines = self.format_snippet_with_lines(snippet, start_line + 1)

        success_msg = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of {path}:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )

        observation = Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )

        return self.run_tests_and_update_observation(
            observation=observation,
            file_path=str(path),
            scratch_pad=args.scratch_pad,
            file_context=file_context
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Update the error message in the validate_user method",
                action=StringReplaceArgs(
                    scratch_pad="Improving the error message to be more descriptive",
                    path="auth/validator.py",
                    old_str='''    if not user.is_active:
        raise ValueError("Invalid user")
    return user''',
                    new_str='''    if not user.is_active:
        raise ValueError(f"Invalid user: {username} does not meet the required criteria")
    return user'''
                )
            ),
            FewShotExample.create(
                user_input="Update the logging configuration",
                action=StringReplaceArgs(
                    scratch_pad="Enhancing the logging configuration with more detailed format and file handler",
                    path="utils/logger.py",
                    old_str='''logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)''',
                    new_str='''logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)'''
                )
            ),
            FewShotExample.create(
                user_input="Update the user validation logic",
                action=StringReplaceArgs(
                    scratch_pad="Adding email validation and password strength check",
                    path="auth/validator.py",
                    old_str='''def validate_user(username, password):
    if len(username) < 3:
        return False
    if len(password) < 8:
        return False
    return True''',
                    new_str='''def validate_user(username, password):
    if len(username) < 3 or not is_valid_email(username):
        return False
    if len(password) < 12 or not has_special_chars(password):
        return False
    if not has_numbers(password):
        return False
    return True'''
                )
            )
        ]
