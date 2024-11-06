import logging
from pathlib import Path
from typing import Literal, Optional, List, Any

from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.completion.model import ToolCall
from moatless.file_context import FileContext
from moatless.repository.file import do_diff

logger = logging.getLogger(__name__)

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]

SNIPPET_LINES: int = 4


class EditActionArguments(ActionArguments):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    """

    command: Command = Field(..., description="The edit command to execute")
    path: str = Field(..., description="The file path to edit")
    file_text: Optional[str] = Field(None, description="The text content for file creation")
    view_range: Optional[List[int]] = Field(None, description="Range of lines to view [start, end]")
    old_str: Optional[str] = Field(None, description="String to replace")
    new_str: Optional[str] = Field(None, description="Replacement string")
    insert_line: Optional[int] = Field(None, description="Line number for insertion")

    class Config:
        title = "str_replace_editor"

    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, type="text_editor_20241022")

    # FIXME: This is a temporary fix to ignore scratch_pad in the validation
    @model_validator(mode='before')
    @classmethod
    def fix_scratch_pad(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("scratch_pad"):
                data["scratch_pad"] = ""

        return data

class ClaudeEditTool(Action):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    args_schema = EditActionArguments

    def execute(self, args: EditActionArguments, file_context: FileContext) -> Observation:
        # Claude tends to add /repo in the start of the file path.
        # TODO: Maybe we should add /repo as default on all paths?
        if args.path.startswith("/repo"):
            args.path = args.path[5:]

        # Remove leading `/` if present
        # TODO: Solve by adding /repo to all paths?
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        validation_error = self.validate_path(file_context, args.command, path)
        if validation_error:
            return Observation(
                message=validation_error,
                properties={"fail_reason": "invalid_path"},
                expect_correction=True,
            )

        if args.command == "view":
            return self._view(file_context, path, args.view_range)
        elif args.command == "create":
            if not args.file_text:
                return Observation(
                    message="Parameter `file_text` is required for command: create",
                    properties={"fail_reason": "missing_file_text"},
                    expect_correction=True,
                )
            return self._create(file_context, path, args.file_text)
        elif args.command == "str_replace":
            if not args.old_str:
                return Observation(
                    message="Parameter `old_str` is required for command: str_replace",
                    properties={"fail_reason": "missing_old_str"},
                    expect_correction=True,
                )
            return self._str_replace(file_context, path, args.old_str, args.new_str or "")
        elif args.command == "insert":
            if args.insert_line is None:
                return Observation(
                    message="Parameter `insert_line` is required for command: insert",
                    properties={"fail_reason": "missing_insert_line"},
                    expect_correction=True,
                )
            if args.new_str is None:
                return Observation(
                    message="Parameter `new_str` is required for command: insert",
                    properties={"fail_reason": "missing_new_str"},
                    expect_correction=True,
                )
            return self._insert(file_context, path, args.insert_line, args.new_str)

        return Observation(
            message=f"Unknown command: {args.command}",
            properties={"fail_reason": "unknown_command"},
            expect_correction=True,
        )

    def validate_path(self, file_context: FileContext, command: str, path: Path) -> str | None:
        """
        Check that the path/command combination is valid.
        """
        # TODO: Check if its an absolute path?
        #if not path.is_absolute():
        #    suggested_path = Path("") / path
        #    return (
        #        f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
        #    )

        # Check if path exists
        if not file_context._repo.file_exists(str(path)) and command != "create":
            return (
                f"The path {path} does not exist. Please provide a valid path."
            )

        if file_context._repo.file_exists(str(path)) and command == "create":
            return (
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )

        # Check if the path points to a directory
        if file_context._repo.is_directory(str(path)):
            if command != "view":
                return (
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

        return None

    def _view(self, file_context: FileContext, path: Path, view_range: Optional[List[int]]) -> Observation:
        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
                expect_correction=True,
            )

        file_content = context_file.content
        init_line = 1
        if view_range:
            if len(view_range) != 2:
                return Observation(
                    message="Invalid view_range. It should be a list of two integers.",
                    properties={"fail_reason": "invalid_view_range"},
                    expect_correction=True,
                )
                
            file_lines = file_content.split("\n")
            n_lines = len(file_lines)
            init_line, final_line = view_range
            
            if init_line < 1 or init_line > n_lines:
                return Observation(
                    message=f"Invalid view_range start line: {init_line}. Should be between 1 and {n_lines}",
                    properties={"fail_reason": "invalid_view_range"},
                    expect_correction=True,
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1:])
            else:
                file_content = "\n".join(file_lines[init_line - 1: final_line])

            file_context.add_line_span_to_context(str(path), init_line, final_line)

        output = self._make_output(file_content, str(path), init_line=init_line)
        return Observation(
            message=output,
            properties={"success": True},
        )

    def _create(self, file_context: FileContext, path: Path, file_text: str) -> Observation:
        if path.exists():
            return Observation(
                message=f"File already exists at: {path}",
                properties={"fail_reason": "file_exists"},
                expect_correction=True,
            )
        
        context_file = file_context.add_file(str(path))
        context_file.apply_changes(file_text)

        diff = do_diff(str(path), "", file_text)

        return Observation(
            message=f"File created successfully at: {path}",
            properties={"diff": diff, "success": True},
        )

    def _str_replace(self, file_context: FileContext, path: Path, old_str: str, new_str: str) -> Observation:
        SNIPPET_LINES = 4

        # Validate file exists and is not a directory
        if not file_context._repo.file_exists(str(path)):
            return Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
                expect_correction=True,
            )

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
                expect_correction=True,
            )
            
        file_content = context_file.content.expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs()

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            return Observation(
                message=f"String '{old_str}' not found in {path}",
                properties={"fail_reason": "string_not_found"},
                expect_correction=True,
            )
        elif occurrences > 1:
            lines = [i + 1 for i, line in enumerate(file_content.split("\n")) if old_str in line]
            return Observation(
                message=f"Multiple occurrences of '{old_str}' found in lines {lines}",
                properties={"fail_reason": "multiple_occurrences"},
                expect_correction=True,
            )

        new_file_content = file_content.replace(old_str, new_str)

        diff = do_diff(str(path), file_content, new_file_content)
        context_file.apply_changes(new_file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )

    def _insert(self, file_context: FileContext, path: Path, insert_line: int, new_str: str) -> Observation:
        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
                expect_correction=True,
            )

        # Validate file exists and is not a directory
        if not file_context._repo.file_exists(str(path)):
            return Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
                expect_correction=True,
            )
        file_text = context_file.content.expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > len(file_text_lines):
            return Observation(
                message=f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}",
                properties={"fail_reason": "invalid_line_number"},
                expect_correction=True,
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        diff = do_diff(str(path), file_text, new_file_text)
        context_file.apply_changes(new_file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )


TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000

def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )
