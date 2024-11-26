import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_serializer

from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from moatless.actions.view_diff import ViewDiffArgs
from moatless.completion.model import Message, UserMessage, AssistantMessage
from moatless.actions.model import ActionArguments
from moatless.file_context import FileContext
from moatless.node import Node
from moatless.schema import MessageHistoryType
from moatless.utils.tokenizer import count_tokens
from testbeds.schema import TestStatus

logger = logging.getLogger(__name__)


class MessageHistoryGenerator(BaseModel):
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Type of message history to generate",
    )
    include_file_context: bool = Field(
        default=True,
        description="Whether to include file context in messages"
    )
    include_git_patch: bool = Field(
        default=True,
        description="Whether to include git patch in messages"
    )
    max_tokens: int = Field(
        default=20000,
        description="Maximum number of tokens allowed in message history"
    )

    model_config = {
        "ser_json_timedelta": "iso8601",
        "ser_json_bytes": "base64",
        "ser_json_inf_nan": "null",
        "json_schema_serialization_defaults": True,
        "json_encoders": None,  # Remove this as it's v1 syntax
    }

    @field_serializer('message_history_type')
    def serialize_message_history_type(self, message_history_type: MessageHistoryType) -> str:
        return message_history_type.value

    def generate(self, node: "Node") -> List[Message]:  # type: ignore
        previous_nodes = node.get_trajectory()[:-1]
        if not previous_nodes:
            return []

        logger.info(
            f"Generating message history for Node{node.node_id}: {self.message_history_type}"
        )

        generators = {
            MessageHistoryType.SUMMARY: self._generate_summary_history,
            MessageHistoryType.REACT: self._generate_react_history,
            MessageHistoryType.MESSAGES: self._generate_message_history
        }

        return generators[self.message_history_type](node, previous_nodes)

    def _generate_message_history(self, node: "Node", previous_nodes: List["Node"]) -> List[Message]:
        messages = [UserMessage(content=node.get_root().message)]

        if len(previous_nodes) <= 1:
            return messages

        node_messages = self.get_node_messages(node)

        for action, observation in node_messages:
            thoughts = (
                f"<thoughts>{action.thoughts}</thoughts>"
                if hasattr(action, "thoughts")
                else ""
            )

            messages.append(AssistantMessage(tool_call=action.to_tool_call(thoughts_in_action=True)))
            messages.append(UserMessage(content=f"<observation>\n{observation}\n</observation>"))

        tokens = count_tokens("".join([m.content for m in messages if m.content is not None]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages

    def _generate_react_history(self, node: "Node", previous_nodes: List["Node"]) -> List[Message]:
        messages = [UserMessage(content=node.get_root().message)]
        
        if len(previous_nodes) <= 1:
            return messages

        node_messages = self.get_node_messages(node)
        
        # Convert node messages to react format
        for action, observation in node_messages:
            # Add thought and action message
            thought = (
                f"Thought: {action.thoughts}"
                if hasattr(action, "thoughts")
                else ""
            )
            action_str = f"Action: {action.name}"
            action_input = action.format_args_for_llm()
            
            assistant_content = f"{thought}\n{action_str}"
            if action_input:
                assistant_content += f"\n{action_input}"
            
            messages.append(AssistantMessage(content=assistant_content))
            
            # Add observation message
            messages.append(UserMessage(content=f"Observation: {observation}"))

        tokens = count_tokens("".join([m.content for m in messages if m.content is not None]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages

    def _generate_summary_history(self, node: Node, previous_nodes: List[Node]) -> List[Message]:
        formatted_history: List[str] = []
        counter = 0

        content = node.get_root().message

        if not previous_nodes:
            return [UserMessage(content=content)]

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = f"\n## {counter}. Action: {previous_node.action.name}\n"
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    if (
                        hasattr(previous_node.observation, "summary")
                        and previous_node.observation.summary
                        and i < len(previous_nodes) - 1
                    ):
                        formatted_state += f"\n\nObservation: {previous_node.observation.summary}"
                    else:
                        formatted_state += f"\n\nObservation: {previous_node.observation.message}"
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")
                    formatted_state += "\n\nObservation: No output found."

                formatted_history.append(formatted_state)

        content += "\n\nBelow is the history of previously executed actions and their observations.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"

        if self.include_file_context:
            content += "\n\nThe following code has already been viewed:\n"
            content += node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

        if self.include_git_patch:
            git_patch = node.file_context.generate_git_patch()
            if git_patch:
                content += "\n\nThe current git diff is:\n"
                content += "```diff\n"
                content += git_patch
                content += "\n```"

        return [UserMessage(content=content)]

    def get_node_messages(self, node: "Node") -> List[tuple[ActionArguments, str]]:
        """
        Creates a list of (action, observation) tuples from the node's trajectory.
        Respects token limits while preserving ViewCode context.
        
        Returns:
            List of tuples where each tuple contains:
            - ActionArguments object
            - Observation summary string
        """
        previous_nodes = node.get_trajectory()[:-1]
        if not previous_nodes:
            return []

        # Calculate initial token count
        total_tokens = node.file_context.context_size()
        total_tokens += count_tokens(node.get_root().message)

        # Pre-calculate test output tokens if there's a patch
        test_output_tokens = 0
        test_output = None
        run_tests_args = None
        if node.file_context.has_patch():
            run_tests_args = RunTestsArgs(
                thoughts=f"Run the tests to verify the changes.",
                test_files=list(node.file_context._test_files.keys())
            )
            
            test_output = ""
            
            # Add failure details if any
            failure_details = node.file_context.get_test_failure_details()
            if failure_details:
                test_output += failure_details + "\n\n"
            
            test_output += node.file_context.get_test_summary()
            test_output_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
            total_tokens += test_output_tokens

        node_messages = []
        shown_files = set()
        shown_diff = False  # Track if we've shown the first diff
        last_test_status = None  # Track the last test status

        for i, previous_node in enumerate(reversed(previous_nodes)):
            current_messages = []
            
            if previous_node.action:
                if previous_node.action.name == "ViewCode":
                    # Always include ViewCode actions
                    file_path = previous_node.action.files[0].file_path

                    if file_path not in shown_files:
                        context_file = previous_node.file_context.get_context_file(file_path)
                        if context_file and (context_file.span_ids or context_file.show_all_spans):
                            shown_files.add(context_file.file_path)
                            observation = context_file.to_prompt(
                                show_span_ids=False,
                                show_line_numbers=True,
                                exclude_comments=False,
                                show_outcommented_code=True,
                                outcomment_code_comment="... rest of the code",
                            )
                        else:
                            observation = previous_node.observation.message
                        current_messages.append((previous_node.action, observation))
                else:
                    # Count tokens for non-ViewCode actions
                    observation_str = (
                        previous_node.observation.summary
                        if self.include_file_context and hasattr(previous_node.observation, "summary") and previous_node.observation.summary
                        else previous_node.observation.message if previous_node.observation
                        else "No output found."
                    )

                    # Calculate tokens for this message pair
                    action_tokens = count_tokens(previous_node.action.model_dump_json())
                    observation_tokens = count_tokens(observation_str)
                    message_tokens = action_tokens + observation_tokens
                    
                    # Only add if within token limit
                    if total_tokens + message_tokens <= self.max_tokens:
                        total_tokens += message_tokens
                        current_messages.append((previous_node.action, observation_str))                       
                    else:
                        # Skip remaining non-ViewCode messages if we're over the limit
                        continue

                # Handle file context for non-ViewCode actions
                if self.include_file_context and previous_node.action.name != "ViewCode":
                    files_to_show = set()
                    has_edits = False
                    for context_file in previous_node.file_context.get_context_files():
                        if (context_file.was_edited or context_file.was_viewed) and context_file.file_path not in shown_files:
                            files_to_show.add(context_file.file_path)
                        if context_file.was_edited:
                            has_edits = True
                    
                    shown_files.update(files_to_show)
                    
                    if files_to_show:
                        # Batch all files into a single ViewCode action
                        code_spans = []
                        observations = []
                        
                        for file_path in files_to_show:
                            context_file = previous_node.file_context.get_context_file(file_path)
                            if context_file.show_all_spans:
                                code_spans.append(CodeSpan(file_path=file_path))
                            elif context_file.span_ids:
                                code_spans.append(CodeSpan(file_path=file_path, span_ids=context_file.span_ids))
                            else:
                                continue
                                
                            observations.append(context_file.to_prompt(
                                show_span_ids=False,
                                show_line_numbers=True,
                                exclude_comments=False,
                                show_outcommented_code=True,
                                outcomment_code_comment="... rest of the code",
                            ))
                        
                        if code_spans:
                            thought = f"Let's view the content in the updated files"
                            args = ViewCodeArgs(files=code_spans, thoughts=thought)
                            current_messages.append((args, "\n\n".join(observations)))

                    # Show ViewDiff on first edit
                    if has_edits and self.include_git_patch and not shown_diff:
                        patch = node.file_context.generate_git_patch()
                        if patch:
                            view_diff_args = ViewDiffArgs(
                                thoughts="Let's review the changes made to ensure we've properly implemented everything required for the task. I'll check the git diff to verify the modifications."
                            )
                            diff_tokens = count_tokens(patch) + count_tokens(view_diff_args.model_dump_json())
                            if total_tokens + diff_tokens <= self.max_tokens:
                                total_tokens += diff_tokens
                                current_messages.append((view_diff_args, f"Current changes in workspace:\n```diff\n{patch}\n```"))
                                shown_diff = True

                    # Add test results only if status changed or first occurrence
                    if (previous_node.observation and  
                        previous_node.observation.properties.get("diff")):
                        
                        current_test_status = node.file_context.get_test_status()
                        if last_test_status is None or current_test_status != last_test_status:
                            run_tests_args = RunTestsArgs(
                                thoughts=f"Run the tests to verify the changes.",
                                test_files=list(node.file_context._test_files.keys())
                            )
                            
                            test_output = ""
                            if last_test_status is None:
                                # Show full details for first test run
                                failure_details = node.file_context.get_test_failure_details()
                                if failure_details:
                                    test_output += failure_details + "\n\n"
                            
                            test_output += node.file_context.get_test_summary()
                            
                            # Calculate and check token limits
                            test_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
                            if total_tokens + test_tokens <= self.max_tokens:
                                total_tokens += test_tokens
                                current_messages.append((run_tests_args, test_output))
                                last_test_status = current_test_status

            # Add current messages to the beginning of the list
            node_messages = current_messages + node_messages

        logger.info(f"Generated message history with {total_tokens} tokens")
        return node_messages
