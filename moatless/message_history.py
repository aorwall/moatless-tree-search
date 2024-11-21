import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_serializer

from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from moatless.completion.model import Message, UserMessage, AssistantMessage
from moatless.actions.model import ActionArguments
from moatless.file_context import FileContext
from moatless.node import Node
from moatless.schema import MessageHistoryType
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ActionFormatter:
    @staticmethod
    def format_action(action: ActionArguments) -> str:
        """Format an action for message history display"""
        if action.name in ["StringReplace", "CreateFile", "InsertLines"]:
            return ActionFormatter._format_file_action(action)
        return f"Action Input: {action.model_dump_json(exclude={'scratch_pad'})}"

    @staticmethod
    def _format_file_action(action: ActionArguments) -> str:
        action_data = action.model_dump(exclude={'scratch_pad'})
        action_input = "Action Input:\n"
        
        if action.name == "StringReplace":
            action_input += f"<path>{action_data['path']}</path>\n"
            action_input += f"<old_str>\n{action_data['old_str']}\n</old_str>\n"
            action_input += f"<new_str>\n{action_data['new_str']}\n</new_str>"
        elif action.name == "CreateFile":
            action_input += f"<path>{action_data['path']}</path>\n"
            action_input += f"<file_text>\n{action_data['file_text']}\n</file_text>"
        elif action.name == "InsertLines":
            action_input += f"<path>{action_data['path']}</path>\n"
            action_input += f"<insert_line>{action_data['insert_line']}</insert_line>\n"
            action_input += f"<new_str>\n{action_data['new_str']}\n</new_str>"
        return action_input

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

    def _generate_react_history(self, node: "Node", previous_nodes: List["Node"]) -> List[Message]:
        messages = [UserMessage(content=node.get_root().message)]
        
        if len(previous_nodes) <= 1:
            return messages

        # Get node messages using the new method
        node_messages = self.get_node_messages(node)
        
        # Convert node messages to react format
        for action, observation in node_messages:
            # Add thought and action message
            thought = (
                f"Thought: {action.scratch_pad}"
                if hasattr(action, "scratch_pad")
                else ""
            )
            action_str = f"Action: {action.name}"
            action_input = ActionFormatter.format_action(action)
            
            assistant_content = f"{thought}\n{action_str}"
            if action_input:
                assistant_content += f"\n{action_input}"
            
            messages.append(AssistantMessage(content=assistant_content))
            
            # Add observation message
            messages.append(UserMessage(content=f"Observation: {observation}"))

        for message in messages:
            logger.info(f"{count_tokens(message.content)} Generated message: {message.content[:100]}")

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

    def _generate_message_history(self, node: Node, previous_nodes: List[Node]) -> List[Message]:
        messages: List[Message] = []
        last_file_updates = {}

        if self.include_file_context:
            # Track when each file was last modified
            for i, node in enumerate(previous_nodes):
                if not node.parent:
                    updated_files = set(
                        [file.file_path for file in node.file_context.get_context_files()]
                    )
                else:
                    updated_files = node.file_context.get_updated_files(
                        node.parent.file_context
                    )
                    for file in updated_files:
                        last_file_updates[file] = i

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.message:
                messages.append(UserMessage(content=previous_node.message))

            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

                content = ""
                if previous_node.observation:
                    if self.include_file_context and previous_node.observation.summary:
                        content += previous_node.observation.summary
                    else:
                        content += previous_node.observation.message

                messages.append(UserMessage(content=content))

            # Show file context for updated files
            self._add_file_context_messages(previous_node, i, last_file_updates, messages)

        if node.feedback:
            messages.append(UserMessage(content=node.feedback))

        return messages

    def _add_file_context_messages(
        self, 
        previous_node: Node,
        i: int, 
        last_file_updates: Dict[str, int], 
        messages: List[Message]
    ):
        if not previous_node.parent:
            updated_files = set(
                [file.file_path for file in previous_node.file_context.get_context_files()]
            )
        else:
            updated_files = previous_node.file_context.get_updated_files(
                previous_node.parent.file_context
            )

        files_to_show = set(
            [f for f in updated_files if last_file_updates.get(f) == i]
        )

        for file_path in files_to_show:
            context_file = previous_node.file_context.get_context_file(file_path)

            if context_file.show_all_spans:
                args = ViewCodeArgs(
                    scratch_pad=f"Let's view the content in {file_path}",
                    files=[CodeSpan(file_path=file_path)],
                )
            elif context_file.span_ids:
                args = ViewCodeArgs(
                    scratch_pad=f"Let's view the content in {file_path}",
                    files=[CodeSpan(file_path=file_path, span_ids=context_file.span_ids)],
                )
            else:
                continue

            messages.append(AssistantMessage(tool_call=args.to_tool_call()))
            messages.append(
                UserMessage(
                    content=context_file.to_prompt(
                        show_span_ids=False,
                        show_line_numbers=True,
                        exclude_comments=False,
                        show_outcommented_code=True,
                        outcomment_code_comment="... rest of the code",
                    )
                )
            )

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

        context_content = node.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        total_tokens = count_tokens(context_content)

        logger.info(f"Generating message history with {total_tokens} tokens")
        node_messages = []
        shown_files = set()
        
        for previous_node in reversed(previous_nodes):
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
                    if not previous_node.parent:
                        updated_files = set([file.file_path for file in previous_node.file_context.get_context_files()])
                    else:
                        updated_files = previous_node.file_context.get_updated_files(previous_node.parent.file_context)
                    
                    files_to_show = updated_files - shown_files
                    shown_files.update(files_to_show)
                    
                    for file_path in files_to_show:
                        context_file = previous_node.file_context.get_context_file(file_path)
                        thought = f"Let's view the content in {file_path}"
                        if context_file.show_all_spans:
                            args = ViewCodeArgs(files=[CodeSpan(file_path=file_path)], scratch_pad=thought)
                        elif context_file.span_ids:
                            args = ViewCodeArgs(files=[CodeSpan(file_path=file_path, span_ids=context_file.span_ids)], scratch_pad=thought)
                        else:
                            continue

                        observation = context_file.to_prompt(
                            show_span_ids=False,
                            show_line_numbers=True,
                            exclude_comments=False,
                            show_outcommented_code=True,
                            outcomment_code_comment="... rest of the code",
                        )
                        current_messages.append((args, observation))

            # Add current messages to the beginning of the list
            node_messages = current_messages + node_messages

        logger.info(f"Generated message history with {total_tokens} tokens")
        return node_messages