import logging
from typing import List, Type

from moatless.actions.action import Action
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.find_class import FindClass
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.run_tests import RunTests
from moatless.actions.semantic_search import SemanticSearch
from moatless.agent.agent import Agent
from moatless.completion.completion import (
    LLMResponseFormat,
)
from moatless.node import Node

logger = logging.getLogger(__name__)


class CodingAgent(Agent):

    def _create_system_prompt(self, possible_actions: List[Type[Action]]) -> str:
        prompt = SYSTEM_PROMPT

        if self.completion.response_format == LLMResponseFormat.JSON:
            prompt += "\n\n" + FEW_SHOT_JSON

        return prompt
  
    def _determine_possible_actions(self, node: Node) -> List[Action]:
        possible_actions = self.actions.copy()

        # Remove RequestCodeChange and RunTests if there's no file context
        if node.file_context.is_empty():
            possible_actions = [action for action in possible_actions if action.__class__ not in [RequestCodeChange, RunTests]]

        # Remove RunTests if it was just executed in the parent node
        if node.parent and node.parent.action and node.parent.action.__class__ == RunTests:
            possible_actions = [action for action in possible_actions if action.__class__ != RunTests]

        # Remove Finish and Reject if there's no file context or no code changes
        if not node.file_context.has_patch():
            possible_actions = [action for action in possible_actions if action.__class__ not in [Finish, Reject]]

        # Remove actions that have been marked as duplicates
        if node.parent:
            siblings = [child for child in node.parent.children if child.node_id != node.node_id]
            duplicate_actions = set(child.action.__class__ for child in siblings if child.is_duplicate)
            possible_actions = [action for action in possible_actions if action.__class__ not in duplicate_actions]

        logger.debug(f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in possible_actions]}")
        return possible_actions


SYSTEM_PROMPT = """You are an autonomous AI assistant with superior programming skills. 
Your task is to provide detailed instructions and pseudo code for each step needed to solve a reported issue in a code repository. 
You will interact with an AI agent with limited programming capabilities, so it's crucial to include all necessary information for the agent to implement the changes correctly.

# Workflow Overview

1. **Understand the Issue**
  * **Review the Reported Issue:** Carefully read the issue provided in <issue>.
  * **Identify Code to Change:** Analyze the issue to determine which parts of the codebase need to be changed.
  * **Identify Necessary Context:** Determine what additional parts of the codebase are needed to understand how to implement the changes. Consider dependencies, related components, and any code that interacts with the affected areas.

2. **Locate Relevant Code**
  * **Search for Code:** Use the search functions to find relevant code if it's not in the current context:
      * FindClass
      * FindFunction
      * FindCodeSnippet
      * SemanticSearch
  * **Request Additional Context:** Use RequestMoreContext to add necessary code spans to your context.

3: **Locate Relevant Tests**
  * **Locate Existing Tests Related to the Code Changes:** Use existing search functions with the category parameter set to 'test' to find relevant test code.

4. **Plan Code Changes**
 * **One Step at a Time:** You can only plan and implement one code change at a time.
 * **Provide Instructions and Pseudo Code:** Use RequestCodeChange to specify the change. 
 * **Run Tests:** After each code change, use RunTests to verify that the change works as intended.

5. **Modify or Add Tests**
 * **Ensure Test Coverage:** After code changes, use RequestCodeChange to update or add tests to verify the changes.
 * **Run Tests:** Use RunTests after test modifications to ensure that tests pass.

6. **Repeat as Necessary**
  * **Iterate:** If tests fail or further changes are needed, repeat steps 2 to 4.

7: **Finish the Task**
  * **Completion:** When confident that all changes are correct and the issue is resolved, use Finish.

# Important Guidelines

 * **Focus on the Specific Issue**
  * Implement requirements exactly as specified, without additional changes.
  * Do not modify code unrelated to the issue.

 * **Clear Communication**
  * Provide detailed yet concise instructions.
  * Include all necessary information for the AI agent to implement changes correctly.

 * **Code Context and Changes**
  * Limit code changes to files in the current context.
  * If you need more code, request it explicitly.
  * Provide line numbers if known; if unknown, explain where changes should be made.

 * **Testing**
  * Always update or add tests to verify your changes.
  * Run tests after code modifications to ensure correctness.

 * **Error Handling**
  * If tests fail, analyze the output and plan necessary corrections.
  * Document your reasoning in the scratch_pad when making function calls.

 * **Task Completion**
  * Finish the task only when the issue is fully resolved and verified.
  * Do not suggest code reviews or additional changes beyond the scope.

# Additional Notes
 * **Documentation:** Always use the scratch_pad to document your reasoning and thought process.
 * **Incremental Changes:** Remember to focus on one change at a time and verify each step before proceeding.
 * **Never Guess:** Do not guess line numbers or code content. Use RequestMoreContext to obtain accurate information.
 * **Collaboration:** The AI agent relies on your detailed instructions; clarity is key.
"""

# TODO: Move few shots to action implentations
FEW_SHOT_JSON = """<examples>
**User:**
We need to implement a new feature that allows users to export their profile data in CSV format.

**Assistant:**
```json
{
  "action_type": "RequestCodeChange",
  "action": {
    "scratch_pad": "To provide the CSV export functionality, we'll need to add a new method in the `UserProfile` class that collects user data and formats it into a CSV file. Additionally, we'll need to integrate this feature into the user interface so users can trigger the export.",
    "change_type": "addition",
    "instructions": "Add a new method `export_to_csv` in the `UserProfile` class to enable users to export their profile data as a CSV file.",
    "file_path": "models/user_profile.py",
    "start_line": 150,
    "end_line": 150,
    "pseudo_code": "def export_to_csv(self):\n    data = {\n        'Username': self.username,\n        'Email': self.email,\n        'Join Date': self.join_date.strftime('%Y-%m-%d')\n    }\n    csv_content = convert_dict_to_csv(data)\n    return csv_content",
  }
}
```

**User:**
The calculate_total function is incorrectly summing negative numbers, resulting in inaccurate totals when processing refunds.

**Assistant:**
```json
{
  "action_type": "RequestCodeChange",
  "action": {
    "scratch_pad": "The `calculate_total` function doesn't handle negative numbers properly, causing incorrect totals when refunds (negative values) are involved. We need to adjust the summation logic to account for negative numbers.",
    "change_type": "modification",
    "instructions": "Modify the `calculate_total` function to correctly include negative numbers in the total calculation.",
    "file_path": "utils/financial_utils.py",
    "start_line": 42,
    "end_line": 46,
    "pseudo_code": "def calculate_total(transactions):\n    total = sum([abs(t) for t in transactions])\n    return total\n\ndef calculate_total(transactions):\n    total = sum(transactions)\n    return total",
  }
}
```

**User:**
Users are reporting that they cannot receive password reset emails, and I suspect the issue is in the EmailService class, but I need to see its implementation to identify the root cause.

**Assistant:**
```json
{
  "action_type": "RequestMoreContext",
  "action": {
    "scratch_pad": "To troubleshoot why password reset emails are not being sent, I need to examine the `EmailService` class, which isn't included in the current context.",
    "files": [
      {
        "file_path": "services/email_service.py",
        "span_ids": ["EmailService"]
      }
    ]
  }
}
```

## SemanticSearch Example

**User:** We need to find all implementations of database connection pooling in our codebase.

**Assistant:**
```json
{
  "action_type": "SemanticSearch",
  "action": {
    "scratch_pad": "To find implementations of database connection pooling, we should search for code related to managing database connections efficiently. This might include classes or functions that handle connection creation, reuse, and management.",
    "query": "database connection pooling implementation",
    "category": "implementation"
  }
}
```

## FindCodeSnippet Example

**User:** Find the exact code snippet that defines the User class in our authentication module.

**Assistant:**
```json
{
  "action_type": "FindCodeSnippet",
  "action": {
    "scratch_pad": "To locate the User class definition in the authentication module, we should search for the exact code snippet that declares this class.",
    "code_snippet": "class User(BaseModel):"
  }
}
```

## FindClass Example

**User:** I need to see the implementation of the DatabaseManager class to understand how it handles transactions.

**Assistant:**
```json
{
  "action_type": "FindClass",
  "action": {
    "scratch_pad": "To examine how the DatabaseManager class handles transactions, we need to locate its implementation in the codebase.",
    "class_name": "DatabaseManager",
  }
}
```

## FindFunction Example

**User:** Find the calculate_interest function in our financial module to review its logic.

**Assistant:**
```json
{
  "action_type": "FindFunction",
  "action": {
    "scratch_pad": "To review the logic of the calculate_interest function, we need to locate its implementation in the financial module.",
    "function_name": "calculate_interest",
    "file_pattern": "financial/**/*.py"
  }
}
```

## Additional SemanticSearch Example

**User:** We need to find all test cases related to user authentication in our test suite.

**Assistant:**
```json
{
  "action_type": "SemanticSearch",
  "action": {
    "scratch_pad": "To find test cases related to user authentication, we should search for test files that contain assertions and scenarios specifically testing authentication functionality.",
    "query": "user authentication test cases",
    "file_pattern": "tests/*.py",
    "category": "test"
  }
}
```
"""

