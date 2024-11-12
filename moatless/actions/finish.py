from typing import ClassVar, List

from litellm import Type
from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, RewardScaleEntry, FewShotExample
from moatless.file_context import FileContext


class FinishArgs(ActionArguments):
    """Indicate that the task is fully completed."""

    scratch_pad: str = Field(
        ..., description="Your reasoning about why the task is complete."
    )
    finish_reason: str = Field(..., description="Explanation of completion.")

    class Config:
        title = "Finish"

    def to_prompt(self):
        return f"Finish with reason: {self.finish_reason}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, FinishArgs)

class Finish(Action):
    args_schema: ClassVar[Type[ActionArguments]] = FinishArgs

    def execute(self, args: FinishArgs, file_context: FileContext | None = None):
        return Observation(message=args.finish_reason, terminal=True)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> List[str]:
        return [
            "**Solution Correctness and Quality:** Verify that the proposed changes logically address the problem statement. Ensure the changes fit contextually within the existing codebase without introducing new issues. Confirm syntactic correctness and that there are no syntax errors or typos. Assess whether the solution represents an overall improvement and is the most optimal approach possible.",
            "**Accuracy of Code Modifications:** Check that the agent correctly identified the appropriate code spans to modify by reviewing the provided Git diff. Ensure the changes made are accurate and do not include unintended modifications. Look for any alterations to unrelated parts of the code that could introduce new problems.",
            "**Testing and Test Results Analysis:**",
            " * **Importance of Test Updates:** It is crucial that the agent updated existing tests or added new tests to verify the solution. Failure to do so should be heavily penalized. The agent should ensure that code changes are validated by appropriate tests to confirm correctness and prevent regressions.",
            " * **Assess Test Coverage:** Evaluate whether the agent has adequately tested the solution, including adding new tests for new functionality or changes. Verify that the tests cover relevant cases and edge conditions.",
            " * **Penalization for Lack of Testing:** When calculating the reward, heavily penalize the agent if they failed to update or add necessary tests to verify the solution.",
            "**Consideration of Alternative Approaches:** Always assess whether there could be a better alternative approach to the problem. Mention any potential alternative solutions in your explanation if they are applicable.",
            "**Identification and Explanation of Mistakes:** If the agent made incorrect actions, identify exactly where and why the mistakes occurred. Explain the impact of any syntax errors, incorrect code modifications, or unintended changes.",
            "**Assessment of Agent's Completion Assertion:** Verify if the agent's assertion that the task is finished is accurate. Determine if substantial work is still required to fully resolve the issue and address this in your evaluation.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The action fully resolves the issue with optimal code modifications, includes updated or added tests that cover all possible scenarios, and ensures no side effects or dependency issues. All tests pass successfully.",
                ),
                (
                    75,
                    89,
                    "The action significantly advances the solution with correct code modifications and includes updated or added tests. Minor improvements or additional testing might be needed to achieve full coverage.",
                ),
                (
                    50,
                    74,
                    "The action contributes positively towards solving the problem with partial code modifications and some test coverage. There are notable issues with test completeness or potential side effects.",
                ),
                (
                    25,
                    49,
                    "The action is acceptable but has several issues, such as incomplete code modifications, insufficient test coverage, or minor side effects introduced.",
                ),
                (
                    0,
                    24,
                    "The action has minimal impact or introduces minor negative consequences, such as failing tests or small syntax errors. The agent's assertion that the task is finished is incorrect.",
                ),
                (
                    -49,
                    -1,
                    "The action is inappropriate or shows a lack of progress, such as incorrect code modifications, failure to update or add necessary tests, or introducing syntax errors.",
                ),
                (
                    -100,
                    -50,
                    "The action is entirely incorrect, causing significant new problems or failing to address the original issue. Immediate and comprehensive changes are necessary.",
                ),
            ]
        )

    @classmethod
    def get_value_function_prompt(cls) -> str:
        return """Your role is to evaluate the executed action of the search tree that our AI agents are traversing, with the goal of ensuring that a complete and verified solution is in place. The agent believes that it has finished solving the programming issue."""

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="""Applied the change to tests/auth_tests/test_validators.py

```diff
--- tests/auth_tests/test_validators.py
+++ tests/auth_tests/test_validators.py
@@ -238,6 +238,7 @@
             "o'connell", "عبد ال",
             "zerowidth\u200Bspace", "nonbreaking\u00A0space",
             "en\u2013dash",
+            "\njoe", "\nRené", "\nUSERNAME", "أحمد\n",
         ]
         v = validators.UnicodeUsernameValidator()
         for valid in valid_usernames:
@@ -250,7 +251,7 @@

     def test_ascii_validator(self):
         valid_usernames = ['glenn', 'GLEnN', 'jean-marc']
-        invalid_usernames = ["o'connell", 'Éric', 'jean marc', "أحمد"]
+        invalid_usernames = ["o'connell", 'Éric', 'jean marc', "أحمد", "\nglenn", "\nGLEnN", "\njean-marc", "USERNAME\n"]
         v = validators.ASCIIUsernameValidator()
         for valid in valid_usernames:
             with self.subTest(valid=valid):
             
Running 45 tests in the following files:
 * tests/calculator/test_operations.py
 * tests/calculator/test_parser.py

45 passed. 0 failed. 0 errors.
""",
                action=FinishArgs(
                    scratch_pad="The implementation has been completed in two steps: First, the username validators were updated to handle leading and trailing newlines correctly. Then, the test suite was enhanced to verify this functionality. The test additions cover both Unicode and ASCII username validators with comprehensive edge cases. All tests are now passing, confirming both the implementation and test coverage are correct.",
                    finish_reason="Username validators now properly handle leading and trailing newlines in usernames, with comprehensive test coverage verifying the functionality."
                )
            )
        ]
