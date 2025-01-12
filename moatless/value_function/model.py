from typing import Optional

from pydantic import Field

from moatless.completion.model import StructuredOutput


class Reward(StructuredOutput):
    """A structured output for providing reward values and feedback for actions."""

    class Config:
        title = "ProvideReward"

    explanation: Optional[str] = Field(
        default=None,
        description="An explanation and the reasoning behind your decision.",
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
        ge=-100,
        le=100,
    )
