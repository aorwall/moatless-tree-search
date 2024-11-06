from typing import Optional
from instructor import OpenAISchema

from pydantic import BaseModel, Field


class Reward(OpenAISchema):
    explanation: Optional[str] = Field(
        None, description="An explanation and the reasoning behind your decision."
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
    )
