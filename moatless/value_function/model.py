from typing import Optional, Dict, Any
from pydantic import Field

from moatless.completion.model import StructuredOutput


class Reward(StructuredOutput):
    """A structured output for providing reward values and feedback for actions."""
    
    name: str = "provide_reward"  # Required for tool calling
    
    explanation: str = Field(  # Changed from Optional[str] to str with default
        default="No explanation provided",
        description="An explanation and the reasoning behind your decision."
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
        ge=-100,
        le=100
    )

    @classmethod
    def anthropic_schema(cls) -> Dict[str, Any]:
        """Provide schema in format expected by Anthropic's tool calling"""
        return {
            "type": "custom",
            "name": "provide_reward",
            "description": "Provide a reward assessment for the current state",
            "input_schema": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "A single integer value between -100 and 100 based on your confidence in the correctness of the action",
                        "minimum": -100,
                        "maximum": 100
                    },
                    "explanation": {
                        "type": "string",
                        "description": "An explanation and the reasoning behind your decision"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Feedback to the alternative branch"
                    }
                },
                "required": ["value", "explanation"]  # Added explanation as required
            }
        }

    @classmethod
    def openai_schema(cls, thoughts_in_action: bool = False) -> Dict[str, Any]:
        """Provide schema in format expected by OpenAI's function calling"""
        return {
            "name": "provide_reward",
            "description": "Provide a reward assessment for the current state",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "A single integer value between -100 and 100 based on your confidence in the correctness of the action",
                        "minimum": -100,
                        "maximum": 100
                    },
                    "explanation": {
                        "type": "string",
                        "description": "An explanation and the reasoning behind your decision"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Feedback to the alternative branch"
                    }
                },
                "required": ["value", "explanation"]  # Added explanation as required
            }
        }

    model_config = {
        "json_schema_extra": {
            "description": "A structured output for providing reward values and feedback for actions.",
            "examples": [{
                "value": 50,
                "explanation": "The action successfully identified the core issue and proposed a reasonable solution approach.",
                "feedback": "Consider exploring alternative test cases to ensure comprehensive coverage."
            }]
        }
    }