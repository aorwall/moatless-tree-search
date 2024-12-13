import litellm
import os
from pydantic import BaseModel, Field
from typing import Optional

class StructuredOutput(BaseModel):
    """Base class for structured outputs from the model."""
    
    @classmethod
    def from_response(cls, completion_response):
        """Parse the completion response into a structured output."""
        content = completion_response.choices[0].message.content
        try:
            # Try to parse as JSON first
            import json
            data = json.loads(content)
            return cls.model_validate(data)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Content: {content}")
            # If not JSON, try simple number for Reward case
            if cls.__name__ == "Reward":
                try:
                    value = int(float(content))
                    return cls(value=value)
                except ValueError:
                    print(f"Failed to parse as number. Content: {content}")
            return None

class Reward(StructuredOutput):
    """A structured output for providing reward values and feedback for actions."""
    explanation: Optional[str] = Field(
        None, description="An explanation and the reasoning behind your decision."
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence"
    )

def test_value_function():
    try:
        messages = [
            {"role": "system", "content": """You are evaluating code changes. 
You must respond with a JSON object containing:
- value: An integer between -100 and 100
- explanation: (optional) Your reasoning
- feedback: (optional) Feedback for alternative approaches

Example response:
{
    "value": 50,
    "explanation": "The change looks good because...",
    "feedback": "Consider also..."
}"""},
            {"role": "user", "content": "Evaluate this change: print('hello world')"}
        ]
        
        print("\nSending request to model...")
        completion = litellm.completion(
            model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            temperature=0.7,
            base_url=os.getenv("CUSTOM_LLM_API_BASE"),
            api_key=os.getenv("CUSTOM_LLM_API_KEY"),
            response_format={"type": "json_object"}  # Request JSON response
        )
        
        print("\nRaw response:")
        print("-------------")
        print(completion.choices[0].message.content)
        
        # Try to parse with Reward model
        print("\nParsing with Reward model:")
        print("-------------------------")
        reward = Reward.from_response(completion)
        print(f"Parsed reward: {reward.model_dump_json(indent=2)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")

if __name__ == "__main__":
    test_value_function()