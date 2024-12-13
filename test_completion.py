import litellm
import os
from pydantic import BaseModel, Field
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.DEBUG)

# This should match your value function's response model
class ValueFunctionResponse(BaseModel):
    reward: float = Field(..., description="Reward value between 0 and 1")
    reasoning: Optional[str] = Field(None, description="Reasoning for the reward value")

    @classmethod
    def from_response(cls, completion_response):
        # Extract the content from the response
        content = completion_response.choices[0].message.content
        
        # Try to convert directly to float if it's a simple number
        try:
            reward = float(content)
            return cls(reward=reward)
        except ValueError:
            # If it's not a simple number, you might need more complex parsing
            # Add your parsing logic here
            return cls(reward=0.0, reasoning="Failed to parse response")

def test_value_function():
    try:
        messages = [
            {"role": "system", "content": "You are evaluating code changes. Provide a reward value between 0 and 1."},
            {"role": "user", "content": "Evaluate this change: print('hello world')"}
        ]
        
        print("\nSending request to model...")
        completion = litellm.completion(
            model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            temperature=0.7,
            base_url=os.getenv("CUSTOM_LLM_API_BASE"),
            api_key=os.getenv("CUSTOM_LLM_API_KEY")
        )
        
        print("\nRaw response:")
        print("-------------")
        print(completion.choices[0].message.content)
        
        # Try to parse with the response model
        print("\nParsing with ValueFunctionResponse:")
        print("----------------------------------")
        response = ValueFunctionResponse.from_response(completion)
        print(f"Parsed response: {response.model_dump_json(indent=2)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")

if __name__ == "__main__":
    test_value_function() 