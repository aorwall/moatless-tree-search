from typing import Optional, Union, Dict, Any, List
from pydantic import Field, field_validator, BaseModel
from moatless.completion.model import StructuredOutput
import json
import logging
import re
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class Reward(StructuredOutput):
    """A structured output for providing reward values and feedback for actions."""
    
    name = "reward"  # Required by StructuredOutput for anthropic_schema()
    
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of eventually leading to resolving the issue",
        ge=-100,  # Minimum value constraint
        le=100,   # Maximum value constraint
    )
    explanation: Optional[str] = Field(
        None, 
        description="An explanation and the reasoning behind your decision.",
        min_length=1
    )
    feedback: Optional[str] = Field(
        None, 
        description="Feedback to the alternative branch.",
        min_length=1
    )

    @field_validator('value')
    def validate_value_range(cls, v: int) -> int:
        """Ensure value is within valid range and provide meaningful error message."""
        try:
            v = int(v)  # Ensure value is an integer
            if not (-100 <= v <= 100):
                raise ValueError(f"Reward value must be between -100 and 100, got {v}")
            return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value format: {str(e)}")

    @classmethod
    def create_fallback(cls, error_message: str, value: int = -100) -> "Reward":
        """
        Create a fallback reward with a specified value and explanation.
        
        Args:
            error_message: Description of what went wrong
            value: Optional custom fallback value (default: -100)
        """
        logger.warning(f"Creating fallback reward due to: {error_message}")
        return cls(
            value=max(-100, min(100, value)),  # Ensure value is within bounds
            explanation=f"Fallback reward created due to: {error_message}",
            feedback="System encountered an error. Consider retrying with a different approach."
        )

    @classmethod
    def extract_content_from_response(cls, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the content from a nested response structure.
        
        Handles various API response formats including OpenAI, Azure, and custom implementations.
        """
        try:
            # Handle OpenAI-style response format
            if isinstance(data, dict):
                # Check for choices array
                if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                    choice = data['choices'][0]
                    
                    # Handle different response structures
                    if isinstance(choice, dict):
                        # Standard OpenAI format
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                        # Direct content format
                        elif 'content' in choice:
                            content = choice['content']
                        # Text/completion format
                        elif 'text' in choice:
                            content = choice['text']
                        else:
                            return None
                        
                        # Parse content if it's a string
                        if isinstance(content, str):
                            try:
                                return json.loads(content)
                            except json.JSONDecodeError:
                                # Try to extract value using regex if JSON parsing fails
                                value_match = re.search(r'value["\s:]+(-?\d+)', content)
                                if value_match:
                                    return {
                                        "value": int(value_match.group(1)),
                                        "explanation": content,
                                        "feedback": "Extracted from non-JSON response"
                                    }
                        elif isinstance(content, dict):
                            return content
                
                # Handle direct response format
                if 'value' in data:
                    return data
                
            return None
        except Exception as e:
            logger.error(f"Failed to extract content: {e}")
            return None

    @staticmethod
    def safe_int_conversion(value: Any) -> Optional[int]:
        """
        Safely convert various types to integer.
        
        Handles strings, floats, Decimals, and other numeric types.
        """
        try:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                cleaned = re.sub(r'[^\d.-]', '', value)
                # Handle decimal numbers
                if '.' in cleaned:
                    return int(float(cleaned))
                return int(cleaned)
            if isinstance(value, Decimal):
                return int(value)
            return None
        except (ValueError, TypeError, InvalidOperation):
            return None

    @classmethod
    def from_response(cls, response: Union[str, Dict[str, Any], BaseModel, Any]) -> "Reward":
        """
        Parse a completion response into a Reward object.
        
        Args:
            response: Can be one of:
                - str: JSON string
                - dict: Dictionary with reward data
                - BaseModel: Pydantic model with reward data
                - Any: Other response type that will be converted appropriately
        
        Returns:
            Reward object
        """
        try:
            # Log the response type and content for debugging
            logger.debug(f"Processing response of type {type(response)}: {response}")

            # Handle None or empty response
            if response is None or (isinstance(response, str) and not response.strip()):
                return cls.create_fallback("Empty or None response received")

            # Return existing Reward instance
            if isinstance(response, cls):
                return response
            
            # Convert response to dictionary format
            try:
                if isinstance(response, BaseModel):
                    data = response.model_dump()
                elif isinstance(response, str):
                    data = json.loads(response)
                elif isinstance(response, dict):
                    data = response
                else:
                    data = json.loads(str(response))
                
                # Extract content from nested structure
                if isinstance(data, dict):
                    extracted_content = cls.extract_content_from_response(data)
                    if extracted_content is not None:
                        data = extracted_content

            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                logger.error(f"Response content: {response}")
                return cls.create_fallback(f"Failed to parse response: {str(e)}")

            # Validate data type
            if not isinstance(data, dict):
                return cls.create_fallback(f"Expected dict, got {type(data)}")

            # Try to find and convert value
            value = None
            if 'value' in data:
                value = cls.safe_int_conversion(data['value'])
            else:
                # Search nested dictionaries for value
                for k, v in data.items():
                    if isinstance(v, dict) and 'value' in v:
                        value = cls.safe_int_conversion(v['value'])
                        if value is not None:
                            data = v
                            break
                    elif k.lower() == 'value':
                        value = cls.safe_int_conversion(v)
                        if value is not None:
                            data = {'value': value}
                            break

            if value is None:
                logger.error(f"Missing or invalid value field in data: {data}")
                return cls.create_fallback("Missing or invalid value field")

            # Create Reward instance with validated data
            try:
                return cls(
                    value=value,
                    explanation=str(data.get('explanation', "No explanation provided")),
                    feedback=str(data.get('feedback', "No feedback provided"))
                )
            except Exception as e:
                logger.error(f"Failed to create Reward instance: {str(e)}")
                logger.error(f"Data: {data}")
                return cls.create_fallback(f"Failed to create Reward instance: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error in from_response: {e}")
            logger.error(f"Response: {response}")
            return cls.create_fallback(f"Unexpected error: {str(e)}")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "value": 50,
                "explanation": "The action successfully addressed the core issue...",
                "feedback": "Consider exploring alternative approaches such as..."
            }],
            "description": "A structured output for providing reward values and feedback for actions."
        }
    }

# Test cases (for reference)
def test_reward_parsing():
    # Test JSON string
    json_str = '{"value": 50, "explanation": "test", "feedback": "test"}'
    assert isinstance(Reward.from_response(json_str), Reward)

    # Test dict
    dict_data = {"value": 50, "explanation": "test", "feedback": "test"}
    assert isinstance(Reward.from_response(dict_data), Reward)

    # Test BaseModel
    class MockModelResponse(BaseModel):
        value: int
        explanation: Optional[str]
        feedback: Optional[str]
    
    model_response = MockModelResponse(value=50, explanation="test", feedback="test")
    assert isinstance(Reward.from_response(model_response), Reward)

    # Test invalid input
    invalid_response = "not json"
    fallback = Reward.from_response(invalid_response)
    assert fallback.value == -100
    assert "Failed to parse" in fallback.explanation

    print("All test cases passed!")

if __name__ == "__main__":
    test_reward_parsing()




