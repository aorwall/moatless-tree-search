import re
import logging
from typing import Optional
logger = logging.getLogger(__name__)

def parse_explanation(response_content, keyword='feedback') -> str:
    explanation_pattern = r'<{keyword}>\s*(.*?)\s*(?:</{keyword}>|<Feedback_to_Alternative_Branch>|<Reward>|$)'
    match = re.search(explanation_pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        return response_content


def parse_value(response_content, keyword='reward', allowed_values=None):
    """
    Parse the value associated with a given keyword from the LLM response content.

    Args:
    response_content (str): The content of the LLM response.
    keyword (str): The keyword to search for (default: 'reward').
    allowed_values (list or range, optional): A list or range of allowed values.

    Returns:
    int: The parsed integer value, or None if not found, not an integer, or not in allowed_values.
    """
    value_patterns = [
        fr'<\s*{keyword}\s*>\s*:?\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'<\s*{keyword}\s*>(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'{keyword}:\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'\*\*{keyword}\*\*\s*:?\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'\*\*{keyword.capitalize()}\*\*\s*:?\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'{keyword.capitalize()}:\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'<\s*{keyword.capitalize()}\s*>\s*:?\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'\*\*<\s*{keyword.capitalize()}\s*>\*\*:\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'\*\*{keyword.capitalize()}:\*\*\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'<\s*{keyword}\s*>\s*(?:[Nn]ode[_\s-]?)?(-?\d+)\s*</\s*{keyword}\s*>',
        fr'<\s*{keyword}\s*>\s*(?:[Nn]ode[_\s-]?)?(-?\d+)',
        fr'<\s*{keyword}\s*>\s*:?\s*(-?\d+)',
        fr'{keyword}\s*:?\s*(-?\d+)',
    ]

    matched_value = None
    try:
        # Try to find value using specific patterns
        for pattern in value_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
            if match:
                matched_value = match.group(1).strip()
                value = int(matched_value)
                if allowed_values is None or value in allowed_values:
                    return value

        # If no pattern matches, look for any number after the keyword
        general_pattern = fr'{keyword}\s*:?\s*(?:[Nn]ode[_\s-]?)?(-?\d+)'
        match = re.search(general_pattern, response_content, re.IGNORECASE | re.DOTALL)
        if match:
            matched_value = match.group(1).strip()
            value = int(matched_value)
            if allowed_values is None or value in allowed_values:
                return value

        # If we reach here, either no value was found or it wasn't an integer
        logger.warning(f"No valid integer {keyword} found in the response content.")
        logger.warning(f"Response content: {response_content}")
        return None
    except ValueError:
        logger.warning(f"Found value {matched_value} at {keyword}, but it's not a valid integer.")
        return None
    except Exception as e:
        logger.error(f"Error parsing {keyword}: {e}")
        return None

def parse_node_id(response_content) -> Optional[int]:
    """
    Parse the node_id from the LLM response content.
    Looks for patterns like:
    - "expand node 5"
    - "node_id: 5"
    - "Node ID: 5"
    - "suggested node: 5"
    - Or any JSON structure containing node_id or suggested_node_id
    
    Args:
    response_content (str): The content of the LLM response.
    
    Returns:
    int: The parsed node ID, or None if not found or not a valid integer.
    """
    node_patterns = [
        r'expand\s*node\s*(\d+)',
        r'node_id\s*:?\s*(\d+)',
        r'node\s*id\s*:?\s*(\d+)',
        r'suggested\s*node\s*:?\s*(\d+)',
        r'suggested_node_id\s*:?\s*(\d+)',
        r'expand\s*node\s*#?(\d+)',
        r'node\s*#?(\d+)',
        # JSON-like patterns
        r'"node_id"\s*:\s*(\d+)',
        r'"suggested_node_id"\s*:\s*(\d+)',
    ]

    matched_value = None
    try:
        # Try each pattern
        for pattern in node_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
            if match:
                matched_value = match.group(1).strip()
                return int(matched_value)

        # If no match found
        logger.debug("No valid node ID found in the response content.")
        return None
        
    except ValueError:
        logger.warning(f"Found value {matched_value}, but it's not a valid integer node ID.")
        return None
    except Exception as e:
        logger.error(f"Error parsing node ID: {e}")
        return None