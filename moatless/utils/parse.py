import re
import logging

logger = logging.getLogger(__name__)

def parse_explanation(response_content) -> str:
    explanation_pattern = r'<Explanation>\s*(.*?)\s*(?:</Explanation>|<Feedback_to_Alternative_Branch>|<Reward>|$)'
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
        fr'<\s*{keyword}\s*>\s*:?\s*(-?\d+)',
        fr'<\s*{keyword}\s*>(-?\d+)',
        fr'{keyword}:\s*(-?\d+)',
        fr'\*\*{keyword}\*\*\s*:?\s*(-?\d+)',
        fr'\*\*{keyword.capitalize()}\*\*\s*:?\s*(-?\d+)',
        fr'{keyword.capitalize()}:\s*(-?\d+)',
        fr'<\s*{keyword.capitalize()}\s*>\s*:?\s*(-?\d+)',
        fr'\*\*<\s*{keyword.capitalize()}\s*>\*\*:\s*(-?\d+)',
        fr'\*\*{keyword.capitalize()}:\*\*\s*(-?\d+)',
        fr'<\s*{keyword}\s*>\s*(-?\d+)\s*</\s*{keyword}\s*>',
        fr'<\s*{keyword}\s*>\s*(-?\d+)'
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
        general_pattern = fr'{keyword}\s*:?\s*(-?\d+)'
        match = re.search(general_pattern, response_content, re.IGNORECASE | re.DOTALL)
        if match:
            matched_value = match.group(1).strip()
            value = int(matched_value)
            if allowed_values is None or value in allowed_values:
                return value

        # If we reach here, either no value was found or it wasn't an integer
        logger.warning(f"No valid integer {keyword} found in the response content.")
        return None
    except ValueError:
        logger.warning(f"Found value {matched_value} at {keyword}, but it's not a valid integer.")
        return None
    except Exception as e:
        logger.error(f"Error parsing {keyword}: {e}")
        return None
