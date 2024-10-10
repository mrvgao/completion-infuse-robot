import re


def parse_next_action(text):
    # Regular expression to match "Next Action" whether or not it's surrounded by **
    match = re.search(r'(?:\*\*)?\s*Next Action\s*:\s*(.*?)\s*(?:\*\*|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None