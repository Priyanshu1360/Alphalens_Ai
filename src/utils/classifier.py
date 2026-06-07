import re

CONVERSATIONAL_REGEX = re.compile(
    r"^(hi|hello|hey|greetings|how are you|who are you|what are you|thanks|thank you|bye|goodbye|good morning|good afternoon|good evening|ok|okay)\b",
    re.IGNORECASE
)

def classify_intent(query: str) -> str:
    """
    Classifies the user query as 'conversational' or 'data'.
    Uses regex heuristics for extreme low latency.
    """
    query_clean = (query or "").strip()
    
    if not query_clean:
        return "data"
        
    # Match against common conversational starters
    if CONVERSATIONAL_REGEX.match(query_clean):
        # Ensure it's not a complex command masquerading as a greeting
        # e.g., "Hello, please summarize the Apple Q3 earnings" -> data
        words = query_clean.split()
        if len(words) <= 8:
            return "conversational"
            
    return "data"
