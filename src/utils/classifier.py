import re

CONVERSATIONAL_REGEX = re.compile(
    r"^(hi|hello|hey|greetings|how are you|who are you|what are you|thanks|thank you|bye|goodbye|good morning|good afternoon|good evening|ok|okay)\b",
    re.IGNORECASE
)

APP_INFO_REGEX = re.compile(
    r"(who are you|what can you help|what do you do|how does this app work|what is this app|what are your capabilities|tum kya karte ho|what are you)",
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
        
    if APP_INFO_REGEX.search(query_clean):
        return "conversational"
        
    # Match against common conversational starters
    if CONVERSATIONAL_REGEX.match(query_clean):
        # Ensure it's not a complex command masquerading as a greeting
        # e.g., "Hello, please summarize the Apple Q3 earnings" -> data
        words = query_clean.split()
        if len(words) <= 15:
            return "conversational"
            
    return "data"
