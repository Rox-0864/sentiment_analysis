from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.lang.en import English


def detect_lang(text: str) -> str:
    """
    Detect if text is Spanish (es), Portuguese (pt), or unknown.
    Uses simple heuristic: check for common stopwords.
    """
    text_lower = text.lower()
    
    es_words = {"el", "la", "de", "que", "y", "en", "un", "por", "con", "no", "una", "los", "las"}
    pt_words = {"o", "a", "de", "que", "e", "em", "um", "por", "com", "não", "uma", "os", "as"}
    
    words = set(text_lower.split())
    
    es_score = len(words & es_words)
    pt_score = len(words & pt_words)
    
    if es_score > pt_score and es_score > 0:
        return "es"
    elif pt_score > es_score and pt_score > 0:
        return "pt"
    elif es_score == pt_score and es_score > 0:
        return "es"
    
    return "unknown"
