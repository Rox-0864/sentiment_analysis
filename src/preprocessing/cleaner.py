import re
import unicodedata


def clean_text(text: str, lang: str = "es") -> str:
    """
    Clean and normalize text for ES/PT.
    
    - Removes URLs, emojis, special characters
    - Lowercases text
    - Preserves accents
    - Normalizes whitespace
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Remove emojis (keep letters, numbers, punctuation, accents)
    text = re.sub(r"[^\w\s\.,;:!?¿¡áéíóúüñçãõáàâêôãõ]", "", text, flags=re.UNICODE)
    
    # Normalize unicode (preserve accents, normalize spaces)
    text = unicodedata.normalize("NFC", text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
