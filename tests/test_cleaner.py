import pytest
from src.preprocessing.cleaner import clean_text


def test_cleaner_spanish_text():
    text = "¡Hola! El servicio es TERRIBLE 😡 https://ejemplo.com"
    result = clean_text(text, lang="es")
    assert "https://" not in result
    assert "😡" not in result
    assert result == "¡hola! el servicio es terrible"


def test_cleaner_portuguese_text():
    text = "Olá! O serviço é terrível 😡 www.exemplo.com"
    result = clean_text(text, lang="pt")
    assert "www." not in result
    assert "😡" not in result
    assert result == "olá! o serviço é terrível"


def test_cleaner_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_cleaner_accents_preserved():
    text = "áéíóú ç ã õ à â ê ô"
    result = clean_text(text)
    assert "áéíóú" in result
    assert "ç" in result


def test_cleaner_multiple_spaces():
    text = "esto   es   una    prueba"
    result = clean_text(text)
    assert "  " not in result
    assert result == "esto es una prueba"
