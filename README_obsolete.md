# IMPORTANTE: La rama 'fasttext-refactor' está OBSOLETA

## ¿Qué pasó?

La rama `fasttext-refactor` se renombró a **`tfidf-logreg`** porque:
1. FastText NO funciona con Python 3.14+ (bug conocido)
2. El nuevo nombre es más honesto: usamos **TF-IDF + Logistic Regression**, NO FastText

## ¿Qué deberías usar?

| Rama | Modelo Real | ¿FastText? |
|------|-------------|------------|
| `main` | BERT (HuggingFace) | ❌ NO |
| `tfidf-logreg` | TF-IDF + LogisticRegression | ❌ NO |

## ¿Cómo proceder?

```bash
# Clonar y usar la rama correcta:
git clone https://github.com/Rox-0864/sentiment_analysis.git
cd sentiment_analysis
git checkout tfidf-logreg
```

**NO uses `fasttext-refactor`** — ¡está obsoleta y no se puede borrar!
