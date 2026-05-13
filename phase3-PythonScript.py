"""
Sentiment Prediction Script
============================
"""

import re
import pickle
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Download required NLTK data (only needed once) ──────────────────────────
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource
                           else f'corpora/{resource}')
        except LookupError:
            print(f"   Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)


# ─── Load saved model and transformers ──────────────────────────────────────
def load_models():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            print('Opened1')
            tfidf = pickle.load(f)
            print('Load successfully1')
        with open('selector.pkl', 'rb') as f:
            print('Opened2')
            selector = pickle.load(f)
            print('Load successfully2')
        with open('random_forest_model.pkl', 'rb') as f:
            print('Opened3')
            model = pickle.load(f)
            print('Load successfully3')
        return tfidf, selector, model
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Please run the save cell at the end of your notebook first.")
        exit(1)


# ─── Preprocessing Pipeline (mirrors PBL_Phase1.ipynb exactly) ──────────────
def preprocess(summary: str, text: str) -> str:

    # Step 1: Combine Summary + Text (same as notebook)
    if summary.strip():
        review = summary.strip() + " " + text.strip()
    else:
        review = text.strip()

    # Step 2: Remove HTML tags
    clean = re.compile('<.*?>')
    review = re.sub(clean, '', review)

    # Step 3: Remove punctuation — keep only letters a-z A-Z
    review = re.sub('[^a-zA-Z]', ' ', review)

    # Step 4: Convert to lowercase
    review = review.lower()

    # Step 5: Tokenize
    tokens = word_tokenize(review)

    # Step 6: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # Step 7: Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Step 8: Join back to string
    processed_text = " ".join(tokens)

    return processed_text


# ─── Predict sentiment ───────────────────────────────────────────────────────
def predict_sentiment(summary: str, text: str, tfidf, selector, model) -> dict:

    if not text.strip():
        return {'error': True,
                'message': 'Review text cannot be empty. Please enter your review.'}

    processed = preprocess(summary, text)

    if not processed.strip():
        return {'error': True,
                'message': 'After preprocessing, no meaningful words were found. '
                           'Please write a more detailed review.'}

    # TF-IDF vectorize  →  sparse matrix, shape (1, n_tfidf_features)
    X = tfidf.transform([processed])

    # ── Smart feature routing ────────────────────────────────────────────────
    # Apply selector ONLY if the model was trained on selected features.
    # Compare model.n_features_in_ against both options and pick the match.
    n_full     = X.shape[1]                        # e.g. 1001
    X_selected = selector.transform(X)             # e.g. 1000
    n_selected = X_selected.shape[1]

    expected = getattr(model, 'n_features_in_', None)

    if expected == n_selected:
        X_input = X_selected          # model trained WITH selector (DT, etc.)
    elif expected == n_full:
        X_input = X                   # model trained WITHOUT selector (RF, etc.)
    else:
        # Last resort: try full, then selected
        try:
            model.predict(X)
            X_input = X
        except Exception:
            X_input = X_selected

    prediction    = model.predict(X_input)[0]
    probabilities = None

    try:
        proba       = model.predict_proba(X_input)[0]
        classes     = model.classes_
        probabilities = {cls: round(float(p) * 100, 1)
                         for cls, p in zip(classes, proba)}
    except Exception:
        pass

    return {
        'error':         False,
        'sentiment':     prediction,
        'probabilities': probabilities,
        'processed':     processed
    }
# ─── Display result ──────────────────────────────────────────────────────────
def display_result(result: dict):
    sentiment_emoji = {
        'Positive': ' POSITIVE',
        'Negative': ' NEGATIVE',
        'Neutral':  ' NEUTRAL'
    }

    print("\n" + "=" * 50)
    print("          SENTIMENT PREDICTION RESULT")
    print("=" * 50)

    if result.get('error'):
        print(f"\n  {result['message']}")
        return

    label = result['sentiment']
    print(f"\n  Sentiment : {sentiment_emoji.get(label, label)}")

    if result['probabilities']:
        print(f"\n  Confidence Scores:")
        for cls, pct in sorted(result['probabilities'].items(),
                                key=lambda x: x[1], reverse=True):
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"    {cls:<10} {bar}  {pct:.1f}%")

    print(f"\n  Processed Text (first 100 chars):")
    print(f"    {result['processed'][:100]}...")
    print("=" * 50)


# ─── Main interactive loop ───────────────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print("   Amazon Review Sentiment Predictor")
    print("   Model: Decision Tree")
    print("=" * 50)

    print("\n⏳ Loading models...")
    download_nltk_resources()
    tfidf, selector, model = load_models()
    print(" Models loaded successfully!\n")

    while True:
        print("\n" + "-" * 50)
        print("Enter your review details below.")
        print("(Type 'quit' or 'exit' at any prompt to stop)\n")

        # ── Get Summary (optional) ──────────────────────────────────────────
        summary = input("📝 Review Summary (optional — press Enter to skip): ").strip()
        if summary.lower() in ('quit', 'exit'):
            print("\n👋 Goodbye!")
            break

        # ── Get Review Text (required) ──────────────────────────────────────
        print("\n📄 Review Text (required):")
        text = input("   > ").strip()
        if text.lower() in ('quit', 'exit'):
            print("\n👋 Goodbye!")
            break

        # ── Predict and display ─────────────────────────────────────────────
        result = predict_sentiment(summary, text, tfidf, selector, model)
        display_result(result)

        # ── Ask to continue ─────────────────────────────────────────────────
        again = input("\n Predict another review? (yes/no): ").strip().lower()
        if again not in ('yes', 'y'):
            print("\n Goodbye!")
            break


if __name__ == '__main__':
    main()
