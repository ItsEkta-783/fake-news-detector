import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources (safe to call multiple times)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize once at module level for efficiency
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw news article text.

    Pipeline:
        1. Lowercase
        2. Remove URLs
        3. Remove punctuation and digits
        4. Tokenize (split on whitespace)
        5. Remove English stopwords
        6. Apply Porter Stemming
        7. Rejoin tokens into a single string

    Args:
        text (str): Raw input text (title + article body)

    Returns:
        str: Cleaned, stemmed text ready for TF-IDF vectorization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase — normalize case
    text = text.lower()

    # 2. Remove URLs — they add noise without semantic value
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove everything except lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Tokenize
    tokens = text.split()

    # 5. Remove stopwords (the, is, at, etc.)
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Stem each token to its root form
    tokens = [stemmer.stem(word) for word in tokens]

    # 7. Rejoin into a string (TF-IDF expects strings, not lists)
    return ' '.join(tokens)