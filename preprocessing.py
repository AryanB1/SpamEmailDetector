import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def clean_text(text):
    """Clean and prepare text data by removing special characters, lowercasing, and removing stopwords"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens)

def load_and_preprocess_data(path, test_size=0.2, random_state=42, max_features=1000):
    """Load email data, preprocess it, and split into training and test sets"""
    # Load the data with the correct column names
    df = pd.read_csv(path, encoding="ISO-8859-1")
    
    # Handle possible different column names in the CSV
    if "text" in df.columns and "spam" in df.columns:
        text_col, label_col = "text", "spam"
    elif "Message" in df.columns and "Category" in df.columns:
        text_col, label_col = "Message", "Category"
        # Convert 'ham'/'spam' to numeric if needed
        if df[label_col].dtype == 'object':
            df["spam"] = df[label_col].map({"ham": 0, "spam": 1})
            label_col = "spam"
    else:
        # If columns don't match expected patterns, use first two columns
        text_col, label_col = df.columns[1], df.columns[0]
        print(f"Using columns: {text_col} for text and {label_col} for labels")
        print(f"Available columns in the dataset: {list(df.columns)}")
        raise KeyError("The dataset does not contain the expected columns for text and labels.")
    
    # Clean the text data
    df["cleaned_text"] = df[text_col].apply(clean_text)
    
    # Vectorize the text using TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df["cleaned_text"]).toarray()
    
    # Get the labels
    y = df[label_col].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
