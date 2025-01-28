import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    """Train a sentiment analysis model using cleaned data."""
    # Load the cleaned data
    conn = sqlite3.connect("imdb_reviews_cleaned.db")
    data = pd.read_sql_query("SELECT * FROM imdb_reviews", conn)
    conn.close()

    # Feature extraction
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()
    y = data['sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model and vectorizer
    with open("./models/model.pkl", "wb") as model_file:
        pickle.dump((model, vectorizer), model_file)
    print("Model and vectorizer saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
