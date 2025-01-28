import sqlite3
import pandas as pd
import re

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def clean_data():
    # Load data from SQLite
    conn = sqlite3.connect("imdb_reviews.db")
    data = pd.read_sql_query("SELECT * FROM imdb_reviews", conn)
    conn.close()
    
    # Drop duplicates
    data.drop_duplicates(subset="review_text", inplace=True)
    
    # Clean the text
    data['cleaned_text'] = data['review_text'].apply(clean_text)
    
    # Save cleaned data back to SQLite
    conn = sqlite3.connect("imdb_reviews_cleaned.db")
    data.to_sql("imdb_reviews", conn, if_exists="replace", index=False)
    print("Cleaned data saved to imdb_reviews_cleaned.db")
    conn.close()
    
    return data

def perform_eda(data):
    """Perform exploratory data analysis (EDA) on the dataset."""
    # Distribution of reviews per sentiment
    sentiment_counts = data['sentiment'].value_counts()
    print("Number of reviews per sentiment:")
    print(sentiment_counts)
    
    
    # Average review length for positive vs. negative
    data['review_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
    avg_lengths = data.groupby('sentiment')['review_length'].mean()
    print("\nAverage review length per sentiment:")
    print(avg_lengths)
    

if __name__ == "__main__":
    # Clean the data
    cleaned_data = clean_data()
    
    # Perform EDA
    perform_eda(cleaned_data)
