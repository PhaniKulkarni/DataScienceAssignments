import sqlite3
import pandas as pd
from datasets import load_dataset

def ingest_data():
    # Load the IMDB dataset
    print("Downloading the IMDB dataset...")
    dataset = load_dataset("imdb")
    print(dataset)
    # Convert to pandas DataFrame
    train_data = pd.DataFrame(dataset['train'])
    print("TRAIN DATA",train_data)
    test_data = pd.DataFrame(dataset['test'])
    print("TEST DATA",test_data)
    # Combine train and test sets
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    
    full_data.rename(columns={"text": "review_text", "label": "sentiment"}, inplace=True)
    print(full_data)
    # Save to SQLite
    print("Inserting data into the database...")
    conn = sqlite3.connect("imdb_reviews.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS imdb_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            sentiment TEXT
        )
    """)
    
    full_data.to_sql("imdb_reviews", conn, if_exists="replace", index=False)
    print("Data successfully ingested into the database.")
    
    conn.close()

if __name__ == "__main__":
    ingest_data()
