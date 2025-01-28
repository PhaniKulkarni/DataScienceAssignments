#Project Overview
This repository contains two projects designed to perform text-based tasks:

DS1: Sentiment Analysis of IMDb Reviews

DS2: Document-based Retrieval-Augmented Generation (RAG)

#Project 1: Sentiment Analysis of IMDb Reviews
1. Files Explanation
   
   (a) ingestion.py:
      Downloads the IMDb dataset, processes it into a pandas DataFrame, and stores it in a SQLite database (imdb_reviews.db).
      
   (b) cleaning.py:
      Cleans raw text reviews by:
      Converting to lowercase
      Removing HTML tags
      Removing punctuation
      Saves the cleaned data into imdb_reviews_cleaned.db.

   (c) model_training.py:
      Trains a Naive Bayes sentiment analysis model using cleaned IMDb data.
      Saves the trained model and vectorizer as model.pkl in the models/ directory.

   (d) app.py:
      Provides a Flask-based API to serve predictions using the trained model.

2. Databases
     imdb_reviews.db: Contains raw data ingested from the IMDb dataset.
   
     imdb_reviews_cleaned.db: Contains cleaned and preprocessed data for training.

4. Execution Flow
     python ingestion.py
   
     python cleaning.py
   
     python model_training.py
   
     python app.py

6. Testing the API
Endpoint: http://127.0.0.1:5000/predict

Request Example (POST):

{
    "review_text": "This movie was amazing!"
}

______________________________________________________________________________________________________________________________________________________________________________
#Project 2: Document-based Retrieval-Augmented Generation (RAG)
Download the LLaMA 2 model file (llama-2-7b-chat.ggmlv3.q8_0.bin) and place it in the appropriate directory.

1. Files Explanation
   (a) data_preprocessing.py:
       Loads and preprocesses PDF files from the data/ folder.
       Splits documents into manageable chunks and cleans text for embedding.

   (b) data_embedding.py:
        Generates embeddings from preprocessed documents using the HuggingFace Embeddings model.
        Stores these embeddings in a FAISS vector store for efficient querying.
        Inside faiss path, index.pkl and index.faiss will be created

   (c) data_generation.py:
        Retrieves the most relevant documents using RetrieverQA.from_chain_type.
        Generates responses using the LLaMA 2 model.

   (d) flaskapp.py:
        Stores user and system interactions in an SQLite database.
        Provides endpoints for generating answers and retrieving chat history.

2. conversation_log.db:
    SQLite database to store chat history (auto-created when the app is run).

3. Execution Flow:
   
    Preprocess the data and generate embeddings:
      python data_embedding.py
   
    Run the Flask application:
      python ragapp.py

5. Test the API:
POST Request:
Endpoint: http://127.0.0.1:5000/generate
Example Body:
{
    "query": "type your query"
}
GET Request:
Endpoint: http://127.0.0.1:5000/history



