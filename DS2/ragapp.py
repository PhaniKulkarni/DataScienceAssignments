import os
import sqlite3
from flask import Flask, request, jsonify
from datetime import datetime
from data_generation import generate_response
from data_preprocessing import load_and_preprocess_pdfs
from data_embedding import store_embeddings_and_vectorstore
from data_embedding import load_vectorstore
# Initialize Flask ragapp
ragapp = Flask(__name__)


VECTORSTORE_PATH = "faiss_index"
DATABASE_PATH = "conversation_log.db"
DATA_FOLDER="data"

def init_db():
    if not os.path.exists(DATABASE_PATH):
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                role TEXT,
                content TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("Database and table created successfully.")
    else:
        print("Database already exists.")


def log_conversation(role, content):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversation_log (timestamp, role, content)
        VALUES (?, ?, ?)
    ''', (timestamp, role, content))
    conn.commit()
    conn.close()


def get_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, role, content FROM conversation_log ORDER BY timestamp ASC')
    rows = cursor.fetchall()
    conn.close()
    

    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'timestamp': row[1],
            'role': row[2],
            'content': row[3]
        })
    return history


@ragapp.route('/generate', methods=['POST'])
def generate():
    # Get the query from the POST request
    '''
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    '''
    try:
        # Log user query to the database
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        user_query = request.form.get('query')  # Get the query text from form data

        if not user_query:
            return jsonify({"error": "Query text is required"}), 400
        file_path = os.path.join(DATA_FOLDER, file.filename)
        file.save(file_path)
        log_conversation('user', user_query)
        documents = load_and_preprocess_pdfs(DATA_FOLDER)

        store_embeddings_and_vectorstore(documents, VECTORSTORE_PATH)

        # Generate the response using the generate_response function
        response = generate_response(user_query, VECTORSTORE_PATH)
        
        # Log system response to the database
        log_conversation('system', response)
        
        # Return the generated response as a JSON
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get the chat history
@ragapp.route('/history', methods=['GET'])
def history():
    try:
        chat_history = get_chat_history()
        return jsonify({"history": chat_history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Run the Flask ragapp
    ragapp.run(debug=False)
